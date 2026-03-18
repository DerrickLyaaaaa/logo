import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve

from .knn_graph import HAS_FAISS, stable_knn_adjacency
from .rbo import rbo_item_scores

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def _l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def _kmeans(x, n_clusters, seed=0, niter=100):
    x = np.ascontiguousarray(x.astype(np.float32))
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0],), dtype=np.int64), x.copy()
    n_clusters = int(max(2, min(n_clusters, x.shape[0])))

    if HAS_FAISS and faiss is not None:
        km = faiss.Kmeans(d=x.shape[1], k=n_clusters, niter=niter, verbose=False, seed=seed)
        km.train(x)
        _, labels = km.index.search(x, 1)
        centers = km.centroids
        return labels.reshape(-1), centers

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_
    return labels.astype(np.int64), centers.astype(np.float32)


def _minmax01(x):
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _top2_margin(scores):
    scores = np.asarray(scores, dtype=np.float32)
    n, c = scores.shape
    if c <= 1:
        return np.clip(scores.reshape(n), 0.0, 1.0)
    top2 = np.argsort(-scores, axis=1)[:, :2]
    top1 = scores[np.arange(n), top2[:, 0]]
    top2v = scores[np.arange(n), top2[:, 1]]
    return np.clip(top1 - top2v, 0.0, 1.0)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _apply_local_score_variant(s_loc, m_loc, local_stats, args):
    """
    A0/A1/A2/A3 local-score variants:
      A0: s = top1
      A1: s = top1 * conc
      A2: s = top1 * margin
      A3: s = top1 * margin * conc
    """
    s_loc = np.clip(np.asarray(s_loc, dtype=np.float32).reshape(-1), 0.0, 1.0)
    m_loc = np.clip(np.asarray(m_loc, dtype=np.float32).reshape(-1), 0.0, 1.0)
    n = int(s_loc.shape[0])
    top1 = np.clip(np.asarray(local_stats.get("top1", s_loc), dtype=np.float32).reshape(-1), 0.0, 1.0)
    if top1.shape[0] != n:
        top1 = s_loc
    margin = np.clip(np.asarray(local_stats.get("margin", m_loc), dtype=np.float32).reshape(-1), 0.0, 1.0)
    if margin.shape[0] != n:
        margin = m_loc
    conc = np.clip(np.asarray(local_stats.get("conc", np.ones(n, dtype=np.float32)), dtype=np.float32).reshape(-1), 0.0, 1.0)
    if conc.shape[0] != n:
        conc = np.ones(n, dtype=np.float32)

    variant = str(getattr(args, "local_score_variant", "a0")).lower()
    if variant not in {"a0", "a1", "a2", "a3"}:
        variant = "a0"

    if variant == "a0":
        s_new = top1
    else:
        if variant == "a1":
            base_factor = np.clip(conc, 0.0, 1.0)
        elif variant == "a2":
            base_factor = np.clip(margin, 0.0, 1.0)
        else:
            base_factor = np.clip(margin * conc, 0.0, 1.0)

        if bool(getattr(args, "local_score_soft_enable", False)):
            lam = float(np.clip(getattr(args, "local_score_soft_lambda", 0.75), 0.0, 1.0))
            alpha = float(max(getattr(args, "local_score_margin_alpha", 0.5), 1e-6))
            beta = float(max(getattr(args, "local_score_conc_beta", 0.5), 1e-6))
            entropy = np.clip(
                np.asarray(local_stats.get("entropy", np.zeros(n, dtype=np.float32)), dtype=np.float32).reshape(-1),
                0.0,
                1.0,
            )
            if entropy.shape[0] != n:
                entropy = np.zeros(n, dtype=np.float32)

            if variant == "a1":
                shaped = np.power(np.clip(conc, 0.0, 1.0), beta)
            elif variant == "a2":
                shaped = np.power(np.clip(margin, 0.0, 1.0), alpha)
            else:
                shaped = np.power(np.clip(margin, 0.0, 1.0), alpha) * np.power(np.clip(conc, 0.0, 1.0), beta)
            shaped = np.clip(shaped, 0.0, 1.0)

            mult = lam + (1.0 - lam) * shaped

            # Apply soft penalty only in uncertain (higher-entropy) region.
            ent_thr = float(np.clip(getattr(args, "local_score_entropy_gate", 0.55), 0.0, 1.0))
            uncertain_mask = entropy >= ent_thr
            mult = np.where(uncertain_mask, mult, 1.0).astype(np.float32)

            # Protect high-confidence local points from over-penalization.
            protect_q = float(np.clip(getattr(args, "local_score_protect_q", 0.60), 0.0, 1.0))
            protect_min_n = int(max(1, getattr(args, "local_score_protect_min_n", 16)))
            if n >= protect_min_n:
                top1_thr = float(np.quantile(top1, protect_q))
                protect_mask = top1 >= top1_thr
                mult = np.where(protect_mask, 1.0, mult).astype(np.float32)

            s_new = top1 * np.clip(mult, 0.0, 1.0)
        else:
            s_new = top1 * base_factor

    return np.clip(s_new, 0.0, 1.0).astype(np.float32), variant


def _softmax_temp(logits, temp):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.size == 0:
        return logits
    t = float(max(temp, 1e-6))
    z = logits / t
    z = z - float(np.max(z))
    ez = np.exp(np.clip(z, -60.0, 60.0))
    den = float(np.sum(ez))
    if den <= 0.0:
        return np.full_like(logits, 1.0 / max(1, logits.size), dtype=np.float32)
    return (ez / den).astype(np.float32)


def _ess_from_weights(weights):
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return float((s1 * s1) / max(s2, 1e-12))


def _weights_with_ess_target(raw_scores, target_ess, temp_min=0.05, temp_max=2.0, n_iter=24):
    """
    Convert reliability scores to soft weights with ESS approximately matching target_ess.
    Returns (weights, ess, used_temp), where weights have mean~=1.
    """
    s = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    n = int(s.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32), 0.0, 1.0
    if n == 1:
        return np.ones(1, dtype=np.float32), 1.0, 1.0

    target = float(np.clip(target_ess, 1.0, float(n)))
    t_lo = float(max(temp_min, 1e-4))
    t_hi = float(max(temp_max, t_lo + 1e-6))

    a_lo = _softmax_temp(s, t_lo)
    ess_lo = _ess_from_weights(a_lo)
    a_hi = _softmax_temp(s, t_hi)
    ess_hi = _ess_from_weights(a_hi)

    if ess_lo >= target:
        a = a_lo
        t_use = t_lo
    elif ess_hi <= target:
        a = a_hi
        t_use = t_hi
    else:
        lo, hi = t_lo, t_hi
        a = a_hi
        t_use = hi
        for _ in range(int(max(1, n_iter))):
            mid = 0.5 * (lo + hi)
            a_mid = _softmax_temp(s, mid)
            ess_mid = _ess_from_weights(a_mid)
            if ess_mid < target:
                lo = mid
            else:
                hi = mid
                a = a_mid
                t_use = mid

    # Keep average magnitude around 1.0 for downstream weighting while preserving ESS.
    w = (a * float(n)).astype(np.float32)
    ess = _ess_from_weights(w)
    return w, float(ess), float(t_use)


def _rank01(values, descending=False):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = values.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(-values if descending else values)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    ranks /= float(n - 1)
    return ranks


def _rank_corr01(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape[0] != b.shape[0] or a.shape[0] <= 2:
        return 1.0
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a < 1e-12 or std_b < 1e-12:
        return 1.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))


def _neg_stability_frequency(
    h,
    p_star_flat,
    s_loc,
    y_cluster,
    k_pool,
    use_joint_rank,
    neg_beta_effective,
    args,
    component_to_class=None,
    num_classes=None,
):
    """
    Estimate per-sample pseudo-negative stability:
    stab(i) = frequency that i appears in top-k_pool OOD-suspicious set across perturbation views.
    """
    n = int(h.shape[0])
    k_pool = int(max(1, min(int(k_pool), n)))
    views = int(max(1, getattr(args, "neg_stab_views", 8)))
    if views <= 1:
        return np.ones(n, dtype=np.float32), 1

    noise_std = float(max(getattr(args, "neg_stab_noise_std", 0.01), 0.0))
    drop_prob = float(np.clip(getattr(args, "neg_stab_drop_prob", 0.0), 0.0, 0.95))
    gamma = float(np.clip(getattr(args, "neg_rank_gamma", 0.6), 0.0, 1.0))
    y_cluster = np.asarray(y_cluster, dtype=np.int64).reshape(-1)
    s_loc = np.asarray(s_loc, dtype=np.float32).reshape(-1)

    counts = np.zeros(n, dtype=np.float32)
    for ridx in range(views):
        if ridx == 0:
            h_aug = h
        else:
            h_aug = h.copy()
            if noise_std > 0.0:
                h_aug = h_aug + np.random.randn(*h_aug.shape).astype(np.float32) * noise_std
            if drop_prob > 0.0:
                keep = (np.random.rand(*h_aug.shape) >= drop_prob).astype(np.float32)
                h_aug = h_aug * keep
            h_aug = _l2_normalize(h_aug)

        sims_proto = h_aug @ p_star_flat.T
        sims_cls = _aggregate_component_sims(
            sims_proto,
            component_to_class=component_to_class,
            num_classes=num_classes,
        )
        s_glo_aug = np.max(sims_cls, axis=1)

        if use_joint_rank:
            dist_aug = np.sqrt(np.clip(2.0 - 2.0 * s_glo_aug, 0.0, None)).astype(np.float32)
            pred_aug = np.argmax(sims_cls, axis=1)
            disagree_aug = (pred_aug != y_cluster).astype(np.float32)
            rank_s = _rank01(s_glo_aug, descending=False)
            rank_d = _rank01(dist_aug, descending=True)
            score_susp = 1.0 - rank_s
            dist_susp = 1.0 - rank_d
            neg_score = gamma * score_susp + (1.0 - gamma) * dist_susp + float(neg_beta_effective) * disagree_aug
            idx = np.argsort(-neg_score)[:k_pool]
        else:
            # For non-joint mode, use score-only perturbation ranking with a light local prior.
            rank_g = _rank01(s_glo_aug, descending=False)
            rank_l = _rank01(s_loc, descending=False)
            neg_score = 0.5 * (1.0 - rank_g) + 0.5 * (1.0 - rank_l)
            idx = np.argsort(-neg_score)[:k_pool]
        counts[idx] += 1.0

    return np.clip(counts / float(views), 0.0, 1.0).astype(np.float32), int(views)


def _aggregate_component_sims(sims, component_to_class=None, num_classes=None):
    sims = np.asarray(sims, dtype=np.float32)
    if component_to_class is None:
        return sims
    comp = np.asarray(component_to_class, dtype=np.int64).reshape(-1)
    if comp.shape[0] != sims.shape[1]:
        raise ValueError("component_to_class size mismatch with prototype similarity matrix")
    c_out = int(num_classes) if num_classes is not None else int(np.max(comp) + 1)
    out = np.full((sims.shape[0], c_out), -1e9, dtype=np.float32)
    for c in range(c_out):
        cols = np.where(comp == c)[0]
        if cols.size > 0:
            out[:, c] = np.max(sims[:, cols], axis=1)
    return out


def _init_multi_prototypes(h, p0, args, y_cluster=None):
    """
    Initialize per-class multi-prototype bank from pseudo-ID features.
    Returns [C, M, D], mixture used flag, and per-class multi-proto mask.
    """
    p0 = np.asarray(p0, dtype=np.float32)
    c_in, d = p0.shape
    m_proto = int(max(1, getattr(args, "glo_num_proto_per_class", 1)))
    if m_proto <= 1:
        return p0[:, None, :], False, np.zeros(c_in, dtype=bool)

    top_frac = float(np.clip(getattr(args, "glo_proto_init_top_frac", 0.3), 0.05, 1.0))
    sims0 = h @ p0.T
    y0 = np.argmax(sims0, axis=1)
    conf0 = np.max(sims0, axis=1)
    margin0 = _top2_margin(sims0)
    p_mix = np.zeros((c_in, m_proto, d), dtype=np.float32)
    class_multi_mask = np.ones(c_in, dtype=bool)
    use_trigger = bool(getattr(args, "glo_triggered_mixture", False))
    trig_delta = float(max(getattr(args, "glo_trigger_delta_thr", 0.12), 0.0))
    trig_unc = float(np.clip(getattr(args, "glo_trigger_margin_thr", 0.20), 0.0, 1.0))
    trig_dis = float(np.clip(getattr(args, "glo_trigger_disagree_thr", 0.20), 0.0, 1.0))
    trig_min = int(max(2, getattr(args, "glo_trigger_min_count", 12)))

    for c in range(c_in):
        idx_cls = np.where(y0 == c)[0]
        if idx_cls.size == 0:
            p_mix[c] = np.repeat(p0[c][None, :], m_proto, axis=0)
            class_multi_mask[c] = False
            continue

        use_multi_c = True
        if use_trigger:
            use_multi_c = False
            k_probe = max(2 * trig_min, int(round(top_frac * idx_cls.size)))
            k_probe = min(k_probe, idx_cls.size)
            idx_probe = idx_cls[np.argsort(-conf0[idx_cls])][:k_probe]
            if idx_probe.size >= 2 * trig_min:
                x_probe = h[idx_probe]
                mean_probe = np.mean(x_probe, axis=0, keepdims=True)
                sse1 = float(np.sum((x_probe - mean_probe) ** 2))
                lab2, ctr2 = _kmeans(
                    x_probe,
                    n_clusters=2,
                    seed=int(getattr(args, "seed", 0)) + 17 + c,
                    niter=int(getattr(args, "kmeans_niter", 100)),
                )
                ctr2 = np.asarray(ctr2, dtype=np.float32)
                x_rec = ctr2[np.asarray(lab2, dtype=np.int64)]
                sse2 = float(np.sum((x_probe - x_rec) ** 2))
                delta = (sse1 - sse2) / max(sse1, 1e-12)
                cnt = np.bincount(np.asarray(lab2, dtype=np.int64), minlength=2)
                cnt_ok = bool(np.all(cnt >= trig_min))
                unc = float(np.clip(1.0 - float(np.mean(margin0[idx_probe])), 0.0, 1.0))
                if y_cluster is not None:
                    disagree = float(np.mean(np.asarray(y_cluster[idx_probe], dtype=np.int64) != int(c)))
                else:
                    disagree = 0.0
                if delta >= trig_delta and cnt_ok and (unc >= trig_unc or disagree >= trig_dis):
                    use_multi_c = True
        class_multi_mask[c] = bool(use_multi_c)

        if not use_multi_c:
            p_mix[c] = np.repeat(p0[c][None, :], m_proto, axis=0)
            continue

        k_keep = max(m_proto, int(round(top_frac * idx_cls.size)))
        k_keep = min(k_keep, idx_cls.size)
        idx_sorted = idx_cls[np.argsort(-conf0[idx_cls])]
        idx_sel = idx_sorted[:k_keep]
        x = h[idx_sel]

        if x.shape[0] >= m_proto:
            _, ctr = _kmeans(
                x,
                n_clusters=m_proto,
                seed=int(getattr(args, "seed", 0)) + 101 + c,
                niter=int(getattr(args, "kmeans_niter", 100)),
            )
            ctr = _l2_normalize(ctr)
            order = np.argsort(-(ctr @ p0[c]))
            p_mix[c] = ctr[order]
        else:
            ctr = np.repeat(p0[c][None, :], m_proto, axis=0)
            ctr[:x.shape[0]] = _l2_normalize(x)
            p_mix[c] = ctr

    mixture_used = bool(np.any(class_multi_mask))
    return _l2_normalize(p_mix.reshape(-1, d)).reshape(c_in, m_proto, d), mixture_used, class_multi_mask


def _tta_consistency(h, p_star, args, component_to_class=None, num_classes=None):
    n = h.shape[0]
    c = int(num_classes) if num_classes is not None else p_star.shape[0]
    a_views = int(max(1, getattr(args, "tta_views", 1)))
    if a_views <= 1:
        sims_cls = _aggregate_component_sims(
            h @ p_star.T,
            component_to_class=component_to_class,
            num_classes=num_classes,
        )
        return {
            "views": 1,
            "agree": np.ones(n, dtype=np.float32),
            "stab": np.ones(n, dtype=np.float32),
            "conf_mean": np.max(sims_cls, axis=1).astype(np.float32),
        }

    noise_std = float(max(getattr(args, "tta_noise_std", 0.01), 0.0))
    drop_prob = float(np.clip(getattr(args, "tta_drop_prob", 0.0), 0.0, 0.95))
    preds = []
    margins = []
    confs = []

    for _ in range(a_views):
        h_aug = h.copy()
        if noise_std > 0.0:
            h_aug = h_aug + np.random.randn(*h_aug.shape).astype(np.float32) * noise_std
        if drop_prob > 0.0:
            keep = (np.random.rand(*h_aug.shape) >= drop_prob).astype(np.float32)
            h_aug = h_aug * keep
        h_aug = _l2_normalize(h_aug)

        sims_proto = h_aug @ p_star.T
        sims = _aggregate_component_sims(
            sims_proto,
            component_to_class=component_to_class,
            num_classes=num_classes,
        )
        preds.append(np.argmax(sims, axis=1))
        confs.append(np.max(sims, axis=1))

        if c <= 1:
            margins.append(np.clip(sims.reshape(-1), 0.0, 1.0))
        else:
            top2 = np.argsort(-sims, axis=1)[:, :2]
            top1 = sims[np.arange(n), top2[:, 0]]
            top2v = sims[np.arange(n), top2[:, 1]]
            margins.append(np.clip(top1 - top2v, 0.0, 1.0))

    pred_arr = np.stack(preds, axis=0)
    mode = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mode[i] = int(np.argmax(np.bincount(pred_arr[:, i], minlength=c)))
    agree = np.mean(pred_arr == mode[None, :], axis=0).astype(np.float32)

    margin_arr = np.stack(margins, axis=0).astype(np.float32)
    stab = np.exp(-np.var(margin_arr, axis=0)).astype(np.float32)
    conf_mean = np.mean(np.stack(confs, axis=0), axis=0).astype(np.float32)
    return {
        "views": a_views,
        "agree": np.clip(agree, 0.0, 1.0),
        "stab": np.clip(stab, 0.0, 1.0),
        "conf_mean": conf_mean,
    }


def _split_masks_scanobjectnn15(targets, split):
    targets = np.asarray(targets, dtype=np.int64)
    if split == "SR1":
        id_mask = targets <= 4
        id_labels = np.arange(0, 5, dtype=np.int64)
    elif split == "SR2":
        id_mask = (targets > 4) & (targets < 10)
        id_labels = np.arange(5, 10, dtype=np.int64)
    elif split == "SR3":
        id_mask = targets >= 10
        id_labels = np.arange(10, 15, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported ScanObjectNN15 split: {split}")
    return id_mask, id_labels


def _split_masks_shapenetcore54(targets, split):
    targets = np.asarray(targets, dtype=np.int64)
    if split == "SN1":
        id_labels = np.array([36, 29, 5, 53, 31, 49, 20, 13, 7, 27, 10, 8, 47, 6, 21, 19, 18, 37], dtype=np.int64)
    elif split == "SN2":
        id_labels = np.array([22, 28, 17, 38, 48, 30, 32, 3, 24, 12, 46, 41, 40, 33, 50, 4, 2, 1], dtype=np.int64)
    elif split == "SN3":
        id_labels = np.array([14, 34, 45, 23, 51, 25, 39, 26, 52, 0, 9, 15, 44, 43, 42, 16, 11, 35], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported ShapeNetCore54 split: {split}")
    id_mask = np.isin(targets, id_labels)
    return id_mask, id_labels


def _split_masks_modelnet40(targets, split):
    targets = np.asarray(targets, dtype=np.int64)
    if split == "MN1":
        id_mask = targets <= 12
        id_labels = np.arange(0, 13, dtype=np.int64)
    elif split == "MN2":
        id_mask = (targets > 12) & (targets < 26)
        id_labels = np.arange(13, 26, dtype=np.int64)
    elif split == "MN3":
        id_mask = targets >= 26
        id_labels = np.arange(26, 40, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported ModelNet40 split: {split}")
    return id_mask, id_labels


def _collapse_proto_mix(p_mix):
    """
    Collapse multi-prototype bank [C, M, D] to class-level anchors [C, D].
    """
    p_mix = np.asarray(p_mix, dtype=np.float32)
    if p_mix.ndim == 2:
        return _l2_normalize(p_mix)
    if p_mix.ndim != 3:
        raise ValueError(f"Unexpected prototype shape for collapse: {p_mix.shape}")
    return _l2_normalize(np.mean(p_mix, axis=1))


def _compute_local_scores(h, proto_for_local, W_tilde, graph_stats, args):
    """
    Compute local ID score and local reliability for the final LP local branch.
    """
    n = int(h.shape[0])
    local_method = "lp_softmax"

    local_stats = {
        "top1": np.zeros(n, dtype=np.float32),
        "margin": np.zeros(n, dtype=np.float32),
        "entropy": np.ones(n, dtype=np.float32),
        "conc": np.ones(n, dtype=np.float32),
    }

    mu = _l2_normalize(np.asarray(proto_for_local, dtype=np.float32))
    if mu.ndim != 2:
        raise ValueError(f"LP local expects [C,D] seeds, got shape={mu.shape}")
    seed_logits = h @ mu.T / max(args.seed_temp, 1e-8)
    seed_logits = seed_logits - seed_logits.max(axis=1, keepdims=True)
    seed_exp = np.exp(seed_logits)
    S0 = seed_exp / np.clip(seed_exp.sum(axis=1, keepdims=True), 1e-12, None)

    S = S0.copy()
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    for _ in range(args.T_lp):
        # Convex propagation to avoid score blow-up in zero-shot mode.
        S = (1.0 - alpha) * W_tilde.dot(S) + alpha * S0
        S = S / np.clip(S.sum(axis=1, keepdims=True), 1e-12, None)

    s_loc = np.clip(np.max(S, axis=1), 0.0, 1.0)
    m_loc = _top2_margin(S)
    if S.shape[1] > 1:
        eps = 1e-12
        entropy = -np.sum(S * np.log(np.clip(S, eps, None)), axis=1) / np.log(float(S.shape[1]))
        local_stats["entropy"] = np.clip(entropy.astype(np.float32), 0.0, 1.0)
    else:
        local_stats["entropy"] = np.zeros(n, dtype=np.float32)
    local_stats["conc"] = np.clip(np.sum(S * S, axis=1).astype(np.float32), 0.0, 1.0)
    local_stats["top1"] = np.clip(s_loc, 0.0, 1.0).astype(np.float32)
    local_stats["margin"] = np.clip(m_loc, 0.0, 1.0).astype(np.float32)

    edge_stability = np.asarray(
        graph_stats.get("edge_stability", np.ones(n, dtype=np.float32)),
        dtype=np.float32,
    )
    return s_loc, m_loc, edge_stability, local_method, local_stats


def compute_fpr95(y_true_ood, y_score_ood):
    fpr, tpr, _ = roc_curve(y_true_ood, y_score_ood)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def _solve_fusion_w_mse(s_loc, s_glo, pos_idx, neg_idx):
    if pos_idx.size == 0 or neg_idx.size == 0:
        return 0.5
    y_cal = np.concatenate([
        np.ones(pos_idx.size, dtype=np.float32),
        np.zeros(neg_idx.size, dtype=np.float32),
    ])
    cal_idx = np.concatenate([pos_idx, neg_idx], axis=0)
    s_loc_cal = s_loc[cal_idx]
    s_glo_cal = s_glo[cal_idx]
    z = s_loc_cal - s_glo_cal
    y_prime = y_cal - s_glo_cal
    denom = float(np.sum(z * z))
    if denom < 1e-12:
        return 0.5
    w = float(np.sum(z * y_prime) / denom)
    return float(np.clip(w, 0.0, 1.0))


def _solve_fusion_w_fpr95_grid(s_loc, s_glo, pos_idx, neg_idx, args, w_prior=0.5):
    if pos_idx.size == 0 or neg_idx.size == 0:
        return float(np.clip(w_prior, 0.0, 1.0))

    w_min = float(np.clip(getattr(args, "fusion_w_min", 0.0), 0.0, 1.0))
    w_max = float(np.clip(getattr(args, "fusion_w_max", 1.0), 0.0, 1.0))
    if w_max < w_min:
        w_min, w_max = w_max, w_min
    steps = int(max(11, getattr(args, "fusion_w_grid_steps", 101)))
    w_grid = np.linspace(w_min, w_max, steps, dtype=np.float32)
    prior_lambda = float(max(getattr(args, "fusion_fpr_prior_lambda", 0.0), 0.0))

    best_w = float(np.clip(w_prior, w_min, w_max))
    best_obj = float("inf")
    q = 0.05  # TPR ~= 95% on positive (ID) calibration set.
    for w in w_grid:
        id_score = w * s_loc + (1.0 - w) * s_glo
        tau = float(np.quantile(id_score[pos_idx], q))
        fpr = float(np.mean(id_score[neg_idx] >= tau))
        obj = fpr + prior_lambda * (float(w) - float(w_prior)) ** 2
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_w = float(w)
    return float(np.clip(best_w, w_min, w_max))


def _build_pseudo_calibration_sets(s_loc, s_glo, gi_ratio, tta_agree, tta_stab, args):
    n = int(s_loc.shape[0])
    k_pos = max(1, int(float(getattr(args, "calib_pos_frac", 0.1)) * n))
    k_neg = max(1, int(float(getattr(args, "calib_neg_frac", 0.1)) * n))

    pos_idx = np.argsort(-s_glo)[:k_pos]
    use_clean = bool(getattr(args, "fusion_use_clean_neg_intersection", False))
    if not use_clean:
        min_joint = np.minimum(s_loc, s_glo)
        neg_idx = np.argsort(min_joint)[:k_neg]
        return pos_idx.astype(np.int64), neg_idx.astype(np.int64), "baseline_min_joint"

    low_sloc = np.argsort(s_loc)[:k_neg]
    low_sglo = np.argsort(s_glo)[:k_neg]
    low_gi = np.argsort(gi_ratio)[:k_neg]
    tta_cons = 0.5 * (np.clip(tta_agree, 0.0, 1.0) + np.clip(tta_stab, 0.0, 1.0))
    low_tta = np.argsort(tta_cons)[:k_neg]
    min_keep = int(max(1, getattr(args, "fusion_clean_neg_min", 8)))

    neg_idx = np.intersect1d(low_sloc, low_sglo, assume_unique=False)
    neg_idx = np.intersect1d(neg_idx, low_gi, assume_unique=False)
    neg_idx = np.intersect1d(neg_idx, low_tta, assume_unique=False)
    neg_mode = "clean4"

    if neg_idx.size < min_keep:
        neg_idx = np.intersect1d(low_sloc, low_sglo, assume_unique=False)
        neg_idx = np.intersect1d(neg_idx, low_gi, assume_unique=False)
        neg_mode = "clean3_no_tta"
    if neg_idx.size < min_keep:
        neg_idx = np.intersect1d(low_sloc, low_sglo, assume_unique=False)
        neg_mode = "clean2_sloc_sglo"
    if neg_idx.size < min_keep:
        neg_idx = low_sglo
        neg_mode = "fallback_low_sglo"

    return pos_idx.astype(np.int64), np.asarray(neg_idx, dtype=np.int64), neg_mode


def _run_global_update_pass(
    h,
    p_start_mix,
    p_text,
    y_cluster,
    class_multi_mask,
    args,
    component_to_class,
    num_classes,
    text_anchor_lambda,
    eta_override=None,
):
    """
    Run one global update pass (the original T_p loop) from a given prototype state.
    Returns updated prototypes and pass statistics for optional outer revisit rounds.
    """
    n, d = h.shape
    c_in = int(num_classes)
    m_proto = int(p_start_mix.shape[1])
    p_prev_mix = np.asarray(p_start_mix, dtype=np.float32).copy()
    y_cluster = np.asarray(y_cluster, dtype=np.int64).reshape(-1)

    traj = [p_prev_mix.copy()]
    cons_history = []
    conf_history = []
    keep_last = np.zeros(n, dtype=bool)
    y_cls_last = np.zeros(n, dtype=np.int64)
    keep_freq_last = np.zeros(n, dtype=np.float32)
    dpam_rate_last = 0.0
    nr_rate_last = 0.0
    dgis_rate_last = 0.0
    class_skip_total = 0
    class_ess_values = []

    eta_val = float(np.clip(args.eta if eta_override is None else eta_override, 0.0, 1.0))
    topk_per_proto = int(max(0, getattr(args, "glo_update_topk_per_proto", 0)))
    use_stability = bool(getattr(args, "glo_revisit_use_stability", False))
    stab_views = int(max(1, getattr(args, "glo_revisit_stab_views", 4)))
    stab_noise_std = float(max(getattr(args, "glo_revisit_stab_noise_std", 0.01), 0.0))
    stab_min_freq = float(np.clip(getattr(args, "glo_revisit_stab_min_freq", 0.6), 0.0, 1.0))
    use_stab_weight = bool(getattr(args, "glo_revisit_use_stab_weight", True))
    ess_min = float(max(getattr(args, "glo_revisit_ess_min", 0.0), 0.0)) if use_stability else 0.0

    def _compute_keep_mask(h_view, sims_view, sims_proto_view, y_cls_view, p_mix_view):
        dpam_mask = y_cls_view == y_cluster
        conf_view = np.max(sims_view, axis=1)

        sims_pred_cls = sims_proto_view.reshape(n, c_in, m_proto)[np.arange(n), y_cls_view, :]
        best_comp = np.argmax(sims_pred_cls, axis=1)
        assigned_centers = p_mix_view[y_cls_view, best_comp]
        dist_assigned = np.linalg.norm(h_view - assigned_centers, axis=1)

        rank_conf = np.argsort(-conf_view)
        rank_geo = np.argsort(dist_assigned)
        rbo_scores = rbo_item_scores(rank_conf, rank_geo, p=args.rbo_p)
        k_nr = max(1, int(args.q_frac * n))
        nr_idx = np.argsort(-rbo_scores)[:k_nr]
        nr_mask = np.zeros(n, dtype=bool)
        nr_mask[nr_idx] = True

        top2 = np.argsort(-sims_view, axis=1)[:, :2]
        sim_nearest = sims_view[np.arange(n), top2[:, 0]]
        sim_second = sims_view[np.arange(n), top2[:, 1]] if c_in > 1 else np.full(n, -1.0, dtype=np.float32)
        delta = np.sqrt(np.clip(2.0 - 2.0 * sim_nearest, 0.0, None))
        beta = np.sqrt(np.clip(2.0 - 2.0 * sim_second, 0.0, None))
        dgis_mask = ((beta - delta) / (delta + 1e-12)) >= args.tau_GI
        if not bool(getattr(args, "glo_keep_use_dpam", True)):
            dpam_mask = np.ones(n, dtype=bool)
        if not bool(getattr(args, "glo_keep_use_nr", True)):
            nr_mask = np.ones(n, dtype=bool)
        if not bool(getattr(args, "glo_keep_use_dgis", True)):
            dgis_mask = np.ones(n, dtype=bool)
        keep_mask = dpam_mask & nr_mask & dgis_mask
        stats = {
            "dpam_rate": float(np.mean(dpam_mask.astype(np.float32))),
            "nr_rate": float(np.mean(nr_mask.astype(np.float32))),
            "dgis_rate": float(np.mean(dgis_mask.astype(np.float32))),
            "keep_rate": float(np.mean(keep_mask.astype(np.float32))),
        }
        return keep_mask, stats

    steps = int(max(getattr(args, "T_p", 0), 0))
    for _ in range(steps):
        p_prev_flat = p_prev_mix.reshape(c_in * m_proto, d)
        sims_proto = h @ p_prev_flat.T
        sims = _aggregate_component_sims(
            sims_proto,
            component_to_class=component_to_class,
            num_classes=c_in,
        )
        y_cls = np.argmax(sims, axis=1)
        conf = np.max(sims, axis=1)
        conf_history.append(conf.astype(np.float32))

        keep_base, keep_stats = _compute_keep_mask(h, sims, sims_proto, y_cls, p_prev_mix)
        keep = keep_base.copy()
        keep_freq = keep_base.astype(np.float32)
        sample_weight = np.ones(n, dtype=np.float32)
        dpam_rate_last = float(keep_stats["dpam_rate"])
        nr_rate_last = float(keep_stats["nr_rate"])
        dgis_rate_last = float(keep_stats["dgis_rate"])

        if use_stability and stab_views > 1:
            votes = keep_base.astype(np.float32).copy()
            for _v in range(1, stab_views):
                if stab_noise_std > 0.0:
                    h_aug = _l2_normalize(h + np.random.normal(0.0, stab_noise_std, size=h.shape).astype(np.float32))
                else:
                    h_aug = h
                sims_proto_aug = h_aug @ p_prev_flat.T
                sims_aug = _aggregate_component_sims(
                    sims_proto_aug,
                    component_to_class=component_to_class,
                    num_classes=c_in,
                )
                y_cls_aug = np.argmax(sims_aug, axis=1)
                keep_aug, _ = _compute_keep_mask(h_aug, sims_aug, sims_proto_aug, y_cls_aug, p_prev_mix)
                votes += keep_aug.astype(np.float32)
            keep_freq = votes / float(stab_views)
            stable_mask = keep_freq >= stab_min_freq
            keep = keep_base & stable_mask
            if use_stab_weight:
                sample_weight = np.clip(keep_freq, 1e-6, None).astype(np.float32)

        keep_last = keep
        keep_freq_last = keep_freq.astype(np.float32)
        y_cls_last = y_cls

        p_new_mix = p_prev_mix.copy()
        lam_txt = float(np.clip(text_anchor_lambda, 0.0, 1.0))
        class_skip_step = 0
        for c in range(c_in):
            cls_rows = np.where(keep & (y_cls == c))[0]
            cls_prev = p_prev_mix[c]

            if cls_rows.size > 0:
                ess_weights = np.clip(sample_weight[cls_rows], 1e-12, None).astype(np.float32)
                ess_cls = float(_ess_from_weights(ess_weights))
                class_ess_values.append(ess_cls)
                if ess_min > 0.0 and ess_cls < ess_min:
                    cls_rows = np.zeros(0, dtype=np.int64)
                    class_skip_step += 1

            if cls_rows.size > 0:
                if bool(class_multi_mask[c]):
                    sims_cls_comp = h[cls_rows] @ cls_prev.T
                    assign = np.argmax(sims_cls_comp, axis=1)
                else:
                    assign = np.zeros(cls_rows.size, dtype=np.int64)
            else:
                assign = np.zeros(0, dtype=np.int64)

            for m in range(m_proto):
                if (not bool(class_multi_mask[c])) and m > 0:
                    p_new_mix[c, m] = p_new_mix[c, 0]
                    continue
                if cls_rows.size > 0:
                    rows_m = cls_rows[assign == m]
                else:
                    rows_m = np.zeros(0, dtype=np.int64)

                if rows_m.size > 0 and topk_per_proto > 0 and rows_m.size > topk_per_proto:
                    # Rank seed points by current prototype similarity and keep only top-k.
                    rank_score = h[rows_m] @ cls_prev[m]
                    top_idx = np.argsort(-rank_score)[:topk_per_proto]
                    rows_m = rows_m[top_idx]

                if rows_m.size > 0:
                    if use_stability and use_stab_weight:
                        w_m = np.clip(sample_weight[rows_m], 1e-6, None).astype(np.float32)
                    else:
                        w_m = np.ones(rows_m.size, dtype=np.float32)
                    mean_h = _l2_normalize(
                        np.sum(h[rows_m] * w_m[:, None], axis=0, keepdims=True)
                        / np.clip(np.sum(w_m), 1e-6, None)
                    )[0]
                    proposal = (1.0 - eta_val) * cls_prev[m] + eta_val * mean_h
                else:
                    proposal = cls_prev[m]

                if lam_txt > 0.0:
                    proposal = (1.0 - lam_txt) * proposal + lam_txt * p_text[c]
                p_new_mix[c, m] = _l2_normalize(proposal.reshape(1, -1))[0]

        class_skip_total += int(class_skip_step)
        cons = np.sum(p_new_mix * p_prev_mix, axis=2)
        cons_history.append(float(np.mean(cons)))
        traj.append(p_new_mix.copy())
        p_prev_mix = p_new_mix

    if len(cons_history) == 0:
        p_star_mix = p_prev_mix
    else:
        omega_logits = args.lambda_cons * np.asarray(cons_history, dtype=np.float32)
        omega_logits = omega_logits - omega_logits.max()
        omega = np.exp(omega_logits)
        omega = omega / np.clip(np.sum(omega), 1e-12, None)

        p_star_mix = np.zeros_like(p_prev_mix)
        for t in range(len(omega)):
            p_star_mix += omega[t] * traj[t + 1]
        p_star_mix = _l2_normalize(p_star_mix.reshape(c_in * m_proto, d)).reshape(c_in, m_proto, d)

    return p_star_mix, {
        "conf_history": conf_history,
        "keep_last": keep_last,
        "y_cls_last": y_cls_last,
        "keep_freq_last": keep_freq_last,
        "glo_gate_dpam_rate": float(dpam_rate_last),
        "glo_gate_nr_rate": float(nr_rate_last),
        "glo_gate_dgis_rate": float(dgis_rate_last),
        "glo_gate_keep_rate": float(np.mean(keep_last.astype(np.float32))),
        "class_skip_total": int(class_skip_total),
        "class_ess_mean": float(np.mean(class_ess_values)) if len(class_ess_values) > 0 else 0.0,
        "class_ess_min": float(np.min(class_ess_values)) if len(class_ess_values) > 0 else 0.0,
    }


def run_logofuse(
    features,
    targets,
    text_prototypes,
    args,
    fewshot_prototypes=None,
    fusion_weight_override=None,
    geo_score=None,
    train_anchor_features=None,
):
    """
    Run improved LoGo-Fuse in strict zero-shot mode (L = empty).

    Args:
        features: [N, D] normalized visual features from ULIP2.
        targets: [N] ground-truth class ids (for metrics only).
        text_prototypes: [C_in, D] normalized text prototypes p_c^(0).
        args: argparse namespace with LoGo-Fuse hyper-params.
    """
    h = _l2_normalize(np.asarray(features, dtype=np.float32))
    y_gt = np.asarray(targets, dtype=np.int64)
    p_text = _l2_normalize(np.asarray(text_prototypes, dtype=np.float32))
    if fewshot_prototypes is not None:
        p_fs = _l2_normalize(np.asarray(fewshot_prototypes, dtype=np.float32))
        shot = max(int(getattr(args, "shot", 1)), 1)
        beta_base = float(np.clip(getattr(args, "fewshot_blend_beta", 0.35), 0.0, 1.0))
        ref_shot = float(max(getattr(args, "fewshot_blend_ref_shot", 5.0), 1e-6))
        beta_eff = float(np.clip(beta_base * (shot / (shot + ref_shot)), 0.0, 0.95))
        # Keep text prototype as anchor, add few-shot prototype as residual evidence.
        p0 = _l2_normalize((1.0 - beta_eff) * p_text + beta_eff * p_fs)
        text_anchor_lambda = float(np.clip(getattr(args, "text_anchor_lambda", 0.15), 0.0, 1.0))
    else:
        p0 = p_text
        text_anchor_lambda = 0.0

    n, d = h.shape
    c_in = p0.shape[0]
    global_score_mode = "maxcos"

    if args.dataset_name == "ScanObjectNN15":
        id_mask, id_labels = _split_masks_scanobjectnn15(y_gt, args.dataset_split)
    elif args.dataset_name == "ShapeNetCore54":
        id_mask, id_labels = _split_masks_shapenetcore54(y_gt, args.dataset_split)
    elif args.dataset_name == "ModelNet40":
        id_mask, id_labels = _split_masks_modelnet40(y_gt, args.dataset_split)
    else:
        id_mask = np.ones_like(y_gt, dtype=bool)
        id_labels = np.arange(c_in, dtype=np.int64)

    # Diagnostic only: use true ID labels from eval set to construct class prototypes.
    oracle_true_id_proto = bool(getattr(args, "oracle_true_id_proto", False))
    if oracle_true_id_proto:
        p_oracle = p0.copy()
        id_labels_arr = np.asarray(id_labels, dtype=np.int64).reshape(-1)
        for class_idx in range(min(c_in, id_labels_arr.shape[0])):
            raw_lab = int(id_labels_arr[class_idx])
            idx = np.where(y_gt == raw_lab)[0]
            if idx.size > 0:
                p_oracle[class_idx] = _l2_normalize(
                    np.mean(h[idx], axis=0, keepdims=True).astype(np.float32)
                )[0]
        p0 = _l2_normalize(p_oracle)

    # (1) Over-clustering + cluster-induced class assignment.
    k_over = args.K_over if args.K_over > 0 else max(c_in * 8, c_in + 1)
    k_over = min(k_over, n)
    cluster_ids, cluster_centers = _kmeans(h, n_clusters=k_over, seed=args.seed, niter=args.kmeans_niter)
    cluster_centers = _l2_normalize(cluster_centers)

    # mu_c from current prototype initialization:
    # - zero-shot: text prototypes
    # - few-shot: few-shot visual prototypes
    mu = p0.copy()

    cluster_to_class = np.argmax(cluster_centers @ mu.T, axis=1)
    y_cluster = cluster_to_class[cluster_ids]

    # (2) LOCAL MODULE
    W_graph = stable_knn_adjacency(
        h,
        k=args.k,
        B=args.B,
        noise_std=args.graph_noise_std,
        stable_pi=args.stable_pi,
        temp=args.temp,
        mutual_only=bool(getattr(args, "graph_mutual_knn", False)),
        local_scaling=bool(getattr(args, "graph_local_scaling", False)),
        local_scaling_k=int(getattr(args, "graph_local_scaling_k", 0)),
        return_stats=True,
    )
    if isinstance(W_graph, tuple):
        W_tilde, graph_stats = W_graph
    else:
        W_tilde = W_graph
        graph_stats = {}
    graph_edge_count = int(graph_stats.get("graph_edge_count", 0))
    s_loc, m_loc, edge_stability, local_method, local_stats = _compute_local_scores(
        h=h,
        proto_for_local=mu,
        W_tilde=W_tilde,
        graph_stats=graph_stats,
        args=args,
    )
    # Apply local score variant (A0/A1/A2/A3) before global pseudo-OOD ranking uses s_loc.
    local_stats["top1"] = np.clip(s_loc, 0.0, 1.0).astype(np.float32)
    local_stats["margin"] = np.clip(m_loc, 0.0, 1.0).astype(np.float32)
    s_loc, local_score_variant = _apply_local_score_variant(
        s_loc=s_loc,
        m_loc=m_loc,
        local_stats=local_stats,
        args=args,
    )
    local_score_top1_mean = float(np.mean(np.asarray(local_stats.get("top1", s_loc), dtype=np.float32)))
    local_score_margin_mean = float(np.mean(np.asarray(local_stats.get("margin", m_loc), dtype=np.float32)))
    local_score_conc_mean = float(np.mean(np.asarray(local_stats.get("conc", np.ones(n, dtype=np.float32)), dtype=np.float32)))
    local_score_entropy_mean = float(np.mean(np.asarray(local_stats.get("entropy", np.ones(n, dtype=np.float32)), dtype=np.float32)))

    # (3) GLOBAL MODULE (DPAM + NR + DGIS).
    p_prev_mix, mixture_used, class_multi_mask = _init_multi_prototypes(h, p0, args, y_cluster=y_cluster)
    m_proto = int(p_prev_mix.shape[1])
    comp_to_class = np.repeat(np.arange(c_in, dtype=np.int64), m_proto)
    conf_history = []
    use_revisit = bool(getattr(args, "glo_use_iterative_revisit", False))
    revisit_rounds = int(max(1, getattr(args, "glo_revisit_rounds", 2))) if use_revisit else 1
    revisit_min_rounds = int(max(1, getattr(args, "glo_revisit_min_rounds", 1)))
    revisit_proto_shift_tol = float(max(getattr(args, "glo_revisit_proto_shift_tol", 5e-4), 0.0))
    revisit_keep_change_tol = float(np.clip(getattr(args, "glo_revisit_keep_change_tol", 0.01), 0.0, 1.0))
    revisit_eta_scale = float(np.clip(getattr(args, "glo_revisit_eta_scale", 0.5), 0.0, 1.0))
    revisit_gap_q = float(np.clip(getattr(args, "glo_revisit_gap_q", 0.10), 1e-4, 0.5))
    revisit_gap_tol = float(getattr(args, "glo_revisit_gap_tol", 0.0))
    revisit_reassign_clusters = bool(getattr(args, "glo_revisit_reassign_clusters", True))
    y_cluster_round = y_cluster.copy()
    keep_prev = None
    revisit_round_used = 0
    revisit_proto_shift = 0.0
    revisit_keep_change = 0.0
    revisit_recovered = 0
    revisit_excluded = 0
    revisit_gap_score = 0.0
    revisit_class_skip_total = 0
    revisit_class_ess_mean = 0.0
    revisit_class_ess_min = 0.0
    glo_gate_dpam_rate = 0.0
    glo_gate_nr_rate = 0.0
    glo_gate_dgis_rate = 0.0
    glo_gate_keep_rate = 0.0
    prev_gap = None

    for ridx in range(revisit_rounds):
        eta_round = float(args.eta if ridx == 0 else args.eta * revisit_eta_scale)
        p_new_mix, pass_info = _run_global_update_pass(
            h=h,
            p_start_mix=p_prev_mix,
            p_text=p_text,
            y_cluster=y_cluster_round,
            class_multi_mask=class_multi_mask,
            args=args,
            component_to_class=comp_to_class,
            num_classes=c_in,
            text_anchor_lambda=text_anchor_lambda,
            eta_override=eta_round,
        )

        conf_history.extend(pass_info.get("conf_history", []))
        revisit_class_skip_total += int(pass_info.get("class_skip_total", 0))
        revisit_class_ess_mean = float(pass_info.get("class_ess_mean", 0.0))
        revisit_class_ess_min = float(pass_info.get("class_ess_min", 0.0))
        glo_gate_dpam_rate = float(pass_info.get("glo_gate_dpam_rate", 0.0))
        glo_gate_nr_rate = float(pass_info.get("glo_gate_nr_rate", 0.0))
        glo_gate_dgis_rate = float(pass_info.get("glo_gate_dgis_rate", 0.0))
        glo_gate_keep_rate = float(pass_info.get("glo_gate_keep_rate", 0.0))
        keep_now = np.asarray(pass_info.get("keep_last", np.zeros(n, dtype=bool)), dtype=bool)
        if keep_prev is not None and keep_now.shape[0] == keep_prev.shape[0]:
            revisit_keep_change = float(np.mean(keep_now != keep_prev))
            revisit_recovered = int(np.sum((~keep_prev) & keep_now))
        else:
            revisit_keep_change = 1.0
            revisit_recovered = 0
        revisit_excluded = int(np.sum(~keep_now)) if keep_now.shape[0] == n else 0

        revisit_proto_shift = float(np.mean(1.0 - np.sum(p_prev_mix * p_new_mix, axis=2)))
        p_prev_mix = p_new_mix
        revisit_round_used = int(ridx + 1)

        mu_round_eval = _collapse_proto_mix(p_prev_mix)
        sims_round_eval = h @ mu_round_eval.T
        conf_round_eval = np.max(sims_round_eval, axis=1)
        qn = max(1, int(revisit_gap_q * n))
        top_gap_idx = np.argsort(-conf_round_eval)[:qn]
        low_gap_idx = np.argsort(conf_round_eval)[:qn]
        gap_now = float(np.median(conf_round_eval[top_gap_idx]) - np.median(conf_round_eval[low_gap_idx]))
        revisit_gap_score = gap_now

        if revisit_reassign_clusters and (ridx + 1) < revisit_rounds:
            mu_round = _collapse_proto_mix(p_prev_mix)
            cluster_to_class_round = np.argmax(cluster_centers @ mu_round.T, axis=1)
            y_cluster_round = cluster_to_class_round[cluster_ids]

        keep_prev = keep_now
        if (ridx + 1) < revisit_rounds and (ridx + 1) >= revisit_min_rounds:
            gap_improve = float("inf") if prev_gap is None else (gap_now - prev_gap)
            if (
                revisit_proto_shift <= revisit_proto_shift_tol
                and revisit_keep_change <= revisit_keep_change_tol
                and gap_improve <= revisit_gap_tol
            ):
                break
        prev_gap = gap_now

    p_star_mix = p_prev_mix
    y_cluster = y_cluster_round

    p_star_flat = p_star_mix.reshape(c_in * m_proto, d)
    sims_star_proto = h @ p_star_flat.T
    sims_star = _aggregate_component_sims(
        sims_star_proto,
        component_to_class=comp_to_class,
        num_classes=c_in,
    )
    s_glo_pos_cos = np.max(sims_star, axis=1).astype(np.float32)
    s_glo_pos = s_glo_pos_cos.copy()
    s_glo_raw = s_glo_pos.copy()
    m_glo_raw = _top2_margin(sims_star)
    s_glo_raw = s_glo_pos.copy()
    tta_info = _tta_consistency(
        h,
        p_star_flat,
        args,
        component_to_class=comp_to_class,
        num_classes=c_in,
    )
    tta_agree = tta_info["agree"]
    tta_stab = tta_info["stab"]
    tta_conf_mean = tta_info["conf_mean"]
    tta_views = int(tta_info["views"])
    neg_bank_used = False
    neg_pool_size = 0
    neg_k_used = 0
    tta_filter_used = False
    tta_pool_size = 0
    neg_ess = 0.0
    neg_weight_mean = 1.0
    neg_margin_effective = float(getattr(args, "neg_margin_scale", 1.0))
    neg_rank_consistency = 1.0
    neg_beta_effective = float(max(getattr(args, "neg_rank_beta", 0.3), 0.0))
    neg_ess_temp_used = 1.0
    neg_stab_used = False
    neg_stab_views = 1
    neg_stab_thr = 0.0
    neg_stab_pool_mean = 1.0
    neg_stab_pool_min = 1.0
    neg_source_mode = "pseudo"
    neg_centers = None
    neg_pool_ood_ratio = 0.0
    neg_fit_size = 0
    neg_fit_ood_ratio = 0.0
    neg_hard_used = False
    neg_hard_count = 0
    neg_hard_ood_ratio = 0.0
    pred_cls_star = np.argmax(sims_star, axis=1)
    dist_star = np.sqrt(np.clip(2.0 - 2.0 * s_glo_pos_cos, 0.0, None)).astype(np.float32)
    disagree_star = (pred_cls_star != y_cluster).astype(np.float32)
    single_neg_used = False
    single_neg_pool_size = 0
    single_neg_ess = 0.0
    single_neg_beta_eff = 0.0
    single_neg_lambda = 0.0
    single_neg_mu_id = 0.0
    single_neg_tau_id = 0.0
    single_neg_pool_ood_ratio = 0.0

    if bool(getattr(args, "glo_use_single_neg_proto", False)):
        pool_frac = float(np.clip(getattr(args, "neg_single_pool_frac", 0.15), 1e-4, 0.5))
        k_pool = max(1, int(pool_frac * n))
        low_glo_idx = np.argsort(s_glo_pos)[:k_pool]
        low_loc_idx = np.argsort(s_loc)[:k_pool]
        pool_idx = np.intersect1d(low_glo_idx, low_loc_idx, assume_unique=False)
        if pool_idx.size < max(1, k_pool // 4):
            pool_idx = low_glo_idx
        pool_idx = np.asarray(pool_idx, dtype=np.int64)
        single_neg_pool_size = int(pool_idx.size)
        if single_neg_pool_size > 0:
            single_neg_pool_ood_ratio = float(np.mean((~id_mask[pool_idx]).astype(np.float32)))
        min_pool = int(max(2, getattr(args, "neg_single_min_pool", 16)))
        if single_neg_pool_size >= min_pool:
            if bool(getattr(args, "neg_single_use_weighted_center", True)):
                susp_g = _minmax01(1.0 - s_glo_pos[pool_idx])
                susp_l = _minmax01(1.0 - s_loc[pool_idx])
                tta_cons = np.clip(0.5 * (tta_agree[pool_idx] + tta_stab[pool_idx]), 0.0, 1.0)
                weights = np.clip((0.5 * susp_g + 0.5 * susp_l) * (1.0 - tta_cons), 1e-6, None).astype(np.float32)
            else:
                weights = np.ones(single_neg_pool_size, dtype=np.float32)

            single_neg_ess = float(_ess_from_weights(weights))
            beta_base = float(max(getattr(args, "neg_single_beta", 1.0), 0.0))
            if bool(getattr(args, "neg_single_use_quality_gate", True)):
                e_min = float(max(getattr(args, "neg_single_ess_min", 16.0), 1.0))
                e_tar = float(max(getattr(args, "neg_single_ess_target", 64.0), e_min + 1e-6))
                lam_ess = float(np.clip((single_neg_ess - e_min) / max(e_tar - e_min, 1e-6), 0.0, 1.0))
                tau_q = float(np.clip(getattr(args, "neg_single_purity_tau_q", 0.35), 0.01, 0.99))
                single_neg_mu_id = float(np.mean(s_glo_pos[pool_idx]))
                single_neg_tau_id = float(np.quantile(s_glo_pos, tau_q))
                pur_delta = float(max(getattr(args, "neg_single_purity_delta", 0.05), 1e-6))
                lam_pur = float(np.clip((single_neg_tau_id - single_neg_mu_id) / pur_delta, 0.0, 1.0))
                lam_gamma = float(max(getattr(args, "neg_single_quality_gamma", 2.0), 1e-6))
                single_neg_lambda = float(np.clip(lam_ess * lam_pur, 0.0, 1.0) ** lam_gamma)
            else:
                single_neg_lambda = 1.0
            single_neg_beta_eff = float(beta_base * single_neg_lambda)
            if single_neg_beta_eff > 1e-8:
                if bool(getattr(args, "neg_single_use_weighted_center", True)):
                    center = np.sum(h[pool_idx] * weights[:, None], axis=0, keepdims=True)
                else:
                    center = np.mean(h[pool_idx], axis=0, keepdims=True)
                neg_center = _l2_normalize(center.astype(np.float32))[0]
                s_neg = h @ neg_center.reshape(-1, 1)
                s_neg = s_neg.reshape(-1).astype(np.float32)
                s_glo_raw = s_glo_raw - single_neg_beta_eff * s_neg
                single_neg_used = True
                neg_bank_used = True
                neg_pool_size = int(single_neg_pool_size)
                neg_k_used = 1
                neg_ess = float(single_neg_ess)
                neg_weight_mean = float(np.mean(weights))
                neg_margin_effective = float(single_neg_beta_eff)
    use_neg_bank = bool(getattr(args, "glo_use_neg_bank", True))
    if bool(getattr(args, "glo_use_single_neg_proto", False)) and bool(getattr(args, "neg_single_replace_bank", True)):
        use_neg_bank = False
    oracle_true_ood_neg_bank = bool(getattr(args, "oracle_true_ood_neg_bank", False))
    if use_neg_bank and oracle_true_ood_neg_bank:
        pool_idx = np.where(~id_mask)[0].astype(np.int64)
        neg_pool_size = int(pool_idx.size)
        if neg_pool_size > 0:
            neg_pool_ood_ratio = 1.0
        min_pool = int(max(2, getattr(args, "neg_min_pool", 32)))
        if neg_pool_size >= min_pool:
            neg_k_req = int(max(1, getattr(args, "neg_k", 10)))
            neg_k_used = int(min(neg_k_req, neg_pool_size))
            _, neg_centers = _kmeans(
                h[pool_idx],
                n_clusters=neg_k_used,
                seed=int(getattr(args, "seed", 0)) + 131,
                niter=int(getattr(args, "kmeans_niter", 100)),
            )
            neg_centers = _l2_normalize(neg_centers)
            s_glo_neg = np.max(h @ neg_centers.T, axis=1)
            neg_margin_effective = float(getattr(args, "neg_margin_scale", 1.0))
            s_glo_raw = s_glo_raw - neg_margin_effective * s_glo_neg
            neg_bank_used = bool(neg_margin_effective > 1e-8)
            neg_ess = float(neg_pool_size)
            neg_weight_mean = 1.0
            neg_fit_size = int(pool_idx.size)
            if neg_fit_size > 0:
                neg_fit_ood_ratio = 1.0
            neg_source_mode = "oracle_true_ood"
            # Skip pseudo pool/ranking path when oracle bank is enabled.
            use_neg_bank = False
        else:
            neg_source_mode = "oracle_true_ood_insufficient_fallback"
    if use_neg_bank:
        pool_frac = float(np.clip(getattr(args, "neg_pool_frac", 0.15), 1e-4, 0.5))
        k_pool = max(1, int(pool_frac * n))
        neg_stab_freq = np.ones(n, dtype=np.float32)
        base_susp = _minmax01(1.0 - s_glo_pos)

        use_joint_rank = bool(getattr(args, "neg_rank_use_joint", False))
        if use_joint_rank:
            gamma = float(np.clip(getattr(args, "neg_rank_gamma", 0.6), 0.0, 1.0))
            beta = float(max(getattr(args, "neg_rank_beta", 0.3), 0.0))
            rank_s = _rank01(s_glo_pos, descending=False)  # lower score => more OOD-like
            rank_d = _rank01(dist_star, descending=True)    # larger distance => more OOD-like
            score_susp = 1.0 - rank_s
            dist_susp = 1.0 - rank_d
            neg_rank_consistency = _rank_corr01(score_susp, dist_susp)
            beta_eff = beta
            if bool(getattr(args, "neg_rank_shot_aware", False)):
                shot_v = float(max(int(getattr(args, "shot", 0)), 0))
                shot_ref = float(max(getattr(args, "neg_rank_ref_shot", 5.0), 1e-6))
                shot_scale = float(np.clip(shot_v / shot_ref, 0.0, 1.0))
                r0 = float(np.clip(getattr(args, "neg_rank_r0", 0.2), 0.0, 0.99))
                rank_scale = float(np.clip((neg_rank_consistency - r0) / max(1.0 - r0, 1e-6), 0.0, 1.0))
                beta_eff = beta * shot_scale * rank_scale
            neg_beta_effective = float(beta_eff)
            neg_score = gamma * score_susp + (1.0 - gamma) * dist_susp + neg_beta_effective * disagree_star
            base_susp = _minmax01(neg_score)
            pool_idx = np.argsort(-neg_score)[:k_pool]
        else:
            # Baseline pool: low global confidence intersect low local confidence.
            low_glo_idx = np.argsort(s_glo_pos)[:k_pool]
            low_loc_idx = np.argsort(s_loc)[:k_pool]
            pool_idx = np.intersect1d(low_glo_idx, low_loc_idx, assume_unique=False)
            if len(pool_idx) < max(1, k_pool // 4):
                # fallback to low global confidence pool if intersection is too small.
                pool_idx = low_glo_idx

        # Hard-negative branch: capture OOD points that still get high ID similarity.
        if bool(getattr(args, "neg_hard_enable", False)):
            hard_frac = float(np.clip(getattr(args, "neg_hard_frac", 0.08), 1e-4, 0.5))
            k_hard = max(1, int(hard_frac * n))
            hard_score = _minmax01(s_glo_pos) * _minmax01(1.0 - m_glo_raw)
            if bool(getattr(args, "neg_hard_use_disagree", True)):
                hard_score = hard_score * (0.5 + 0.5 * disagree_star)
            hard_idx = np.argsort(-hard_score)[:k_hard]
            merged = np.unique(np.concatenate([np.asarray(pool_idx, dtype=np.int64), hard_idx.astype(np.int64)]))

            cap_mult = float(max(getattr(args, "neg_pool_cap_mult", 2.0), 1.0))
            k_cap = int(min(n, max(k_pool, int(round(cap_mult * k_pool)))))
            if merged.size > k_cap:
                alpha = float(np.clip(getattr(args, "neg_hard_alpha", 0.5), 0.0, 1.0))
                merge_score = (1.0 - alpha) * base_susp + alpha * _minmax01(hard_score)
                keep = np.argsort(-merge_score[merged])[:k_cap]
                merged = merged[keep]
            pool_idx = np.asarray(merged, dtype=np.int64)
            neg_hard_used = True
            neg_hard_count = int(hard_idx.size)
            if hard_idx.size > 0:
                neg_hard_ood_ratio = float(np.mean((~id_mask[hard_idx]).astype(np.float32)))

        use_hard_tta_filter = bool(getattr(args, "tta_filter_neg_pool", False))
        if bool(getattr(args, "neg_tta_soft_only", False)):
            use_hard_tta_filter = False
        if use_hard_tta_filter and tta_views > 1:
            low_tta_idx = np.argsort(tta_conf_mean)[:k_pool]
            agree_thr = float(np.clip(getattr(args, "tta_agree_thr", 0.75), 0.0, 1.0))
            stab_thr = float(np.clip(getattr(args, "tta_stab_thr", 0.85), 0.0, 1.0))
            stable_idx = np.where((tta_agree >= agree_thr) & (tta_stab >= stab_thr))[0]
            tta_candidates = np.intersect1d(low_tta_idx, stable_idx, assume_unique=False)
            tta_pool_size = int(len(tta_candidates))

            min_keep = max(1, k_pool // 5)
            refined = np.intersect1d(pool_idx, tta_candidates, assume_unique=False)
            if len(refined) >= min_keep:
                pool_idx = refined
                tta_filter_used = True
            else:
                # Soft fallback: keep only TTA low-confidence constraint.
                refined_low = np.intersect1d(pool_idx, low_tta_idx, assume_unique=False)
                if len(refined_low) >= min_keep:
                    pool_idx = refined_low
                    tta_filter_used = True

        use_neg_stab = bool(getattr(args, "neg_stab_enable", False))
        if use_neg_stab and len(pool_idx) > 0:
            neg_stab_freq, neg_stab_views = _neg_stability_frequency(
                h=h,
                p_star_flat=p_star_flat,
                s_loc=s_loc,
                y_cluster=y_cluster,
                k_pool=k_pool,
                use_joint_rank=use_joint_rank,
                neg_beta_effective=neg_beta_effective,
                args=args,
                component_to_class=comp_to_class,
                num_classes=c_in,
            )
            neg_stab_used = True
            neg_stab_thr = float(np.clip(getattr(args, "neg_stab_thr", 0.8), 0.0, 1.0))
            if bool(getattr(args, "neg_stab_hard", True)):
                min_keep_stab = int(max(1, getattr(args, "neg_stab_min_keep", max(4, k_pool // 5))))
                stab_cur = neg_stab_freq[np.asarray(pool_idx, dtype=np.int64)]
                keep_mask = stab_cur >= neg_stab_thr
                refined = np.asarray(pool_idx, dtype=np.int64)[keep_mask]
                if refined.size >= min_keep_stab:
                    pool_idx = refined
                else:
                    order = np.argsort(-stab_cur)
                    take = int(min(max(min_keep_stab, 1), len(pool_idx)))
                    pool_idx = np.asarray(pool_idx, dtype=np.int64)[order[:take]]

        pool_idx = np.asarray(pool_idx, dtype=np.int64)
        neg_pool_size = int(len(pool_idx))
        if neg_pool_size > 0:
            neg_pool_ood_ratio = float(np.mean((~id_mask[pool_idx]).astype(np.float32)))
        min_pool = int(max(2, getattr(args, "neg_min_pool", 32)))
        use_soft_ess = bool(getattr(args, "neg_use_soft_weight_ess", False))
        use_ess_target = bool(getattr(args, "neg_use_ess_target", False))
        if neg_pool_size > 0:
            if use_soft_ess and tta_views > 1:
                tau_a = float(np.clip(getattr(args, "neg_weight_tau_a", 0.75), 0.0, 1.0))
                tau_s = float(np.clip(getattr(args, "neg_weight_tau_s", 0.85), 0.0, 1.0))
                delta_w = float(max(getattr(args, "neg_weight_delta", 0.08), 1e-6))
                wa = _sigmoid((tta_agree[pool_idx] - tau_a) / delta_w)
                ws = _sigmoid((tta_stab[pool_idx] - tau_s) / delta_w)
                if bool(getattr(args, "neg_weight_use_margin", True)):
                    m0 = float(np.clip(getattr(args, "neg_weight_margin_center", 0.15), 0.0, 1.0))
                    tm = float(max(getattr(args, "neg_weight_margin_temp", 0.05), 1e-6))
                    wm = _sigmoid((m0 - m_glo_raw[pool_idx]) / tm)
                else:
                    wm = np.ones_like(wa, dtype=np.float32)
                neg_rel_raw = np.clip((wa * ws * wm).astype(np.float32), 1e-8, None)
                if neg_stab_used and bool(getattr(args, "neg_stab_soft_weight", True)):
                    w_floor = float(np.clip(getattr(args, "neg_stab_weight_floor", 0.05), 1e-6, 1.0))
                    w_pow = float(max(getattr(args, "neg_stab_weight_power", 1.0), 1e-6))
                    w_stab = np.clip(neg_stab_freq[pool_idx], w_floor, 1.0) ** w_pow
                    neg_rel_raw = np.clip(neg_rel_raw * w_stab.astype(np.float32), 1e-8, None)
                if use_ess_target:
                    ess_target = float(max(getattr(args, "neg_ess_target_value", 32.0), 1.0))
                    t_min = float(max(getattr(args, "neg_ess_temp_min", 0.05), 1e-4))
                    t_max = float(max(getattr(args, "neg_ess_temp_max", 2.0), t_min + 1e-6))
                    t_iters = int(max(1, getattr(args, "neg_ess_temp_iters", 24)))
                    neg_weights, neg_ess, neg_ess_temp_used = _weights_with_ess_target(
                        np.log(neg_rel_raw),
                        target_ess=min(ess_target, float(neg_pool_size)),
                        temp_min=t_min,
                        temp_max=t_max,
                        n_iter=t_iters,
                    )
                else:
                    neg_weights = neg_rel_raw
            else:
                neg_weights = np.ones(neg_pool_size, dtype=np.float32)
                if neg_stab_used and bool(getattr(args, "neg_stab_soft_weight", True)):
                    w_floor = float(np.clip(getattr(args, "neg_stab_weight_floor", 0.05), 1e-6, 1.0))
                    w_pow = float(max(getattr(args, "neg_stab_weight_power", 1.0), 1e-6))
                    w_stab = np.clip(neg_stab_freq[pool_idx], w_floor, 1.0) ** w_pow
                    neg_weights = np.clip(w_stab.astype(np.float32), 1e-8, None)
            neg_ess = _ess_from_weights(neg_weights)
            neg_weight_mean = float(np.mean(neg_weights))
            neg_stab_pool_mean = float(np.mean(neg_stab_freq[pool_idx])) if neg_stab_used else 1.0
            neg_stab_pool_min = float(np.min(neg_stab_freq[pool_idx])) if neg_stab_used else 1.0
        else:
            neg_weights = np.zeros(0, dtype=np.float32)

        trigger_ok = False
        if use_soft_ess:
            ess_min = float(max(getattr(args, "neg_ess_min", 32.0), 1.0))
            trigger_ok = bool(neg_ess >= ess_min)
        else:
            trigger_ok = bool(neg_pool_size >= min_pool)

        if trigger_ok and neg_pool_size > 0:
            k_neg_req = int(max(1, getattr(args, "neg_k", 10)))
            neg_k_used = int(min(k_neg_req, max(1, neg_pool_size)))
            if use_soft_ess:
                sel_scale = float(max(getattr(args, "neg_soft_select_scale", 2.0), 1.0))
                ess_min = float(max(getattr(args, "neg_ess_min", 32.0), 1.0))
                sel_n = int(max(neg_k_used * 4, np.ceil(sel_scale * ess_min)))
                sel_n = int(min(max(sel_n, neg_k_used), neg_pool_size))
                sel_rel = np.argsort(-neg_weights)[:sel_n]
                neg_fit_idx = pool_idx[sel_rel]
                w_fit = neg_weights[sel_rel]
            else:
                neg_fit_idx = pool_idx
                w_fit = neg_weights
            neg_fit_size = int(neg_fit_idx.size)
            if neg_fit_size > 0:
                neg_fit_ood_ratio = float(np.mean((~id_mask[neg_fit_idx]).astype(np.float32)))

            _, neg_centers = _kmeans(
                h[neg_fit_idx],
                n_clusters=neg_k_used,
                seed=int(getattr(args, "seed", 0)) + 31,
                niter=int(getattr(args, "kmeans_niter", 100)),
            )
            neg_centers = _l2_normalize(neg_centers)
            s_glo_neg = np.max(h @ neg_centers.T, axis=1)
            margin_scale = float(getattr(args, "neg_margin_scale", 1.0))
            base_strength = float(margin_scale)
            if bool(getattr(args, "neg_adaptive_margin", False)) and use_soft_ess:
                ess_target = float(max(getattr(args, "neg_ess_target", 32.0), 1.0))
                ess_temp = float(max(getattr(args, "neg_ess_temp", 8.0), 1e-6))
                ess_gate = float(_sigmoid((neg_ess - ess_target) / ess_temp))
                purity = float(np.mean(np.clip(w_fit, 0.0, 1.0))) if w_fit.size > 0 else 0.0
                base_strength = float(base_strength * ess_gate * purity)

            neg_margin_effective = float(base_strength)
            s_glo_raw = s_glo_raw - neg_margin_effective * s_glo_neg
            neg_bank_used = bool(neg_margin_effective > 1e-8)

    s_glo = _minmax01(s_glo_raw)
    m_glo = m_glo_raw
    local_top1 = np.clip(np.asarray(local_stats.get("top1", s_loc), dtype=np.float32).reshape(-1), 0.0, 1.0)
    if local_top1.shape[0] != n:
        local_top1 = np.clip(s_loc, 0.0, 1.0).astype(np.float32)
    local_margin = np.clip(np.asarray(local_stats.get("margin", m_loc), dtype=np.float32).reshape(-1), 0.0, 1.0)
    if local_margin.shape[0] != n:
        local_margin = np.clip(m_loc, 0.0, 1.0).astype(np.float32)
    local_entropy = np.clip(np.asarray(local_stats.get("entropy", np.ones(n, dtype=np.float32)), dtype=np.float32).reshape(-1), 0.0, 1.0)
    if local_entropy.shape[0] != n:
        local_entropy = np.ones(n, dtype=np.float32)
    # If local scalar score is overridden (full-shot step), keep top1/margin aligned.
    local_top1 = np.clip(s_loc, 0.0, 1.0).astype(np.float32)
    local_margin = np.clip(m_loc, 0.0, 1.0).astype(np.float32)

    gi_ratio = np.zeros(n, dtype=np.float32)
    if c_in > 1:
        top2_star = np.argsort(-sims_star, axis=1)[:, :2]
        sim_nearest = sims_star[np.arange(n), top2_star[:, 0]]
        sim_second = sims_star[np.arange(n), top2_star[:, 1]]
        delta = np.sqrt(np.clip(2.0 - 2.0 * sim_nearest, 0.0, None))
        beta = np.sqrt(np.clip(2.0 - 2.0 * sim_second, 0.0, None))
        gi_ratio = (beta - delta) / (delta + 1e-12)

    # (4) Fusion.
    fusion_solver = str(getattr(args, "fusion_weight_solver", "mse"))
    cal_pos_size = 0
    cal_neg_size = 0
    cal_neg_mode = "none"
    if fusion_weight_override is not None:
        w_scalar = float(np.clip(float(fusion_weight_override), 0.0, 1.0))
    else:
        pos_idx, neg_idx, cal_neg_mode = _build_pseudo_calibration_sets(
            s_loc=s_loc,
            s_glo=s_glo,
            gi_ratio=gi_ratio,
            tta_agree=tta_agree,
            tta_stab=tta_stab,
            args=args,
        )
        cal_pos_size = int(pos_idx.size)
        cal_neg_size = int(neg_idx.size)
        w_mse = _solve_fusion_w_mse(s_loc, s_glo, pos_idx, neg_idx)

        if fusion_solver == "fpr95_grid":
            w_scalar = _solve_fusion_w_fpr95_grid(
                s_loc=s_loc,
                s_glo=s_glo,
                pos_idx=pos_idx,
                neg_idx=neg_idx,
                args=args,
                w_prior=w_mse,
            )
        else:
            w_scalar = w_mse

        # Keep zero-shot protective cap/floor only for legacy MSE solver.
        if fusion_solver != "fpr95_grid" and int(getattr(args, "shot", 0)) == 0:
            loc_std = float(np.std(s_loc))
            w_cap = float(np.clip(getattr(args, "w_cap", 0.05), 0.0, 1.0))
            w_floor = float(np.clip(getattr(args, "w_floor", 0.05), 0.0, 1.0))
            if w_cap < 1.0:
                w_scalar = min(w_scalar, w_cap)
            if w_scalar < 1e-8 and loc_std >= float(getattr(args, "w_floor_loc_std", 0.08)):
                w_scalar = max(w_scalar, min(w_floor, w_cap))
        w_scalar = float(np.clip(w_scalar, 0.0, 1.0))

    w_min_sw = 0.0
    w_star = w_scalar
    w_vec = np.full(n, w_scalar, dtype=np.float32)
    id_score = w_star * s_loc + (1.0 - w_star) * s_glo
    geo_used = False
    geo_weight = 0.0
    geo_weight_mean = 0.0
    geo_weight_std = 0.0
    s_geo = None
    if bool(getattr(args, "use_geo_signal", False)) and geo_score is not None:
        g = np.asarray(geo_score, dtype=np.float32).reshape(-1)
        if g.shape[0] == n:
            s_geo = np.clip(_minmax01(g), 0.0, 1.0).astype(np.float32)
            geo_weight = float(np.clip(getattr(args, "geo_fusion_weight", 0.20), 0.0, 1.0))
            if geo_weight > 1e-8:
                if bool(getattr(args, "geo_adaptive_weight", True)):
                    p = float(max(getattr(args, "geo_adaptive_power", 1.0), 1e-6))
                    floor = float(np.clip(getattr(args, "geo_adaptive_floor", 0.0), 0.0, 1.0))
                    conf = np.clip(s_glo_pos, 0.0, 1.0).astype(np.float32)
                    geo_w_vec = geo_weight * np.clip(np.power(1.0 - conf, p), floor, 1.0)
                else:
                    geo_w_vec = np.full(n, geo_weight, dtype=np.float32)
                id_score = (1.0 - geo_w_vec) * id_score + geo_w_vec * s_geo
                geo_weight_mean = float(np.mean(geo_w_vec))
                geo_weight_std = float(np.std(geo_w_vec))
                geo_used = True
    ood_score = 1.0 - id_score

    y_true_ood = (~id_mask).astype(np.int64)
    if np.unique(y_true_ood).size < 2:
        auroc = float("nan")
        fpr95 = float("nan")
    else:
        auroc = float(roc_auc_score(y_true_ood, ood_score))
        fpr95 = compute_fpr95(y_true_ood, ood_score)

    return {
        "auroc": auroc,
        "fpr95": fpr95,
        "w_star": w_star,
        "w_scalar": w_scalar,
        "w_mean": float(np.mean(w_vec)),
        "w_std": float(np.std(w_vec)),
        "local_method": str(local_method),
        "local_score_variant": str(local_score_variant),
        "local_score_top1_mean": float(local_score_top1_mean),
        "local_score_margin_mean": float(local_score_margin_mean),
        "local_score_conc_mean": float(local_score_conc_mean),
        "local_score_entropy_mean": float(local_score_entropy_mean),
        "fusion_solver": str(fusion_solver),
        "cal_pos_size": int(cal_pos_size),
        "cal_neg_size": int(cal_neg_size),
        "cal_neg_mode": str(cal_neg_mode),
        "glo_revisit_used": bool(use_revisit),
        "glo_revisit_rounds_used": int(revisit_round_used),
        "glo_revisit_proto_shift": float(revisit_proto_shift),
        "glo_revisit_keep_change": float(revisit_keep_change),
        "glo_revisit_gap_score": float(revisit_gap_score),
        "glo_revisit_recovered": int(revisit_recovered),
        "glo_revisit_excluded": int(revisit_excluded),
        "glo_revisit_class_skip_total": int(revisit_class_skip_total),
        "glo_revisit_class_ess_mean": float(revisit_class_ess_mean),
        "glo_revisit_class_ess_min": float(revisit_class_ess_min),
        "glo_gate_dpam_rate": float(glo_gate_dpam_rate),
        "glo_gate_nr_rate": float(glo_gate_nr_rate),
        "glo_gate_dgis_rate": float(glo_gate_dgis_rate),
        "glo_gate_keep_rate": float(glo_gate_keep_rate),
        "local_top1_mean": float(np.mean(local_top1)),
        "local_margin_mean": float(np.mean(local_margin)),
        "local_entropy_mean": float(np.mean(local_entropy)),
        "global_class_scores": sims_star.astype(np.float32),
        "global_component_scores": sims_star_proto.astype(np.float32),
        "global_pred_class": pred_cls_star.astype(np.int64),
        "s_loc": s_loc,
        "s_glo": s_glo,
        "s_glo_pos": s_glo_pos,
        "global_score_mode": str(global_score_mode),
        "glo_mixture_used": bool(mixture_used),
        "glo_num_proto_per_class": int(m_proto),
        "glo_multi_class_count": int(np.sum(class_multi_mask)),
        "neg_bank_used": neg_bank_used,
        "neg_source_mode": str(neg_source_mode),
        "neg_pool_size": neg_pool_size,
        "neg_k_used": neg_k_used,
        "neg_ess": float(neg_ess),
        "neg_weight_mean": float(neg_weight_mean),
        "neg_margin_effective": float(neg_margin_effective),
        "neg_rank_consistency": float(neg_rank_consistency),
        "neg_beta_effective": float(neg_beta_effective),
        "neg_stab_used": bool(neg_stab_used),
        "neg_stab_views": int(neg_stab_views),
        "neg_stab_thr": float(neg_stab_thr),
        "neg_stab_pool_mean": float(neg_stab_pool_mean),
        "neg_stab_pool_min": float(neg_stab_pool_min),
        "neg_ess_temp_used": float(neg_ess_temp_used),
        "neg_pool_ood_ratio": float(neg_pool_ood_ratio),
        "neg_fit_size": int(neg_fit_size),
        "neg_fit_ood_ratio": float(neg_fit_ood_ratio),
        "neg_hard_used": bool(neg_hard_used),
        "neg_hard_count": int(neg_hard_count),
        "neg_hard_ood_ratio": float(neg_hard_ood_ratio),
        "single_neg_used": bool(single_neg_used),
        "single_neg_pool_size": int(single_neg_pool_size),
        "single_neg_pool_ood_ratio": float(single_neg_pool_ood_ratio),
        "single_neg_ess": float(single_neg_ess),
        "single_neg_beta_eff": float(single_neg_beta_eff),
        "single_neg_lambda": float(single_neg_lambda),
        "single_neg_mu_id": float(single_neg_mu_id),
        "single_neg_tau_id": float(single_neg_tau_id),
        "tta_views": tta_views,
        "tta_agree_mean": float(np.mean(tta_agree)),
        "tta_stab_mean": float(np.mean(tta_stab)),
        "tta_filter_used": tta_filter_used,
        "tta_pool_size": int(tta_pool_size),
        "graph_edge_count": int(graph_edge_count),
        "geo_used": bool(geo_used),
        "geo_weight": float(geo_weight),
        "geo_weight_mean": float(geo_weight_mean),
        "geo_weight_std": float(geo_weight_std),
        "s_geo": s_geo,
        "id_score": id_score,
        "ood_score": ood_score,
        "y_true_ood": y_true_ood,
        "oracle_true_id_proto": bool(oracle_true_id_proto),
        "oracle_true_ood_neg_bank": bool(oracle_true_ood_neg_bank),
    }
