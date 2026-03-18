import torch


def l2_normalize(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def top1_top2(scores):
    vals, idx = torch.topk(scores, k=min(2, scores.shape[1]), dim=1)
    top1 = vals[:, 0]
    if vals.shape[1] > 1:
        top2 = vals[:, 1]
    else:
        top2 = torch.zeros_like(top1)
    margin = (top1 - top2).clamp(min=0.0)
    return top1, top2, idx[:, 0], margin


def confidence_mask(scores, ood_score, conf_threshold, margin_threshold, ood_threshold):
    conf, _, pred, margin = top1_top2(scores)
    mask = (
        (conf >= float(conf_threshold))
        & (margin >= float(margin_threshold))
        & (ood_score <= float(ood_threshold))
    )
    return mask, pred, conf, margin


def init_student_prototypes(
    features,
    scores,
    ood_score,
    fallback_proto,
    conf_threshold,
    margin_threshold,
    ood_threshold,
    min_samples=1,
):
    """
    Initialize student visual prototypes from confident ID-like samples.
    """
    x = l2_normalize(features)
    q = l2_normalize(fallback_proto.clone())
    c = int(q.shape[0])
    mask, pred, _, _ = confidence_mask(
        scores,
        ood_score,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
        ood_threshold=ood_threshold,
    )

    counts = []
    min_n = int(max(1, min_samples))
    for cls_id in range(c):
        cls_mask = mask & (pred == cls_id)
        cnt = int(cls_mask.sum().item())
        counts.append(cnt)
        if cnt >= min_n:
            center = x[cls_mask].mean(dim=0, keepdim=True)
            q[cls_id] = l2_normalize(center, dim=-1)[0]
    stats = {
        "selected_counts": counts,
        "selected_total": int(sum(counts)),
    }
    return q, stats


def update_student_prototypes_ema(
    prev_proto,
    features,
    scores,
    ood_score,
    conf_threshold,
    margin_threshold,
    ood_threshold,
    ema=0.2,
    min_samples=1,
):
    """
    EMA update from confident class-consistent samples only.
    """
    x = l2_normalize(features)
    q_prev = l2_normalize(prev_proto)
    q_new = q_prev.clone()
    c = int(q_prev.shape[0])

    mask, pred, conf, _ = confidence_mask(
        scores,
        ood_score,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
        ood_threshold=ood_threshold,
    )

    alpha = float(max(ema, 0.0))
    min_n = int(max(1, min_samples))
    counts = []
    for cls_id in range(c):
        cls_mask = mask & (pred == cls_id)
        cnt = int(cls_mask.sum().item())
        counts.append(cnt)
        if cnt < min_n:
            continue
        w = conf[cls_mask]
        w = w / w.sum().clamp(min=1e-12)
        center = (x[cls_mask] * w[:, None]).sum(dim=0, keepdim=True)
        center = l2_normalize(center, dim=-1)[0]
        q_new[cls_id] = l2_normalize((1.0 - alpha) * q_prev[cls_id] + alpha * center, dim=-1)

    drift = (1.0 - (q_prev * q_new).sum(dim=1)).mean().item()
    stats = {
        "updated_counts": counts,
        "updated_total": int(sum(counts)),
        "drift": float(drift),
    }
    return q_new, stats


def confidence_vector(scores):
    conf, _, _, _ = top1_top2(scores)
    return conf
