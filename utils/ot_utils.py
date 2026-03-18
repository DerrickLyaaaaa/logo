import torch


def l2_normalize(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def row_minmax(x, eps=1e-12):
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    return (x - mn) / (mx - mn).clamp(min=eps)


def sinkhorn_uniform_cost(cost, epsilon=0.05, max_iter=80):
    """
    Stable entropic OT with uniform marginals.
    cost: [B, M, K]
    return: transport cost [B]
    """
    if cost.ndim != 3:
        raise ValueError(f"sinkhorn cost expects [B,M,K], got {tuple(cost.shape)}")

    bsz, m, k = cost.shape
    if bsz == 0:
        return cost.new_zeros((0,))

    eps = float(max(epsilon, 1e-6))
    log_k = -cost / eps
    log_a = cost.new_full((bsz, m), -torch.log(torch.tensor(float(m), device=cost.device, dtype=cost.dtype)))
    log_b = cost.new_full((bsz, k), -torch.log(torch.tensor(float(k), device=cost.device, dtype=cost.dtype)))

    u = cost.new_zeros((bsz, m))
    v = cost.new_zeros((bsz, k))
    iters = int(max(1, max_iter))
    for _ in range(iters):
        u = log_a - torch.logsumexp(log_k + v[:, None, :], dim=2)
        v = log_b - torch.logsumexp(log_k + u[:, :, None], dim=1)

    log_pi = log_k + u[:, :, None] + v[:, None, :]
    pi = torch.exp(log_pi)
    return (pi * cost).sum(dim=(1, 2))


def softmin_proto_scores(global_feats, proto_bank, tau=0.10):
    """
    Surrogate OT-style class score from global features.
    global_feats: [N, D]
    proto_bank: [C, K, D]
    return: [N, C]
    """
    if global_feats.ndim != 2:
        raise ValueError(f"global_feats expects [N,D], got {tuple(global_feats.shape)}")
    if proto_bank.ndim != 3:
        raise ValueError(f"proto_bank expects [C,K,D], got {tuple(proto_bank.shape)}")

    z = l2_normalize(global_feats, dim=-1)
    p = l2_normalize(proto_bank, dim=-1)
    sim = torch.einsum("nd,ckd->nck", z, p)
    cost = 1.0 - sim
    t = float(max(tau, 1e-6))
    return -t * torch.logsumexp(-cost / t, dim=2)


def token_sinkhorn_scores(token_feats, proto_bank, sinkhorn_eps=0.05, sinkhorn_max_iter=80, class_chunk=8):
    """
    Token-level OT score between visual token sets and class prototype sets.
    token_feats: [N, M, D]
    proto_bank: [C, K, D]
    return: [N, C]
    """
    if token_feats.ndim != 3:
        raise ValueError(f"token_feats expects [N,M,D], got {tuple(token_feats.shape)}")
    if proto_bank.ndim != 3:
        raise ValueError(f"proto_bank expects [C,K,D], got {tuple(proto_bank.shape)}")

    v = l2_normalize(token_feats, dim=-1)
    p = l2_normalize(proto_bank, dim=-1)

    n, _, d = v.shape
    c, _, dp = p.shape
    if d != dp:
        raise ValueError(f"token/proto dim mismatch: {d} vs {dp}")

    out = []
    step = int(max(1, class_chunk))
    for st in range(0, c, step):
        ed = min(st + step, c)
        p_chunk = p[st:ed]  # [Cc, K, D]
        # [N, Cc, M, K]
        sim = torch.einsum("nmd,ckd->ncmk", v, p_chunk)
        cost = 1.0 - sim
        flat = cost.reshape(-1, cost.shape[2], cost.shape[3])
        ot_cost = sinkhorn_uniform_cost(flat, epsilon=sinkhorn_eps, max_iter=sinkhorn_max_iter)
        out.append((-ot_cost).reshape(n, ed - st))
    return torch.cat(out, dim=1)


def compute_ot_scores(
    global_feats,
    proto_bank,
    ot_mode="hybrid",
    tau=0.10,
    sinkhorn_eps=0.05,
    sinkhorn_max_iter=80,
    token_feats=None,
    class_chunk=8,
):
    """
    Compute class OT scores and report effective mode.
    When token features are unavailable, token/hybrid modes auto-fallback to softmin_proto.
    """
    mode = str(ot_mode).lower()
    if mode not in {"token_sinkhorn", "softmin_proto", "hybrid"}:
        mode = "hybrid"

    info = {
        "requested_mode": mode,
        "effective_mode": mode,
        "token_available": token_feats is not None,
        "fallback_reason": "",
    }

    if mode == "softmin_proto":
        return softmin_proto_scores(global_feats, proto_bank, tau=tau), info

    if token_feats is None:
        info["effective_mode"] = "softmin_proto"
        info["fallback_reason"] = "token features unavailable"
        return softmin_proto_scores(global_feats, proto_bank, tau=tau), info

    token_scores = token_sinkhorn_scores(
        token_feats,
        proto_bank,
        sinkhorn_eps=sinkhorn_eps,
        sinkhorn_max_iter=sinkhorn_max_iter,
        class_chunk=class_chunk,
    )
    if mode == "token_sinkhorn":
        info["effective_mode"] = "token_sinkhorn"
        return token_scores, info

    soft_scores = softmin_proto_scores(global_feats, proto_bank, tau=tau)
    token_scores_n = row_minmax(token_scores)
    soft_scores_n = row_minmax(soft_scores)
    info["effective_mode"] = "hybrid"
    return 0.5 * token_scores_n + 0.5 * soft_scores_n, info
