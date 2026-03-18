import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .logofuse import (
    compute_fpr95,
    run_logofuse,
    _split_masks_modelnet40,
    _split_masks_scanobjectnn15,
    _split_masks_shapenetcore54,
)
from utils.ot_utils import compute_ot_scores, l2_normalize as ot_l2_normalize, row_minmax
from utils.prototype_refine import (
    confidence_vector,
    init_student_prototypes,
    top1_top2,
    update_student_prototypes_ema,
)


def _resolve_id_mask(y_gt, args):
    if args.dataset_name == "ScanObjectNN15":
        id_mask, _ = _split_masks_scanobjectnn15(y_gt, args.dataset_split)
    elif args.dataset_name == "ShapeNetCore54":
        id_mask, _ = _split_masks_shapenetcore54(y_gt, args.dataset_split)
    elif args.dataset_name == "ModelNet40":
        id_mask, _ = _split_masks_modelnet40(y_gt, args.dataset_split)
    else:
        id_mask = np.ones_like(y_gt, dtype=bool)
    return id_mask


def _component_weights(args):
    w_cos = float(getattr(args, "w_cos", 0.35))
    w_ot = float(getattr(args, "w_ot", 0.30))
    w_gsp = float(getattr(args, "w_gsp", 0.25))
    w_stu = float(getattr(args, "w_stu", 0.10))

    mode = str(getattr(args, "gsp_ot_ablation", "full")).lower()
    if mode == "cosine_only":
        w_cos, w_ot, w_gsp, w_stu = 1.0, 0.0, 0.0, 0.0
    elif mode == "gsp_only":
        w_cos, w_ot, w_gsp, w_stu = 0.0, 0.0, 1.0, 0.0
    elif mode == "ot_only":
        w_cos, w_ot, w_gsp, w_stu = 0.0, 1.0, 0.0, 0.0
    elif mode == "cosine_gsp":
        w_cos, w_ot, w_gsp, w_stu = 0.5, 0.0, 0.5, 0.0
    elif mode == "ot_gsp":
        w_cos, w_ot, w_gsp, w_stu = 0.0, 0.5, 0.5, 0.0
    elif mode == "cosine_ot_gsp":
        w_cos, w_ot, w_gsp, w_stu = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0

    vals = np.asarray([w_cos, w_ot, w_gsp, w_stu], dtype=np.float32)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= 1e-12:
        vals = np.asarray([0.35, 0.30, 0.25, 0.10], dtype=np.float32)
        s = float(vals.sum())
    vals = vals / max(s, 1e-12)
    return {
        "w_cos": float(vals[0]),
        "w_ot": float(vals[1]),
        "w_gsp": float(vals[2]),
        "w_stu": float(vals[3]),
        "ablation": mode,
    }


def _ood_from_class_scores(scores, args):
    top1, top2, pred, margin = top1_top2(scores)
    lam = float(getattr(args, "lambda_margin", 0.20))
    eps = float(max(getattr(args, "eps", 1e-6), 1e-12))
    ood = -top1 + lam / (margin + eps)
    return top1, margin, pred, ood


def run_gsp_ot(
    features,
    targets,
    text_prototypes,
    text_proto_clusters,
    args,
    fewshot_prototypes=None,
    fusion_weight_override=None,
    geo_score=None,
    train_anchor_features=None,
    token_features=None,
):
    """
    OT-enhanced branch on top of baseline GSP/LoGo-Fuse pipeline.
    Baseline behavior remains untouched when method != gsp_ot.
    """
    baseline = run_logofuse(
        features,
        targets,
        text_prototypes,
        args,
        fewshot_prototypes=fewshot_prototypes,
        fusion_weight_override=fusion_weight_override,
        geo_score=geo_score,
        train_anchor_features=train_anchor_features,
    )

    y_gt = np.asarray(targets, dtype=np.int64)
    id_mask = _resolve_id_mask(y_gt, args)
    y_true_ood = (~id_mask).astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    h = F.normalize(h, dim=-1)

    teacher_cls = ot_l2_normalize(torch.from_numpy(np.asarray(text_prototypes, dtype=np.float32)).to(device), dim=-1)
    teacher_bank = ot_l2_normalize(torch.from_numpy(np.asarray(text_proto_clusters, dtype=np.float32)).to(device), dim=-1)

    gsp_cls_np = baseline.get("global_class_scores", None)
    if gsp_cls_np is None:
        gsp_cls = h @ teacher_cls.T
    else:
        gsp_cls = torch.from_numpy(np.asarray(gsp_cls_np, dtype=np.float32)).to(device)

    cos_cls = h @ teacher_cls.T

    token_feats_t = None
    if token_features is not None:
        token_arr = np.asarray(token_features, dtype=np.float32)
        if token_arr.ndim == 3:
            token_feats_t = ot_l2_normalize(torch.from_numpy(token_arr).to(device), dim=-1)

    ot_scores, ot_info = compute_ot_scores(
        global_feats=h,
        proto_bank=teacher_bank,
        ot_mode=str(getattr(args, "ot_mode", "hybrid")),
        tau=float(getattr(args, "tau", 0.10)),
        sinkhorn_eps=float(getattr(args, "sinkhorn_eps", 0.05)),
        sinkhorn_max_iter=int(getattr(args, "sinkhorn_max_iter", 80)),
        token_feats=token_feats_t,
        class_chunk=int(max(1, getattr(args, "ot_class_chunk", 8))),
    )

    s_cos_cls = row_minmax(cos_cls)
    s_ot_cls = row_minmax(ot_scores)
    s_gsp_cls = row_minmax(gsp_cls)

    if bool(getattr(args, "reweight_graph_with_ot", False)):
        # Optional light reweighting: increase/decrease GSP class confidence with OT confidence.
        ot_conf = torch.max(s_ot_cls, dim=1, keepdim=True).values
        s_gsp_cls = row_minmax(s_gsp_cls * (0.5 + 0.5 * ot_conf))

    w_cfg = _component_weights(args)
    w_cos = w_cfg["w_cos"]
    w_ot = w_cfg["w_ot"]
    w_gsp = w_cfg["w_gsp"]
    w_stu = w_cfg["w_stu"]

    s_seed = w_cos * s_cos_cls + w_ot * s_ot_cls + w_gsp * s_gsp_cls
    _, _, _, ood_seed = _ood_from_class_scores(s_seed, args)

    student_proto, init_stats = init_student_prototypes(
        features=h,
        scores=s_seed,
        ood_score=ood_seed,
        fallback_proto=teacher_cls,
        conf_threshold=float(getattr(args, "conf_threshold", 0.55)),
        margin_threshold=float(getattr(args, "margin_threshold", 0.05)),
        ood_threshold=float(getattr(args, "ood_threshold", 0.0)),
        min_samples=int(max(1, getattr(args, "proto_min_samples", 1))),
    )

    max_iter = int(max(1, getattr(args, "max_iter", 5)))
    tol_score = float(max(getattr(args, "tol_score", 1e-4), 0.0))
    tol_proto = float(max(getattr(args, "tol_proto", 1e-4), 0.0))

    proto_drift_hist = []
    conf_change_hist = []
    score_change_hist = []
    student_update_hist = []

    prev_ood = None
    prev_conf = None
    used_iter = 0

    for it in range(max_iter):
        s_stu_cls = row_minmax(h @ student_proto.T)
        s_final_cls = w_cos * s_cos_cls + w_ot * s_ot_cls + w_gsp * s_gsp_cls + w_stu * s_stu_cls

        id_conf, margin, _, ood_score = _ood_from_class_scores(s_final_cls, args)

        if prev_ood is None:
            score_delta = float("inf")
            conf_delta = float("inf")
        else:
            score_delta = float(torch.mean(torch.abs(ood_score - prev_ood)).item())
            conf_delta = float(torch.mean(torch.abs(id_conf - prev_conf)).item())

        student_new, update_stats = update_student_prototypes_ema(
            prev_proto=student_proto,
            features=h,
            scores=s_final_cls,
            ood_score=ood_score,
            conf_threshold=float(getattr(args, "conf_threshold", 0.55)),
            margin_threshold=float(getattr(args, "margin_threshold", 0.05)),
            ood_threshold=float(getattr(args, "ood_threshold", 0.0)),
            ema=float(np.clip(getattr(args, "proto_ema", 0.2), 0.0, 1.0)),
            min_samples=int(max(1, getattr(args, "proto_min_samples", 1))),
        )
        proto_drift = float(update_stats.get("drift", 0.0))

        proto_drift_hist.append(proto_drift)
        conf_change_hist.append(conf_delta)
        score_change_hist.append(score_delta)
        student_update_hist.append(int(update_stats.get("updated_total", 0)))

        student_proto = student_new
        prev_ood = ood_score.detach()
        prev_conf = id_conf.detach()
        used_iter = it + 1

        if it > 0 and score_delta < tol_score and proto_drift < tol_proto:
            break

    s_stu_cls = row_minmax(h @ student_proto.T)
    s_final_cls = w_cos * s_cos_cls + w_ot * s_ot_cls + w_gsp * s_gsp_cls + w_stu * s_stu_cls
    id_conf, margin, pred_cls, ood_score = _ood_from_class_scores(s_final_cls, args)

    id_score_np = id_conf.detach().cpu().numpy().astype(np.float32)
    ood_score_np = ood_score.detach().cpu().numpy().astype(np.float32)

    if np.unique(y_true_ood).size < 2:
        auroc = float("nan")
        fpr95 = float("nan")
    else:
        auroc = float(roc_auc_score(y_true_ood, ood_score_np))
        fpr95 = float(compute_fpr95(y_true_ood, ood_score_np))

    results = dict(baseline)
    results.update(
        {
            "auroc": auroc,
            "fpr95": fpr95,
            "id_score": id_score_np,
            "ood_score": ood_score_np,
            "y_true_ood": y_true_ood,
            "gsp_ot_used": True,
            "ot_mode_requested": str(getattr(args, "ot_mode", "hybrid")),
            "ot_mode_effective": str(ot_info.get("effective_mode", "softmin_proto")),
            "ot_token_available": bool(ot_info.get("token_available", False)),
            "ot_fallback_reason": str(ot_info.get("fallback_reason", "")),
            "token_ot_available": bool(token_feats_t is not None),
            "gsp_ot_ablation": w_cfg["ablation"],
            "w_cos": float(w_cos),
            "w_ot": float(w_ot),
            "w_gsp": float(w_gsp),
            "w_stu": float(w_stu),
            "lambda_margin": float(getattr(args, "lambda_margin", 0.20)),
            "margin_eps": float(getattr(args, "eps", 1e-6)),
            "max_iter": int(max_iter),
            "iter_used": int(used_iter),
            "tol_score": float(tol_score),
            "tol_proto": float(tol_proto),
            "proto_drift_history": np.asarray(proto_drift_hist, dtype=np.float32),
            "conf_change_history": np.asarray(conf_change_hist, dtype=np.float32),
            "score_change_history": np.asarray(score_change_hist, dtype=np.float32),
            "student_update_history": np.asarray(student_update_hist, dtype=np.int64),
            "student_init_selected_total": int(init_stats.get("selected_total", 0)),
            "s_cos_cls": s_cos_cls.detach().cpu().numpy().astype(np.float32),
            "s_ot_cls": s_ot_cls.detach().cpu().numpy().astype(np.float32),
            "s_gsp_cls": s_gsp_cls.detach().cpu().numpy().astype(np.float32),
            "s_stu_cls": s_stu_cls.detach().cpu().numpy().astype(np.float32),
            "s_final_cls": s_final_cls.detach().cpu().numpy().astype(np.float32),
            "fused_scores_cls": s_final_cls.detach().cpu().numpy().astype(np.float32),
            "student_prototypes": student_proto.detach().cpu().numpy().astype(np.float32),
            "pred_class": pred_cls.detach().cpu().numpy().astype(np.int64),
            "id_conf": id_conf.detach().cpu().numpy().astype(np.float32),
            "margin": margin.detach().cpu().numpy().astype(np.float32),
            "conf_vector": confidence_vector(s_final_cls).detach().cpu().numpy().astype(np.float32),
        }
    )
    return results
