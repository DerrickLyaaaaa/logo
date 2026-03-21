import argparse
import hashlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve

import models.ULIP_models as models
from ood_methods.logofuse import run_logofuse
from utils.tokenizer import SimpleTokenizer
from utils.utils import get_dataset

# Official 3DOS ShapeNetCore54 splits are class-name sets (non-contiguous in canonical label ids).
_SNCORE_SPLIT_CLASS_NAMES = {
    "SN1": [
        "mug", "lamp", "bed", "washer", "loudspeaker", "telephone",
        "dishwasher", "camera", "birdhouse", "jar", "bowl", "bookshelf",
        "stove", "bench", "display", "keyboard", "clock", "piano",
    ],
    "SN2": [
        "earphone", "knife", "chair", "pillow", "table", "laptop",
        "mailbox", "basket", "file cabinet", "cabinet", "sofa", "printer",
        "flowerpot", "microphone", "tower", "bathtub", "bag", "trash bin",
    ],
    "SN3": [
        "can", "microwave", "skateboard", "faucet", "train", "guitar",
        "pistol", "helmet", "watercraft", "airplane", "bottle", "cap",
        "rocket", "rifle", "remote", "car", "bus", "motorbike",
    ],
}


def _indices_from_class_names(all_labels, class_names, split_name):
    label_to_idx = {str(v): i for i, v in enumerate(all_labels)}
    missing = [name for name in class_names if name not in label_to_idx]
    if missing:
        raise ValueError(
            f"{split_name}: missing classes in dataset label space: {missing}. "
            f"Have labels: {all_labels[:10]}... (total {len(all_labels)})"
        )
    return np.asarray([label_to_idx[name] for name in class_names], dtype=np.int64)


def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP + LoGo-Fuse evaluation', add_help=False)

    # Evaluation and dataset args.
    parser.add_argument("--dataset_name", type=str,
                        default="ScanObjectNN15",
                        choices=["ScanObjectNN15", "ShapeNetCore54", "ModelNet40", "S3DIS7"],
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_split", type=str,
                        default="SR1",
                        choices=["SR1", "SR2", "SR3", "SN1", "SN2", "SN3", "MN1", "MN2", "MN3"],
                        help="Name of the dataset split")
    parser.add_argument('--synth2real_official_protocol', dest='synth2real_official_protocol', action='store_true',
                        help='use the official 3DOS synth-to-real protocol (SR1/SR2 only) with split-specific label space')
    parser.add_argument('--no_synth2real_official_protocol', dest='synth2real_official_protocol', action='store_false',
                        help='disable official 3DOS synth-to-real split handling')
    parser.set_defaults(synth2real_official_protocol=False)
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height information')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for test')
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--test_ckpt_addr', default='', type=str, help='ckpt for zero-shot 3d eval')

    # LoGo-Fuse method args.
    parser.add_argument('--method', default='logofuse', choices=['logofuse'])
    parser.add_argument('--cache_features', action='store_true', help='cache ULIP point features to .npz')
    parser.add_argument('--rebuild_feature_cache', action='store_true',
                        help='force rebuild feature cache files for this run (ignore existing .npz)')
    parser.add_argument('--feature_cache_dir', default='./outputs/feature_cache', type=str)
    parser.add_argument('--save_scores', default='', type=str, help='optional .npz path for final scores')
    parser.add_argument('--scanobject_train_dat', default='', type=str,
                        help='optional custom ScanObjectNN train .dat path')
    parser.add_argument('--scanobject_test_dat', default='', type=str,
                        help='optional custom ScanObjectNN test .dat path')

    parser.add_argument('--k', default=10, type=int, help='kNN size')
    parser.add_argument('--local_method', default='lp_softmax', choices=['lp_softmax'],
                        help='local branch method: propagation softmax')
    parser.add_argument('--local_score_variant', default='a0', choices=['a0', 'a1', 'a2', 'a3'],
                        help='local score shaping variant: a0=top1, a1=top1*conc, a2=top1*margin, a3=top1*margin*conc')
    parser.add_argument('--local_score_soft_enable', dest='local_score_soft_enable', action='store_true',
                        help='enable soft shaping for a1/a2/a3: weak penalty + entropy trigger + high-confidence protection')
    parser.add_argument('--no_local_score_soft_enable', dest='local_score_soft_enable', action='store_false',
                        help='disable soft shaping and use direct multiplicative a1/a2/a3')
    parser.set_defaults(local_score_soft_enable=False)
    parser.add_argument('--local_score_soft_lambda', default=0.75, type=float,
                        help='soft shaping factor lambda in s=top1*(lambda+(1-lambda)*factor)')
    parser.add_argument('--local_score_margin_alpha', default=0.5, type=float,
                        help='power alpha on margin for soft a2/a3')
    parser.add_argument('--local_score_conc_beta', default=0.5, type=float,
                        help='power beta on concentration for soft a1/a3')
    parser.add_argument('--local_score_entropy_gate', default=0.55, type=float,
                        help='apply soft shaping only when entropy >= this threshold')
    parser.add_argument('--local_score_protect_q', default=0.60, type=float,
                        help='protect top local-confidence points above this top1 quantile from shaping')
    parser.add_argument('--local_score_protect_min_n', default=16, type=int,
                        help='minimum sample count to enable quantile-based high-confidence protection')
    parser.add_argument('--gsp_local_anchor_k', default=10, type=int,
                        help='number of nearest test points linked by each local GSP anchor')
    parser.add_argument('--gsp_local_alpha', default=0.5, type=float,
                        help='propagation alpha in local GSP graph score')
    parser.add_argument('--gsp_local_steps', default=0, type=int,
                        help='propagation steps in local GSP graph score (0 means ceil(log N))')
    parser.add_argument('--local_seed_source', default='init_proto', choices=['init_proto', 'updated_proto'],
                        help='prototype source for local seeds: initial proto or global-updated proto')
    parser.add_argument('--B', default=10, type=int, help='number of noisy graph views')
    parser.add_argument('--T_lp', default=5, type=int, help='label propagation steps')
    parser.add_argument('--alpha', default=0.5, type=float, help='LP injection coefficient')
    parser.add_argument('--stable_pi', default=0.6, type=float, help='stable-edge frequency threshold')
    parser.add_argument('--temp', default=0.07, type=float, help='stable graph edge temperature')
    parser.add_argument('--seed_temp', default=0.07, type=float, help='seed affinity temperature')
    parser.add_argument('--local_bimodal_balance', dest='local_bimodal_balance', action='store_true',
                        help='enable per-modality balanced local propagation (point-point + point-text)')
    parser.add_argument('--no_local_bimodal_balance', dest='local_bimodal_balance', action='store_false',
                        help='disable per-modality balanced local propagation')
    parser.set_defaults(local_bimodal_balance=False)
    parser.add_argument('--local_bimodal_k_text', default=2, type=int,
                        help='top-k text neighbors per point for bimodal local propagation')
    parser.add_argument('--local_bimodal_pp_weight', default=0.5, type=float,
                        help='point-point contribution weight in bimodal local propagation')
    parser.add_argument('--local_bimodal_pt_weight', default=0.5, type=float,
                        help='point-text contribution weight in bimodal local propagation')
    parser.add_argument('--graph_noise_std', default=0.01, type=float, help='feature perturbation std for stable graph')
    parser.add_argument('--graph_mutual_knn', dest='graph_mutual_knn', action='store_true',
                        help='keep only mutual-kNN edges when building stable graph')
    parser.add_argument('--no_graph_mutual_knn', dest='graph_mutual_knn', action='store_false',
                        help='disable mutual-kNN filtering in stable graph')
    parser.set_defaults(graph_mutual_knn=False)
    parser.add_argument('--graph_local_scaling', dest='graph_local_scaling', action='store_true',
                        help='use self-tuning local-scaling kernel for graph edge weights')
    parser.add_argument('--no_graph_local_scaling', dest='graph_local_scaling', action='store_false',
                        help='disable local-scaling kernel in stable graph')
    parser.set_defaults(graph_local_scaling=False)
    parser.add_argument('--graph_local_scaling_k', default=0, type=int,
                        help='k-th neighbor for local-scaling sigma (0 means reuse --k)')
    parser.add_argument('--graph_score_prune', dest='graph_score_prune', action='store_true',
                        help='enable score-discontinuity edge pruning: remove edges with |g_i-g_j| > delta')
    parser.add_argument('--no_graph_score_prune', dest='graph_score_prune', action='store_false',
                        help='disable score-discontinuity edge pruning')
    parser.set_defaults(graph_score_prune=False)
    parser.add_argument('--graph_score_prune_delta', default=0.25, type=float,
                        help='score discontinuity threshold delta for graph edge pruning')
    parser.add_argument('--graph_score_use_minmax', dest='graph_score_use_minmax', action='store_true',
                        help='min-max normalize seed global score before graph score gating')
    parser.add_argument('--no_graph_score_use_minmax', dest='graph_score_use_minmax', action='store_false',
                        help='use raw cosine score (without min-max) for graph score gating')
    parser.set_defaults(graph_score_use_minmax=True)
    parser.add_argument('--graph_intra_aug', dest='graph_intra_aug', action='store_true',
                        help='enable intra-edge augmentation on top/bottom score anchors')
    parser.add_argument('--no_graph_intra_aug', dest='graph_intra_aug', action='store_false',
                        help='disable intra-edge augmentation')
    parser.set_defaults(graph_intra_aug=False)
    parser.add_argument('--graph_intra_aug_q', default=0.10, type=float,
                        help='anchor fraction q for intra-edge augmentation (top q and bottom q)')
    parser.add_argument('--graph_intra_aug_k', default=5, type=int,
                        help='intra-anchor kNN size for graph augmentation')
    parser.add_argument('--graph_intra_aug_mutual', dest='graph_intra_aug_mutual', action='store_true',
                        help='use mutual-kNN when adding intra-anchor edges')
    parser.add_argument('--no_graph_intra_aug_mutual', dest='graph_intra_aug_mutual', action='store_false',
                        help='use directed kNN when adding intra-anchor edges')
    parser.set_defaults(graph_intra_aug_mutual=True)

    parser.add_argument('--K_over', default=150, type=int, help='over-clustering K')
    parser.add_argument('--kmeans_niter', default=100, type=int, help='FAISS kmeans iterations')

    parser.add_argument('--T_p', default=5, type=int, help='dynamic prototype update steps')
    parser.add_argument('--eta', default=0.2, type=float, help='prototype update momentum')
    parser.add_argument('--glo_update_topk_per_proto', default=0, type=int,
                        help='if >0, each update keeps only top-k highest-rank seed points per prototype component')
    parser.add_argument('--q_frac', default=0.3, type=float, help='NR top-q fraction')
    parser.add_argument('--tau_GI', default=0.0, type=float, help='DGIS margin threshold')
    parser.add_argument('--lambda_cons', default=15.0, type=float, help='consistency weighting scale')
    parser.add_argument('--rbo_p', default=0.9, type=float, help='RBO persistence parameter')
    parser.add_argument('--glo_use_iterative_revisit', dest='glo_use_iterative_revisit', action='store_true',
                        help='enable outer iterative revisit rounds for global prototype refinement')
    parser.add_argument('--no_glo_use_iterative_revisit', dest='glo_use_iterative_revisit', action='store_false',
                        help='disable outer iterative revisit rounds for global refinement')
    parser.set_defaults(glo_use_iterative_revisit=False)
    parser.add_argument('--glo_revisit_rounds', default=2, type=int,
                        help='maximum outer revisit rounds after initial global update pass')
    parser.add_argument('--glo_revisit_min_rounds', default=1, type=int,
                        help='minimum revisit rounds before early-stop checks')
    parser.add_argument('--glo_revisit_reassign_clusters', dest='glo_revisit_reassign_clusters', action='store_true',
                        help='reassign over-cluster pseudo labels with updated prototypes between revisit rounds')
    parser.add_argument('--no_glo_revisit_reassign_clusters', dest='glo_revisit_reassign_clusters', action='store_false',
                        help='keep fixed over-cluster pseudo labels across revisit rounds')
    parser.set_defaults(glo_revisit_reassign_clusters=True)
    parser.add_argument('--glo_revisit_proto_shift_tol', default=5e-4, type=float,
                        help='early-stop threshold on mean prototype cosine shift between revisit rounds')
    parser.add_argument('--glo_revisit_keep_change_tol', default=0.01, type=float,
                        help='early-stop threshold on keep-mask change ratio between revisit rounds')
    parser.add_argument('--glo_revisit_eta_scale', default=0.5, type=float,
                        help='eta scale applied to revisit rounds after the first round')
    parser.add_argument('--glo_revisit_use_stability', dest='glo_revisit_use_stability', action='store_true',
                        help='enable stability selection in each revisit/global pass')
    parser.add_argument('--no_glo_revisit_use_stability', dest='glo_revisit_use_stability', action='store_false',
                        help='disable stability selection in revisit/global pass')
    parser.set_defaults(glo_revisit_use_stability=False)
    parser.add_argument('--glo_revisit_stab_views', default=4, type=int,
                        help='number of perturbation views for revisit stability selection')
    parser.add_argument('--glo_revisit_stab_noise_std', default=0.01, type=float,
                        help='feature perturbation std in revisit stability selection')
    parser.add_argument('--glo_revisit_stab_min_freq', default=0.6, type=float,
                        help='minimum keep frequency across views for stable sample selection')
    parser.add_argument('--glo_revisit_use_stab_weight', dest='glo_revisit_use_stab_weight', action='store_true',
                        help='use keep-frequency as sample weight in prototype update')
    parser.add_argument('--no_glo_revisit_use_stab_weight', dest='glo_revisit_use_stab_weight', action='store_false',
                        help='disable keep-frequency weighting in prototype update')
    parser.set_defaults(glo_revisit_use_stab_weight=True)
    parser.add_argument('--glo_revisit_ess_min', default=12.0, type=float,
                        help='minimum class-wise effective sample size to allow prototype update')
    parser.add_argument('--glo_revisit_gap_q', default=0.10, type=float,
                        help='top/bottom quantile used in revisit confidence-gap early stop')
    parser.add_argument('--glo_revisit_gap_tol', default=0.0, type=float,
                        help='minimum required confidence-gap improvement between rounds')

    parser.add_argument('--calib_pos_frac', default=0.1, type=float, help='top global score fraction as positives')
    parser.add_argument('--calib_neg_frac', default=0.1, type=float, help='bottom min(local,global) as negatives')
    parser.add_argument('--fusion_weight_solver', default='mse', choices=['mse', 'fpr95_grid'],
                        help='solver for fusion scalar weight in run_logofuse')
    parser.add_argument('--fusion_w_grid_steps', default=101, type=int,
                        help='grid steps for fusion FPR95 1D search')
    parser.add_argument('--fusion_w_min', default=0.0, type=float,
                        help='minimum w for fusion FPR95 1D search')
    parser.add_argument('--fusion_w_max', default=1.0, type=float,
                        help='maximum w for fusion FPR95 1D search')
    parser.add_argument('--fusion_fpr_prior_lambda', default=0.0, type=float,
                        help='optional prior shrinkage towards MSE w in fusion FPR95 1D search')
    parser.add_argument('--fusion_use_clean_neg_intersection', dest='fusion_use_clean_neg_intersection', action='store_true',
                        help='build pseudo-OOD negatives by clean intersection: low sloc & low sglo & low GI & low TTA')
    parser.add_argument('--no_fusion_use_clean_neg_intersection', dest='fusion_use_clean_neg_intersection', action='store_false',
                        help='disable clean-intersection pseudo-OOD negatives')
    parser.set_defaults(fusion_use_clean_neg_intersection=False)
    parser.add_argument('--fusion_clean_neg_min', default=8, type=int,
                        help='minimum negatives required before relaxing clean-intersection constraints')
    parser.add_argument('--w_floor', default=0.05, type=float,
                        help='minimum local fusion weight when closed-form collapses in zero-shot')
    parser.add_argument('--w_cap', default=0.05, type=float,
                        help='maximum local fusion weight in zero-shot (set 1.0 to disable cap)')
    parser.add_argument('--w_floor_loc_std', default=0.08, type=float,
                        help='enable w_floor only when std(s_loc) >= this threshold')
    parser.add_argument('--shot', default=0, type=int,
                        help='few-shot K. 0 means zero-shot (default).')
    parser.add_argument('--fewshot_seed', default=0, type=int,
                        help='random seed for few-shot sampling')
    parser.add_argument('--fewshot_align_labels', dest='fewshot_align_labels', action='store_true',
                        help='align train-label ids to test-label ids before few-shot prototype building')
    parser.add_argument('--no_fewshot_align_labels', dest='fewshot_align_labels', action='store_false',
                        help='disable train/test label-id alignment for few-shot')
    parser.set_defaults(fewshot_align_labels=True)
    parser.add_argument('--fewshot_blend_beta', default=0.35, type=float,
                        help='base blend ratio for few-shot prototype in p0 (text/few-shot interpolation)')
    parser.add_argument('--fewshot_blend_ref_shot', default=5.0, type=float,
                        help='reference shot for adaptive few-shot blend ratio')
    parser.add_argument('--text_anchor_lambda', default=0.15, type=float,
                        help='text-anchor strength in dynamic prototype updates')
    parser.add_argument('--fewshot_weight_source', default='support', choices=['support', 'pseudo'],
                        help='how to estimate fusion weight w in few-shot mode')
    parser.add_argument('--fewshot_strict_id_only_weight', dest='fewshot_strict_id_only_weight', action='store_true',
                        help='in few/full-shot, disable support-supervised w fitting and use pseudo-only fusion weight')
    parser.add_argument('--no_fewshot_strict_id_only_weight', dest='fewshot_strict_id_only_weight', action='store_false',
                        help='allow support-supervised w fitting using split-defined ID/OOD labels on train')
    parser.set_defaults(fewshot_strict_id_only_weight=False)
    parser.add_argument('--fewshot_weight_solver', default='blend', choices=['blend', 'map', 'fpr95', 'conformal'],
                        help='few-shot fusion solver: blend(support+pseudo) or MAP closed-form')
    parser.add_argument('--fewshot_weight_blend_rho', default=0.5, type=float,
                        help='blend ratio between support-learned w and pseudo-test w in few-shot mode')
    parser.add_argument('--fewshot_weight_cap', default=0.6, type=float,
                        help='upper bound of few-shot fusion weight after blending')
    parser.add_argument('--fewshot_map_lambda', default=25.0, type=float,
                        help='prior strength lambda in MAP support solver')
    parser.add_argument('--fewshot_support_importance', default='test_affinity',
                        choices=['none', 'test_affinity', 'affinity_margin'],
                        help='support importance weighting mode for MAP support solver')
    parser.add_argument('--fewshot_support_imp_topk', default=16, type=int,
                        help='top-k test neighbors for support affinity weighting')
    parser.add_argument('--fewshot_support_imp_temp', default=0.05, type=float,
                        help='temperature for support affinity weighting')
    parser.add_argument('--fewshot_support_imp_floor', default=0.2, type=float,
                        help='minimum support importance weight before mean-normalization')
    parser.add_argument('--fewshot_support_imp_ceil', default=5.0, type=float,
                        help='maximum support importance weight before mean-normalization')
    parser.add_argument('--fewshot_support_imp_chunk', default=1024, type=int,
                        help='chunk size for support-test affinity matrix multiplication')
    parser.add_argument('--fewshot_fpr_grid_steps', default=101, type=int,
                        help='grid steps for FPR95-opt support solver')
    parser.add_argument('--fewshot_fpr_prior_lambda', default=0.02, type=float,
                        help='prior penalty strength towards pseudo w in FPR95-opt support solver')
    parser.add_argument('--conformal_alpha', default=0.05, type=float,
                        help='target alpha in conformal calibration (0.05 corresponds to TPR~95 percent thresholding)')
    parser.add_argument('--conformal_prior_lambda', default=0.02, type=float,
                        help='prior shrinkage towards pseudo w in conformal w-search')
    parser.add_argument('--glo_use_neg_bank', dest='glo_use_neg_bank', action='store_true',
                        help='use negative prototype bank to build global margin score')
    parser.add_argument('--no_glo_use_neg_bank', dest='glo_use_neg_bank', action='store_false',
                        help='disable negative prototype bank for global score')
    parser.set_defaults(glo_use_neg_bank=False)
    parser.add_argument('--neg_pool_frac', default=0.15, type=float,
                        help='fraction of low-confidence samples used as pseudo-OOD pool')
    parser.add_argument('--neg_k', default=10, type=int,
                        help='number of negative prototypes (k-means on pseudo-OOD pool)')
    parser.add_argument('--neg_margin_scale', default=1.0, type=float,
                        help='scale on max negative similarity in margin score')
    parser.add_argument('--neg_min_pool', default=32, type=int,
                        help='minimum pseudo-OOD pool size to enable negative bank')
    parser.add_argument('--glo_use_single_neg_proto', dest='glo_use_single_neg_proto', action='store_true',
                        help='use one quality-gated negative prototype: s=max_id_sim - beta*sim_to_neg')
    parser.add_argument('--no_glo_use_single_neg_proto', dest='glo_use_single_neg_proto', action='store_false',
                        help='disable one-prototype negative repulsion')
    parser.set_defaults(glo_use_single_neg_proto=False)
    parser.add_argument('--neg_single_replace_bank', dest='neg_single_replace_bank', action='store_true',
                        help='when single-neg is enabled, skip kmeans negative bank')
    parser.add_argument('--no_neg_single_replace_bank', dest='neg_single_replace_bank', action='store_false',
                        help='when single-neg is enabled, still allow kmeans negative bank')
    parser.set_defaults(neg_single_replace_bank=True)
    parser.add_argument('--neg_single_pool_frac', default=0.15, type=float,
                        help='candidate fraction for single negative prototype selection')
    parser.add_argument('--neg_single_min_pool', default=16, type=int,
                        help='minimum pool size to activate single negative prototype')
    parser.add_argument('--neg_single_beta', default=1.0, type=float,
                        help='base repulsion strength for single negative prototype')
    parser.add_argument('--neg_single_use_quality_gate', dest='neg_single_use_quality_gate', action='store_true',
                        help='apply ESS/purity gates to single-negative beta')
    parser.add_argument('--no_neg_single_use_quality_gate', dest='neg_single_use_quality_gate', action='store_false',
                        help='disable quality gating on single-negative beta')
    parser.set_defaults(neg_single_use_quality_gate=True)
    parser.add_argument('--neg_single_ess_min', default=16.0, type=float,
                        help='ESS lower anchor for single-negative quality gate')
    parser.add_argument('--neg_single_ess_target', default=64.0, type=float,
                        help='ESS target anchor for single-negative quality gate')
    parser.add_argument('--neg_single_purity_tau_q', default=0.35, type=float,
                        help='quantile for adaptive purity tau in single-negative quality gate')
    parser.add_argument('--neg_single_purity_delta', default=0.05, type=float,
                        help='delta for single-negative purity gate')
    parser.add_argument('--neg_single_quality_gamma', default=2.0, type=float,
                        help='conservative exponent on single-negative quality lambda')
    parser.add_argument('--neg_single_use_weighted_center', dest='neg_single_use_weighted_center', action='store_true',
                        help='use weighted mean center when building single negative prototype')
    parser.add_argument('--no_neg_single_use_weighted_center', dest='neg_single_use_weighted_center', action='store_false',
                        help='use unweighted mean center for single negative prototype')
    parser.set_defaults(neg_single_use_weighted_center=True)
    parser.add_argument('--neg_rank_use_joint', dest='neg_rank_use_joint', action='store_true',
                        help='use joint ranking (score+distance+disagree) to build neg pool')
    parser.add_argument('--no_neg_rank_use_joint', dest='neg_rank_use_joint', action='store_false',
                        help='disable joint ranking for neg pool')
    parser.set_defaults(neg_rank_use_joint=False)
    parser.add_argument('--neg_rank_gamma', default=0.6, type=float,
                        help='score-vs-distance mix ratio in neg joint ranking')
    parser.add_argument('--neg_rank_beta', default=0.3, type=float,
                        help='disagree bonus in neg joint ranking')
    parser.add_argument('--neg_rank_shot_aware', dest='neg_rank_shot_aware', action='store_true',
                        help='scale joint-ranking beta by shot size and rank consistency')
    parser.add_argument('--no_neg_rank_shot_aware', dest='neg_rank_shot_aware', action='store_false',
                        help='disable shot-aware scaling on joint-ranking beta')
    parser.set_defaults(neg_rank_shot_aware=False)
    parser.add_argument('--neg_rank_ref_shot', default=5.0, type=float,
                        help='reference shot for shot-aware joint-ranking beta scaling')
    parser.add_argument('--neg_rank_r0', default=0.2, type=float,
                        help='minimum rank-consistency anchor for shot-aware beta scaling')
    parser.add_argument('--neg_use_soft_weight_ess', dest='neg_use_soft_weight_ess', action='store_true',
                        help='use soft reliability weights and ESS trigger for neg bank')
    parser.add_argument('--no_neg_use_soft_weight_ess', dest='neg_use_soft_weight_ess', action='store_false',
                        help='disable soft-weight ESS trigger for neg bank')
    parser.set_defaults(neg_use_soft_weight_ess=False)
    parser.add_argument('--neg_ess_min', default=32.0, type=float,
                        help='minimum effective sample size (ESS) to activate neg bank')
    parser.add_argument('--neg_weight_tau_a', default=0.75, type=float,
                        help='TTA-agree center for soft neg reliability')
    parser.add_argument('--neg_weight_tau_s', default=0.85, type=float,
                        help='TTA-stability center for soft neg reliability')
    parser.add_argument('--neg_weight_delta', default=0.08, type=float,
                        help='temperature for soft neg reliability sigmoid')
    parser.add_argument('--neg_weight_use_margin', dest='neg_weight_use_margin', action='store_true',
                        help='include low-margin confidence as extra soft weight for neg reliability')
    parser.add_argument('--no_neg_weight_use_margin', dest='neg_weight_use_margin', action='store_false',
                        help='disable margin factor in soft neg reliability')
    parser.set_defaults(neg_weight_use_margin=True)
    parser.add_argument('--neg_weight_margin_center', default=0.15, type=float,
                        help='center m0 for margin-based neg reliability sigmoid')
    parser.add_argument('--neg_weight_margin_temp', default=0.05, type=float,
                        help='temperature for margin-based neg reliability sigmoid')
    parser.add_argument('--neg_use_ess_target', dest='neg_use_ess_target', action='store_true',
                        help='use temperature-scaled soft weights to match an ESS target instead of hard ESS cutoff')
    parser.add_argument('--no_neg_use_ess_target', dest='neg_use_ess_target', action='store_false',
                        help='disable ESS-target temperature scaling')
    parser.set_defaults(neg_use_ess_target=False)
    parser.add_argument('--neg_ess_target_value', default=32.0, type=float,
                        help='target effective sample size for neg soft weights (per pool)')
    parser.add_argument('--neg_ess_temp_min', default=0.05, type=float,
                        help='minimum temperature for ESS-target binary search')
    parser.add_argument('--neg_ess_temp_max', default=2.0, type=float,
                        help='maximum temperature for ESS-target binary search')
    parser.add_argument('--neg_ess_temp_iters', default=24, type=int,
                        help='binary-search iterations for ESS-target temperature')
    parser.add_argument('--neg_soft_select_scale', default=2.0, type=float,
                        help='candidate count scale from ESS when selecting weighted neg points')
    parser.add_argument('--neg_tta_soft_only', dest='neg_tta_soft_only', action='store_true',
                        help='disable hard TTA neg-pool filtering and rely on soft reliability weights')
    parser.add_argument('--no_neg_tta_soft_only', dest='neg_tta_soft_only', action='store_false',
                        help='allow hard TTA neg-pool filtering when enabled')
    parser.set_defaults(neg_tta_soft_only=False)
    parser.add_argument('--neg_stab_enable', dest='neg_stab_enable', action='store_true',
                        help='enable stability selection for pseudo-negative pool using multi-view bottom frequency')
    parser.add_argument('--no_neg_stab_enable', dest='neg_stab_enable', action='store_false',
                        help='disable stability selection for pseudo-negative pool')
    parser.set_defaults(neg_stab_enable=False)
    parser.add_argument('--neg_stab_views', default=8, type=int,
                        help='number of perturbation views for pseudo-negative stability estimation')
    parser.add_argument('--neg_stab_noise_std', default=0.01, type=float,
                        help='feature noise std for pseudo-negative stability views')
    parser.add_argument('--neg_stab_drop_prob', default=0.0, type=float,
                        help='feature dropout probability for pseudo-negative stability views')
    parser.add_argument('--neg_stab_thr', default=0.8, type=float,
                        help='hard stability threshold on bottom-frequency stab(i)')
    parser.add_argument('--neg_stab_hard', dest='neg_stab_hard', action='store_true',
                        help='hard filter pseudo-negative pool with stab(i) threshold')
    parser.add_argument('--no_neg_stab_hard', dest='neg_stab_hard', action='store_false',
                        help='disable hard stability filter (keep only soft stability weighting)')
    parser.set_defaults(neg_stab_hard=True)
    parser.add_argument('--neg_stab_min_keep', default=8, type=int,
                        help='minimum pseudo-negative size after hard stability filtering')
    parser.add_argument('--neg_stab_soft_weight', dest='neg_stab_soft_weight', action='store_true',
                        help='use stab(i) as multiplicative soft weight for pseudo-negative samples')
    parser.add_argument('--no_neg_stab_soft_weight', dest='neg_stab_soft_weight', action='store_false',
                        help='disable stab(i)-based soft weighting on pseudo-negative samples')
    parser.set_defaults(neg_stab_soft_weight=True)
    parser.add_argument('--neg_stab_weight_floor', default=0.05, type=float,
                        help='minimum stability weight floor before weighting exponent')
    parser.add_argument('--neg_stab_weight_power', default=1.0, type=float,
                        help='exponent on stability soft weight')
    parser.add_argument('--neg_adaptive_margin', dest='neg_adaptive_margin', action='store_true',
                        help='adapt neg margin scale by ESS and reliability purity')
    parser.add_argument('--no_neg_adaptive_margin', dest='neg_adaptive_margin', action='store_false',
                        help='disable adaptive neg margin scaling')
    parser.set_defaults(neg_adaptive_margin=False)
    parser.add_argument('--neg_ess_target', default=32.0, type=float,
                        help='ESS target center for adaptive neg margin scaling')
    parser.add_argument('--neg_ess_temp', default=8.0, type=float,
                        help='ESS temperature for adaptive neg margin scaling')
    parser.add_argument('--neg_use_quality_lambda', dest='neg_use_quality_lambda', action='store_true',
                        help='use quality-driven continuous lambda scaling for neg-bank strength')
    parser.add_argument('--no_neg_use_quality_lambda', dest='neg_use_quality_lambda', action='store_false',
                        help='disable quality-driven lambda scaling for neg-bank strength')
    parser.set_defaults(neg_use_quality_lambda=False)
    parser.add_argument('--neg_lambda_mode', default='legacy', choices=['legacy', 'rneg'],
                        help='continuous lambda mode: legacy(ESS/purity/rank) or rneg(TTA/GI/margin)')
    parser.add_argument('--neg_lambda_ess_min', default=32.0, type=float,
                        help='E_min for lambda_ess clip((ESS-E_min)/(E_tar-E_min))')
    parser.add_argument('--neg_lambda_ess_target', default=80.0, type=float,
                        help='E_tar for lambda_ess clip((ESS-E_min)/(E_tar-E_min))')
    parser.add_argument('--neg_lambda_purity_tau_mode', default='adaptive', choices=['adaptive', 'fixed'],
                        help='tau_id mode for lambda_pur')
    parser.add_argument('--neg_lambda_purity_tau_q', default=0.35, type=float,
                        help='quantile for adaptive tau_id from s_glo_pos')
    parser.add_argument('--neg_lambda_purity_tau_fixed', default=0.40, type=float,
                        help='fixed tau_id for purity gate')
    parser.add_argument('--neg_lambda_purity_margin', default=0.00, type=float,
                        help='extra margin subtracted from tau_id before purity gate')
    parser.add_argument('--neg_lambda_purity_delta', default=0.05, type=float,
                        help='delta for lambda_pur clip((tau_id-mu_id)/delta)')
    parser.add_argument('--neg_lambda_rank_rmin', default=0.2, type=float,
                        help='r_min for lambda_rank clip((r-r_min)/(1-r_min))')
    parser.add_argument('--neg_lambda_gamma', default=2.0, type=float,
                        help='exponent gamma for conservative lambda scaling (lambda^gamma)')
    parser.add_argument('--neg_rneg_alpha', default=0.4, type=float,
                        help='alpha weight for r_tta in rneg lambda mode')
    parser.add_argument('--neg_rneg_beta', default=0.3, type=float,
                        help='beta weight for r_gi in rneg lambda mode')
    parser.add_argument('--neg_rneg_gamma', default=0.3, type=float,
                        help='gamma weight for r_margin in rneg lambda mode')
    parser.add_argument('--neg_rneg_margin_m0', default=0.20, type=float,
                        help='m0 scaling for r_margin = clip(margin/m0,0,1) in rneg mode')
    parser.add_argument('--neg_rneg_r0', default=0.60, type=float,
                        help='r0 center in lambda(r_neg) sigmoid mapping')
    parser.add_argument('--neg_rneg_temp', default=0.10, type=float,
                        help='temperature in lambda(r_neg) sigmoid mapping')
    parser.add_argument('--neg_rneg_lambda_min', default=0.0, type=float,
                        help='minimum lambda in rneg mode')
    parser.add_argument('--neg_rneg_lambda_max', default=1.0, type=float,
                        help='maximum lambda in rneg mode')
    parser.add_argument('--neg_rneg_use_ess_gate', dest='neg_rneg_use_ess_gate', action='store_true',
                        help='multiply rneg lambda by ESS gate clip((ESS-E_min)/(E_tar-E_min),0,1)')
    parser.add_argument('--no_neg_rneg_use_ess_gate', dest='neg_rneg_use_ess_gate', action='store_false',
                        help='disable ESS multiplicative gate in rneg lambda mode')
    parser.set_defaults(neg_rneg_use_ess_gate=True)
    parser.add_argument('--neg_min_pool_fit', default=2, type=int,
                        help='minimum neg pool size to fit negative centers')
    parser.add_argument('--glo_num_proto_per_class', default=1, type=int,
                        help='number of global prototypes per class (1 disables mixture)')
    parser.add_argument('--glo_proto_init_top_frac', default=0.3, type=float,
                        help='top-confidence fraction per class used to initialize mixture prototypes')
    parser.add_argument('--glo_triggered_mixture', dest='glo_triggered_mixture', action='store_true',
                        help='enable per-class triggered mixture (instead of fixed M for all classes)')
    parser.add_argument('--no_glo_triggered_mixture', dest='glo_triggered_mixture', action='store_false',
                        help='disable per-class triggered mixture')
    parser.set_defaults(glo_triggered_mixture=False)
    parser.add_argument('--glo_trigger_delta_thr', default=0.12, type=float,
                        help='minimum relative SSE drop to trigger class-wise mixture')
    parser.add_argument('--glo_trigger_margin_thr', default=0.20, type=float,
                        help='minimum uncertainty (1-margin) to trigger class-wise mixture')
    parser.add_argument('--glo_trigger_disagree_thr', default=0.20, type=float,
                        help='minimum disagree rate to trigger class-wise mixture')
    parser.add_argument('--glo_trigger_min_count', default=12, type=int,
                        help='minimum samples per k=2 cluster when triggering mixture')
    parser.add_argument('--tta_views', default=1, type=int,
                        help='number of TTA views for consistency estimation (1 disables TTA)')
    parser.add_argument('--tta_noise_std', default=0.01, type=float,
                        help='feature-space Gaussian noise std for each TTA view')
    parser.add_argument('--tta_drop_prob', default=0.0, type=float,
                        help='optional feature dropout probability for each TTA view')
    parser.add_argument('--tta_filter_neg_pool', dest='tta_filter_neg_pool', action='store_true',
                        help='filter pseudo-OOD pool with TTA consistency')
    parser.add_argument('--no_tta_filter_neg_pool', dest='tta_filter_neg_pool', action='store_false',
                        help='disable TTA consistency filtering for pseudo-OOD pool')
    parser.set_defaults(tta_filter_neg_pool=False)
    parser.add_argument('--tta_agree_thr', default=0.75, type=float,
                        help='TTA class-consensus threshold for reliable pseudo-OOD')
    parser.add_argument('--tta_stab_thr', default=0.85, type=float,
                        help='TTA margin-stability threshold for reliable pseudo-OOD')
    return parser


def _l2_normalize_torch(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _l2_normalize_np(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _resolve_labels(args, dataset_obj):
    labels_path_candidates = []
    if args.dataset_name == 'ScanObjectNN15':
        labels_path_candidates.append(os.path.join('./data/SR', 'labels.json'))
    elif args.dataset_name == 'ModelNet40':
        labels_path_candidates.append(os.path.join('./data/MN', 'labels.json'))
    labels_path_candidates.append(os.path.join('./data', 'labels.json'))

    all_labels = None
    # For ShapeNetCore54, always trust dataset-provided shape_names to keep class ids
    # aligned with stored .dat labels. External labels.json may use a different order.
    if args.dataset_name == 'ShapeNetCore54' and hasattr(dataset_obj, 'shape_names'):
        all_labels = list(dataset_obj.shape_names)
    else:
        for p in labels_path_candidates:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    data = json.load(f)
                if args.dataset_name in data:
                    all_labels = data[args.dataset_name]
                    break

    if all_labels is None:
        if hasattr(dataset_obj, 'shape_names'):
            all_labels = list(dataset_obj.shape_names)
        else:
            raise ValueError(f'Cannot resolve labels for dataset {args.dataset_name}')

    if args.dataset_name == 'ScanObjectNN15' and bool(getattr(args, 'synth2real_official_protocol', False)):
        if args.dataset_split == 'SR1':
            label_indices = np.arange(0, 5, dtype=np.int64)
            id_class_names = ['chair', 'shelf', 'door', 'sink', 'sofa']
        elif args.dataset_split == 'SR2':
            label_indices = np.arange(0, 4, dtype=np.int64)
            id_class_names = ['bed', 'toilet', 'desk or table', 'display']
        else:
            raise ValueError('Official synth-to-real protocol only supports SR1/SR2')
    elif args.dataset_name == 'ScanObjectNN15':
        if args.dataset_split == 'SR1':
            label_indices = np.arange(0, 5, dtype=np.int64)
        elif args.dataset_split == 'SR2':
            label_indices = np.arange(5, 10, dtype=np.int64)
        elif args.dataset_split == 'SR3':
            label_indices = np.arange(10, 15, dtype=np.int64)
        else:
            raise ValueError(f'Unsupported ScanObjectNN15 split: {args.dataset_split}')
        id_class_names = [all_labels[i] for i in label_indices]
    elif args.dataset_name == 'ShapeNetCore54':
        split_names = _SNCORE_SPLIT_CLASS_NAMES.get(args.dataset_split, None)
        if split_names is None:
            raise ValueError(f'Unsupported ShapeNetCore54 split: {args.dataset_split}')
        label_indices = _indices_from_class_names(all_labels, split_names, args.dataset_split)
        id_class_names = list(split_names)
    elif args.dataset_name == 'ModelNet40':
        if args.dataset_split == 'MN1':
            label_indices = np.arange(0, 13, dtype=np.int64)
        elif args.dataset_split == 'MN2':
            label_indices = np.arange(13, 26, dtype=np.int64)
        elif args.dataset_split == 'MN3':
            label_indices = np.arange(26, 40, dtype=np.int64)
        else:
            raise ValueError(f'Unsupported ModelNet40 split: {args.dataset_split}')
        id_class_names = [all_labels[i] for i in label_indices]
    else:
        label_indices = np.arange(len(all_labels), dtype=np.int64)
        id_class_names = list(all_labels)

    return id_class_names, label_indices, all_labels


def _build_text_prototypes(model, tokenizer, args, class_names, device):
    with open(os.path.join('./data', 'templates.json'), 'r') as f:
        templates_all = json.load(f)
    if args.validate_dataset_prompt not in templates_all:
        raise ValueError(f'Prompt key {args.validate_dataset_prompt} not found in data/templates.json')
    templates = templates_all[args.validate_dataset_prompt]

    protos = []
    with torch.no_grad():
        for cname in class_names:
            texts = [t.format(cname) for t in templates]
            tokenized = tokenizer(texts).to(device, non_blocking=True)
            if tokenized.ndim == 1:
                tokenized = tokenized[None, ...]
            text_feats = model.encode_text(tokenized)
            text_feats = _l2_normalize_torch(text_feats)
            proto = _l2_normalize_torch(text_feats.mean(dim=0, keepdim=True))[0]
            protos.append(proto.detach().cpu().numpy())

    protos = np.stack(protos, axis=0).astype(np.float32)
    protos = _l2_normalize_np(protos)
    return protos


def _cache_file_path(args, split_tag='test'):
    key_raw = (
        f"{args.model}|{args.dataset_name}|{args.dataset_split}|{split_tag}|"
        f"{args.npoints}|{args.test_ckpt_addr}|"
        f"scan_train={str(getattr(args, 'scanobject_train_dat', ''))}|"
        f"scan_test={str(getattr(args, 'scanobject_test_dat', ''))}"
    )
    key = hashlib.md5(key_raw.encode('utf-8')).hexdigest()[:16]
    fname = f"feat_{args.dataset_name}_{args.dataset_split}_{split_tag}_{args.npoints}_{key}.npz"
    return os.path.join(args.feature_cache_dir, fname)


def _minmax01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _entropy_from_hist(hist):
    hist = np.asarray(hist, dtype=np.float32).reshape(-1)
    s = float(np.sum(hist))
    if s <= 0.0:
        return 0.0
    p = np.clip(hist / s, 1e-12, None)
    return float(-np.sum(p * np.log(p)))
def _extract_or_load_features(model, data_loader, device, args, split_tag='test'):
    if args.cache_features:
        os.makedirs(args.feature_cache_dir, exist_ok=True)
        cache_path = _cache_file_path(args, split_tag=split_tag)
        if bool(getattr(args, 'rebuild_feature_cache', False)) and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data['features'], data['targets'], cache_path
    else:
        cache_path = ''

    feats = []
    tgts = []

    model.eval()
    with torch.no_grad():
        for pc, target in data_loader:
            pc = pc.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            h = model.encode_pc(pc)
            h = _l2_normalize_torch(h)

            feats.append(h.detach().cpu().numpy())
            tgts.append(target.detach().cpu().numpy())

    features = np.concatenate(feats, axis=0).astype(np.float32)
    targets = np.concatenate(tgts, axis=0).astype(np.int64)

    if args.cache_features:
        np.savez_compressed(cache_path, features=features, targets=targets)

    return features, targets, cache_path


def _build_fewshot_prototypes(
    features,
    targets,
    id_label_indices,
    shot,
    seed,
    fallback_text_prototypes,
):
    """
    Build class prototypes from labeled few-shot data.
    We sample up to `shot` points per ID class from labeled ID pool L.
    """
    if shot <= 0:
        return None, {}

    rng = np.random.default_rng(seed)
    h = _l2_normalize_np(np.asarray(features, dtype=np.float32))
    y = np.asarray(targets, dtype=np.int64)

    protos = []
    class_stats = {}

    for class_idx, raw_label in enumerate(id_label_indices.tolist()):
        idx = np.where(y == int(raw_label))[0]
        total = int(len(idx))
        if total == 0:
            proto = fallback_text_prototypes[class_idx]
            used = 0
        else:
            take = min(int(shot), total)
            pick = rng.choice(idx, size=take, replace=False)
            x_pick = h[pick]
            proto = _l2_normalize_np(np.mean(x_pick, axis=0, keepdims=True))[0]
            used = int(take)
        protos.append(proto)
        class_stats[int(raw_label)] = {
            "total": total,
            "used": used,
            "clusters": 1 if total > 0 else 0,
        }

    protos = _l2_normalize_np(np.stack(protos, axis=0).astype(np.float32))
    return protos, class_stats


def _align_train_labels_to_test(train_features, train_targets, test_features, test_targets):
    """
    Align train label ids to test label ids via class-centroid matching.
    This fixes dataset variants where train/test label encodings are permuted.
    """
    y_tr = np.asarray(train_targets, dtype=np.int64)
    y_te = np.asarray(test_targets, dtype=np.int64)

    tr_labels = sorted(np.unique(y_tr).tolist())
    te_labels = sorted(np.unique(y_te).tolist())
    common = sorted(set(tr_labels).intersection(set(te_labels)))
    if len(common) < 2:
        return y_tr, {}

    h_tr = _l2_normalize_np(np.asarray(train_features, dtype=np.float32))
    h_te = _l2_normalize_np(np.asarray(test_features, dtype=np.float32))

    valid = []
    ctr_tr = []
    ctr_te = []
    for lab in common:
        idx_tr = np.where(y_tr == lab)[0]
        idx_te = np.where(y_te == lab)[0]
        if len(idx_tr) == 0 or len(idx_te) == 0:
            continue
        ctr_tr.append(_l2_normalize_np(np.mean(h_tr[idx_tr], axis=0, keepdims=True))[0])
        ctr_te.append(_l2_normalize_np(np.mean(h_te[idx_te], axis=0, keepdims=True))[0])
        valid.append(int(lab))

    if len(valid) < 2:
        return y_tr, {}

    sim = np.asarray(ctr_tr, dtype=np.float32) @ np.asarray(ctr_te, dtype=np.float32).T
    row, col = linear_sum_assignment(-sim)

    mapping = {valid[int(r)]: valid[int(c)] for r, c in zip(row, col)}
    remapped = np.asarray([mapping.get(int(v), int(v)) for v in y_tr], dtype=np.int64)
    return remapped, mapping


def _solve_weight_from_labeled_support(s_loc, s_glo, y_true_ood):
    """
    Closed-form fusion weight using true labels from few-shot support set.
    y_true_ood: 1 for OOD, 0 for ID.
    """
    s_loc = np.asarray(s_loc, dtype=np.float32)
    s_glo = np.asarray(s_glo, dtype=np.float32)
    y_id = 1.0 - np.asarray(y_true_ood, dtype=np.float32)

    z = s_loc - s_glo
    y_prime = y_id - s_glo
    denom = float(np.sum(z * z))
    if denom < 1e-12:
        return 0.5
    w = float(np.sum(z * y_prime) / denom)
    return float(np.clip(w, 0.0, 1.0))


def _solve_weight_map_from_labeled_support(s_loc, s_glo, y_true_ood, w_prior, reg_lambda, sample_weights=None):
    """
    MAP-style closed-form fusion:
    min_w sum_i omega_i (y_i - (w*s_loc_i + (1-w)*s_glo_i))^2 + lambda (w - w_prior)^2
    """
    s_loc = np.asarray(s_loc, dtype=np.float32)
    s_glo = np.asarray(s_glo, dtype=np.float32)
    y_id = 1.0 - np.asarray(y_true_ood, dtype=np.float32)
    if sample_weights is None:
        omega = np.ones_like(s_loc, dtype=np.float32)
    else:
        omega = np.asarray(sample_weights, dtype=np.float32)
        if omega.shape[0] != s_loc.shape[0]:
            raise ValueError("sample_weights length mismatch in MAP support solver")
    omega = np.clip(omega, 1e-6, None)

    d = s_loc - s_glo
    e = y_id - s_glo

    reg_lambda = float(max(reg_lambda, 0.0))
    num = float(np.sum(omega * d * e) + reg_lambda * float(w_prior))
    den = float(np.sum(omega * d * d) + reg_lambda)
    if den < 1e-12:
        return float(np.clip(float(w_prior), 0.0, 1.0))
    w = num / den
    return float(np.clip(w, 0.0, 1.0))


def _compute_fpr95(y_true_ood, y_score_ood):
    fpr, tpr, _ = roc_curve(y_true_ood, y_score_ood)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def _solve_weight_fpr95_from_support(s_loc, s_glo, y_true_ood, w_prior, cap, args):
    """
    Grid-search w on [0, cap] by FPR95 objective with prior shrinkage.
    """
    s_loc = np.asarray(s_loc, dtype=np.float32)
    s_glo = np.asarray(s_glo, dtype=np.float32)
    y_true_ood = np.asarray(y_true_ood, dtype=np.int64)
    if np.unique(y_true_ood).size < 2:
        return float(np.clip(float(w_prior), 0.0, float(cap)))

    grid_steps = int(max(11, getattr(args, 'fewshot_fpr_grid_steps', 101)))
    cap = float(np.clip(cap, 0.0, 1.0))
    w_grid = np.linspace(0.0, cap, grid_steps, dtype=np.float32)
    prior_lambda = float(max(getattr(args, 'fewshot_fpr_prior_lambda', 0.02), 0.0))

    best_obj = float('inf')
    best_w = float(w_prior)
    for w in w_grid:
        id_score = w * s_loc + (1.0 - w) * s_glo
        ood_score = 1.0 - id_score
        fpr95 = _compute_fpr95(y_true_ood, ood_score)
        obj = float(fpr95 + prior_lambda * (float(w) - float(w_prior)) ** 2)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_w = float(w)
    return float(np.clip(best_w, 0.0, cap))


def _solve_weight_conformal_from_support(s_loc, s_glo, y_true_ood, w_prior, cap, args):
    """
    Conformal-calibrated 1D search:
    - nonconformity e = 1 - id_score on ID support
    - threshold from finite-sample conformal quantile
    - objective: FPR on OOD support + prior shrinkage
    """
    s_loc = np.asarray(s_loc, dtype=np.float32)
    s_glo = np.asarray(s_glo, dtype=np.float32)
    y_true_ood = np.asarray(y_true_ood, dtype=np.int64)
    idx_pos = np.where(y_true_ood == 0)[0]
    idx_neg = np.where(y_true_ood == 1)[0]
    if idx_pos.size == 0 or idx_neg.size == 0:
        return float(np.clip(float(w_prior), 0.0, float(cap)))

    grid_steps = int(max(11, getattr(args, 'fewshot_fpr_grid_steps', 101)))
    cap = float(np.clip(cap, 0.0, 1.0))
    w_grid = np.linspace(0.0, cap, grid_steps, dtype=np.float32)
    alpha = float(np.clip(getattr(args, 'conformal_alpha', 0.05), 1e-3, 0.5))
    prior_lambda = float(max(getattr(args, 'conformal_prior_lambda', 0.02), 0.0))

    best_obj = float('inf')
    best_w = float(np.clip(float(w_prior), 0.0, cap))
    n_pos = int(idx_pos.size)
    k = int(np.ceil((1.0 - alpha) * (n_pos + 1)))
    k = int(np.clip(k, 1, n_pos))

    for w in w_grid:
        id_score = w * s_loc + (1.0 - w) * s_glo
        e = 1.0 - id_score[idx_pos]
        e_sorted = np.sort(e)
        e_hat = float(e_sorted[k - 1])
        tau_id = 1.0 - e_hat
        fpr = float(np.mean(id_score[idx_neg] >= tau_id))
        obj = float(fpr + prior_lambda * (float(w) - float(w_prior)) ** 2)
        if obj < best_obj - 1e-12:
            best_obj = obj
            best_w = float(w)
    return float(np.clip(best_w, 0.0, cap))


def _compute_support_importance_weights(train_features, test_features, s_loc, s_glo, args):
    mode = str(getattr(args, 'fewshot_support_importance', 'none'))
    n = int(np.asarray(train_features).shape[0])
    if n == 0 or mode == 'none':
        ones = np.ones(n, dtype=np.float32)
        return ones, {'mode': mode, 'min': 1.0, 'max': 1.0, 'mean': 1.0}

    h_tr = _l2_normalize_np(np.asarray(train_features, dtype=np.float32))
    h_te = _l2_normalize_np(np.asarray(test_features, dtype=np.float32))
    m = int(h_te.shape[0])
    if m == 0:
        ones = np.ones(n, dtype=np.float32)
        return ones, {'mode': mode, 'min': 1.0, 'max': 1.0, 'mean': 1.0}

    topk = int(max(1, getattr(args, 'fewshot_support_imp_topk', 16)))
    topk = min(topk, m)
    chunk = int(max(32, getattr(args, 'fewshot_support_imp_chunk', 1024)))

    aff = np.zeros(n, dtype=np.float32)
    for st in range(0, n, chunk):
        ed = min(st + chunk, n)
        sims = h_tr[st:ed] @ h_te.T
        if topk == 1:
            aff[st:ed] = np.max(sims, axis=1)
        else:
            kth = max(0, sims.shape[1] - topk)
            topk_vals = np.partition(sims, kth, axis=1)[:, -topk:]
            aff[st:ed] = np.mean(topk_vals, axis=1)

    temp = float(max(getattr(args, 'fewshot_support_imp_temp', 0.05), 1e-6))
    center = float(np.median(aff))
    w_dom = np.exp((aff - center) / temp).astype(np.float32)
    w_floor = float(max(getattr(args, 'fewshot_support_imp_floor', 0.2), 1e-4))
    w_ceil = float(max(getattr(args, 'fewshot_support_imp_ceil', 5.0), w_floor))
    w_dom = np.clip(w_dom, w_floor, w_ceil)

    if mode == 'affinity_margin':
        margin = np.abs(np.asarray(s_loc, dtype=np.float32) - np.asarray(s_glo, dtype=np.float32))
        q90 = float(np.percentile(margin, 90))
        scale = max(q90, 1e-6)
        w_margin = np.clip(0.5 + margin / scale, 0.5, 2.0).astype(np.float32)
        weights = w_dom * w_margin
    else:
        weights = w_dom

    mean_w = float(np.mean(weights))
    if mean_w < 1e-12:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = weights / mean_w

    stats = {
        'mode': mode,
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'mean': float(np.mean(weights)),
    }
    return weights.astype(np.float32), stats


def _load_model(args, device):
    if not args.test_ckpt_addr:
        raise ValueError('--test_ckpt_addr is required for LoGo-Fuse evaluation')

    try:
        ckpt = torch.load(args.test_ckpt_addr, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict_raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    state_dict = OrderedDict()
    for k, v in state_dict_raw.items():
        state_dict[k.replace('module.', '')] = v

    ckpt_args = ckpt.get('args', None)
    ckpt_model_name = getattr(ckpt_args, 'model', None) if ckpt_args is not None else None

    if hasattr(models, args.model):
        model_name = args.model
    elif ckpt_model_name is not None and hasattr(models, ckpt_model_name):
        model_name = ckpt_model_name
    else:
        raise ValueError(
            f"Neither args.model={args.model} nor ckpt model={ckpt_model_name} exists in models.ULIP_models"
        )

    print(f"[INFO] using model: {model_name}")
    model = getattr(models, model_name)(args=args).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.evaluate_3d:
        raise ValueError('Use --evaluate_3d for LoGo-Fuse evaluation')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this repository for ULIP2 eval')

    if args.gpu is None:
        args.gpu = 0
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = SimpleTokenizer()
    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = _load_model(args, device)

    id_class_names, id_label_indices, _ = _resolve_labels(args, test_dataset)
    text_prototypes = _build_text_prototypes(model, tokenizer, args, id_class_names, device)

    test_features, test_targets, test_cache_path = _extract_or_load_features(
        model, test_loader, device, args, split_tag='test'
    )

    fewshot_prototypes = None
    fewshot_stats = {}
    support_learned_w = None
    support_raw_w = None
    pseudo_test_w = None
    support_weight_stats = {}
    support_results = None
    effective_fewshot_weight_source = str(getattr(args, 'fewshot_weight_source', 'support'))
    if (
        args.shot > 0
        and effective_fewshot_weight_source == 'support'
        and bool(getattr(args, 'fewshot_strict_id_only_weight', False))
    ):
        effective_fewshot_weight_source = 'pseudo'
    train_id_anchor_features = None
    if args.shot > 0:
        train_dataset = get_dataset(None, tokenizer, args, 'train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        train_features, train_targets, train_cache_path = _extract_or_load_features(
            model, train_loader, device, args, split_tag='train'
        )
        if args.fewshot_align_labels:
            train_targets, label_mapping = _align_train_labels_to_test(
                train_features=train_features,
                train_targets=train_targets,
                test_features=test_features,
                test_targets=test_targets,
            )
        else:
            label_mapping = {}
        train_id_anchor_idx = np.where(np.isin(train_targets, id_label_indices))[0].astype(np.int64)
        if train_id_anchor_idx.size > 0:
            train_id_anchor_features = train_features[train_id_anchor_idx]
        fewshot_prototypes, fewshot_stats = _build_fewshot_prototypes(
            features=train_features,
            targets=train_targets,
            id_label_indices=id_label_indices,
            shot=args.shot,
            seed=args.fewshot_seed,
            fallback_text_prototypes=text_prototypes,
        )

        if effective_fewshot_weight_source == 'support':
            support_results = run_logofuse(
                train_features,
                train_targets,
                text_prototypes,
                args,
                fewshot_prototypes=fewshot_prototypes,
            )
    else:
        train_cache_path = ''

    if args.shot > 0 and effective_fewshot_weight_source == 'support' and support_results is not None:
        pseudo_preview = run_logofuse(
            test_features,
            test_targets,
            text_prototypes,
            args,
            fewshot_prototypes=fewshot_prototypes,
            train_anchor_features=train_id_anchor_features,
        )
        pseudo_test_w = float(pseudo_preview['w_star'])

        cap = float(np.clip(args.fewshot_weight_cap, 0.0, 1.0))
        if args.fewshot_weight_solver == 'map':
            support_weights, support_weight_stats = _compute_support_importance_weights(
                train_features=train_features,
                test_features=test_features,
                s_loc=support_results['s_loc'],
                s_glo=support_results['s_glo'],
                args=args,
            )
            support_raw_w = _solve_weight_map_from_labeled_support(
                support_results['s_loc'],
                support_results['s_glo'],
                support_results['y_true_ood'],
                w_prior=pseudo_test_w,
                reg_lambda=float(getattr(args, 'fewshot_map_lambda', 25.0)),
                sample_weights=support_weights,
            )
            support_learned_w = float(np.clip(float(support_raw_w), 0.0, cap))
        elif args.fewshot_weight_solver == 'fpr95':
            support_raw_w = _solve_weight_fpr95_from_support(
                support_results['s_loc'],
                support_results['s_glo'],
                support_results['y_true_ood'],
                w_prior=pseudo_test_w,
                cap=cap,
                args=args,
            )
            support_learned_w = float(np.clip(float(support_raw_w), 0.0, cap))
        elif args.fewshot_weight_solver == 'conformal':
            support_raw_w = _solve_weight_conformal_from_support(
                support_results['s_loc'],
                support_results['s_glo'],
                support_results['y_true_ood'],
                w_prior=pseudo_test_w,
                cap=cap,
                args=args,
            )
            support_learned_w = float(np.clip(float(support_raw_w), 0.0, cap))
        else:
            support_raw_w = _solve_weight_from_labeled_support(
                support_results['s_loc'],
                support_results['s_glo'],
                support_results['y_true_ood'],
            )
            rho = float(np.clip(args.fewshot_weight_blend_rho, 0.0, 1.0))
            mixed_w = (1.0 - rho) * float(support_raw_w) + rho * float(pseudo_test_w)
            support_learned_w = float(np.clip(mixed_w, 0.0, cap))

        results = run_logofuse(
            test_features,
            test_targets,
            text_prototypes,
            args,
            fewshot_prototypes=fewshot_prototypes,
            fusion_weight_override=support_learned_w,
            train_anchor_features=train_id_anchor_features,
        )
    else:
        results = run_logofuse(
            test_features,
            test_targets,
            text_prototypes,
            args,
            fewshot_prototypes=fewshot_prototypes,
            fusion_weight_override=support_learned_w,
            train_anchor_features=train_id_anchor_features,
        )

    print('===== LoGo-Fuse Results =====')
    print(f"dataset: {args.dataset_name} split: {args.dataset_split}")
    print(f"method: {args.method}")
    print(f"local_method: {results.get('local_method', getattr(args, 'local_method', 'lp_softmax'))}")
    print(f"local_score_variant: {results.get('local_score_variant', getattr(args, 'local_score_variant', 'a0'))}")
    print(f"local_seed_source: {results.get('local_seed_source', getattr(args, 'local_seed_source', 'init_proto'))}")
    if 'local_bimodal_used' in results:
        print(
            "local_bimodal: "
            f"used={results.get('local_bimodal_used', False)} "
            f"k_text={results.get('local_bimodal_k_text', 0)} "
            f"w_pp={results.get('local_bimodal_pp_weight', 0.0):.3f} "
            f"w_pt={results.get('local_bimodal_pt_weight', 0.0):.3f}"
        )
    if 'glo_revisit_used' in results:
        print(
            "glo_revisit: "
            f"used={results.get('glo_revisit_used', False)} "
            f"rounds={results.get('glo_revisit_rounds_used', 0)} "
            f"proto_shift={results.get('glo_revisit_proto_shift', 0.0):.6f} "
            f"keep_change={results.get('glo_revisit_keep_change', 0.0):.6f} "
            f"recovered={results.get('glo_revisit_recovered', 0)} "
            f"excluded={results.get('glo_revisit_excluded', 0)}"
        )
        print(
            "glo_revisit_stable: "
            f"gap={results.get('glo_revisit_gap_score', 0.0):.6f} "
            f"class_skip={results.get('glo_revisit_class_skip_total', 0)} "
            f"class_ess_mean={results.get('glo_revisit_class_ess_mean', 0.0):.6f} "
            f"class_ess_min={results.get('glo_revisit_class_ess_min', 0.0):.6f}"
        )
    if 'graph_edge_count' in results:
        print(
            "graph_gate: "
            f"score_prune={results.get('graph_score_prune_used', False)} "
            f"delta={results.get('graph_score_prune_delta', 0.0):.6f} "
            f"intra_aug={results.get('graph_intra_aug_used', False)} "
            f"q={results.get('graph_intra_aug_q', 0.0):.6f} "
            f"k={results.get('graph_intra_aug_k', 0)} "
            f"mutual={results.get('graph_intra_aug_mutual', False)} "
            f"edges={results.get('graph_edge_count', 0)} "
            f"pre_prune={results.get('graph_edge_pre_prune', 0)} "
            f"pruned={results.get('graph_edge_pruned', 0)} "
            f"intra_added={results.get('graph_edge_intra_added', 0)} "
            f"anchor={results.get('graph_anchor_size', 0)}"
        )
    print(f"shot: {args.shot}")
    print(f"global_update: T_p={int(getattr(args, 'T_p', 0))} topk_per_proto={int(getattr(args, 'glo_update_topk_per_proto', 0))}")
    print("global_score: mode=maxcos")
    print(f"AUROC: {results['auroc']:.6f}" if not np.isnan(results['auroc']) else 'AUROC: nan')
    print(f"FPR@TPR95: {results['fpr95']:.6f}" if not np.isnan(results['fpr95']) else 'FPR@TPR95: nan')
    print(f"w*: {results['w_star']:.6f}")
    if 'fusion_solver' in results:
        print(
            "fusion_calib: "
            f"solver={results.get('fusion_solver', 'na')} "
            f"pos={results.get('cal_pos_size', 0)} "
            f"neg={results.get('cal_neg_size', 0)} "
            f"neg_mode={results.get('cal_neg_mode', 'na')}"
        )
    if 'neg_bank_used' in results:
        print(
            "neg_bank_used: "
            f"{results['neg_bank_used']} "
            f"pool={results.get('neg_pool_size', 0)} "
            f"k={results.get('neg_k_used', 0)} "
            f"ess={results.get('neg_ess', 0.0):.3f} "
            f"marg_eff={results.get('neg_margin_effective', 0.0):.4f}"
        )
        if 'neg_ess_temp_used' in results:
            print(f"neg_ess_temp_used: {results.get('neg_ess_temp_used', 1.0):.6f}")
        if 'neg_stab_used' in results:
            print(
                "neg_stability: "
                f"used={results.get('neg_stab_used', False)} "
                f"views={results.get('neg_stab_views', 1)} "
                f"thr={results.get('neg_stab_thr', 0.0):.3f} "
                f"pool_mean={results.get('neg_stab_pool_mean', 1.0):.6f} "
                f"pool_min={results.get('neg_stab_pool_min', 1.0):.6f}"
            )
    if 'single_neg_used' in results:
        print(
            "single_neg: "
            f"used={results.get('single_neg_used', False)} "
            f"pool={results.get('single_neg_pool_size', 0)} "
            f"ess={results.get('single_neg_ess', 0.0):.3f} "
            f"beta_eff={results.get('single_neg_beta_eff', 0.0):.6f} "
            f"lambda={results.get('single_neg_lambda', 0.0):.6f}"
        )
    if bool(results.get('neg_lambda_quality_used', False)):
        if str(results.get('neg_lambda_mode', 'legacy')) == 'rneg':
            print(
                "neg_lambda: "
                f"mode=rneg "
                f"all={results.get('neg_lambda', 0.0):.6f} "
                f"ess={results.get('neg_lambda_ess', 0.0):.6f} "
                f"r_tta={results.get('neg_r_tta', 0.0):.6f} "
                f"r_gi={results.get('neg_r_gi', 0.0):.6f} "
                f"r_m={results.get('neg_r_margin', 0.0):.6f} "
                f"r_neg={results.get('neg_r_neg', 0.0):.6f} "
                f"rank_cons={results.get('neg_rank_consistency', 0.0):.6f} "
                f"beta_eff={results.get('neg_beta_effective', 0.0):.6f}"
            )
        else:
            print(
                "neg_lambda: "
                f"mode=legacy "
                f"all={results.get('neg_lambda', 0.0):.6f} "
                f"ess={results.get('neg_lambda_ess', 0.0):.6f} "
                f"pur={results.get('neg_lambda_pur', 0.0):.6f} "
                f"rank={results.get('neg_lambda_rank', 0.0):.6f} "
                f"mu_id={results.get('neg_mu_id', 0.0):.6f} "
                f"tau_id={results.get('neg_tau_id', 0.0):.6f} "
                f"rank_cons={results.get('neg_rank_consistency', 0.0):.6f} "
                f"beta_eff={results.get('neg_beta_effective', 0.0):.6f}"
            )
    if results.get('glo_mixture_used', False):
        print(
            "glo_mixture: "
            f"num_proto_per_class={results.get('glo_num_proto_per_class', 1)} "
            f"class_multi_count={results.get('glo_multi_class_count', 0)}"
        )
    if 'tta_views' in results and int(results.get('tta_views', 1)) > 1:
        print(
            "tta_stats: "
            f"views={results.get('tta_views', 1)} "
            f"agree_mean={results.get('tta_agree_mean', 0.0):.6f} "
            f"stab_mean={results.get('tta_stab_mean', 0.0):.6f} "
            f"neg_pool_filtered={results.get('tta_filter_used', False)}"
        )
    if args.shot > 0:
        print("fewshot_source: train split (L)")
        print("fewshot_proto_mode: class-mean")
        if args.fewshot_align_labels and label_mapping:
            print(f"fewshot_label_mapping(train->test): {label_mapping}")
        print(f"fewshot_weight_source: {args.fewshot_weight_source}")
        print(f"fewshot_weight_source_effective: {effective_fewshot_weight_source}")
        print(f"fewshot_strict_id_only_weight: {bool(getattr(args, 'fewshot_strict_id_only_weight', False))}")
        if support_learned_w is not None:
            print(f"fewshot_weight_solver: {args.fewshot_weight_solver}")
            if support_raw_w is not None and pseudo_test_w is not None:
                print(f"fewshot_support_w_raw: {support_raw_w:.6f}")
                print(f"fewshot_pseudo_w: {pseudo_test_w:.6f}")
            if support_weight_stats:
                print(
                    "fewshot_support_weight_stats: "
                    f"mode={support_weight_stats.get('mode', 'none')} "
                    f"min={support_weight_stats.get('min', 1.0):.6f} "
                    f"mean={support_weight_stats.get('mean', 1.0):.6f} "
                    f"max={support_weight_stats.get('max', 1.0):.6f}"
                )
            print(f"fewshot_support_w: {support_learned_w:.6f}")
        print(f"fewshot_stats: {fewshot_stats}")
    if args.cache_features:
        print(f"feature_cache_test: {test_cache_path}")
        if args.shot > 0:
            print(f"feature_cache_train: {train_cache_path}")

    if args.save_scores:
        np.savez_compressed(
            args.save_scores,
            ood_score=results['ood_score'],
            id_score=results['id_score'],
            s_loc=results['s_loc'],
            s_glo=results['s_glo'],
            s_geo=results.get('s_geo', None),
            y_true_ood=results['y_true_ood'],
            targets=test_targets,
            features=test_features,
        )
        print(f"saved_scores: {args.save_scores}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP + LoGo-Fuse evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
