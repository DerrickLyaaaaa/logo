import argparse
import hashlib
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_curve

import models.ULIP_models as models
from ood_methods.logofuse import run_logofuse
from ood_methods.gsp_ot import run_gsp_ot
from utils.tokenizer import SimpleTokenizer
from utils.utils import get_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP + LoGo-Fuse evaluation', add_help=False)

    # Baseline-compatible args.
    parser.add_argument("--dataset_name", type=str,
                        default="ScanObjectNN15",
                        choices=["ScanObjectNN15", "ShapeNetCore54", "ModelNet40", "S3DIS7"],
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_split", type=str,
                        default="SR1",
                        choices=["SR1", "SR2", "SR3", "SN1", "SN2", "SN3", "MN1", "MN2", "MN3"],
                        help="Name of the dataset split")
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

    # LoGo-Fuse args.
    parser.add_argument('--method', default='logofuse', choices=['logofuse', 'gsp', 'gsp_ot'])
    parser.add_argument('--cache_features', action='store_true', help='cache ULIP point features to .npz')
    parser.add_argument('--rebuild_feature_cache', action='store_true',
                        help='force rebuild feature cache files for this run (ignore existing .npz)')
    parser.add_argument('--feature_cache_dir', default='./outputs/feature_cache', type=str)
    parser.add_argument('--save_scores', default='', type=str, help='optional .npz path for final scores')
    parser.add_argument('--scanobject_train_dat', default='', type=str,
                        help='optional custom ScanObjectNN train .dat path')
    parser.add_argument('--scanobject_test_dat', default='', type=str,
                        help='optional custom ScanObjectNN test .dat path')

    # GSP-OT branch args.
    parser.add_argument('--ot_mode', default='hybrid', choices=['token_sinkhorn', 'softmin_proto', 'hybrid'],
                        help='OT branch mode: token-level Sinkhorn, softmin prototype surrogate, or hybrid')
    parser.add_argument('--sinkhorn_eps', default=0.05, type=float, help='entropic regularization in Sinkhorn OT')
    parser.add_argument('--sinkhorn_max_iter', default=80, type=int, help='iterations in Sinkhorn OT')
    parser.add_argument('--ot_class_chunk', default=8, type=int,
                        help='class chunk size when computing token-level OT scores')
    parser.add_argument('--tau', default=0.10, type=float, help='temperature for softmin-prototype OT surrogate')
    parser.add_argument('--w_cos', default=0.35, type=float, help='fusion weight for cosine class score')
    parser.add_argument('--w_ot', default=0.30, type=float, help='fusion weight for OT class score')
    parser.add_argument('--w_gsp', default=0.25, type=float, help='fusion weight for GSP propagated class score')
    parser.add_argument('--w_stu', default=0.10, type=float, help='fusion weight for student-prototype class score')
    parser.add_argument('--lambda_margin', default=0.20, type=float,
                        help='margin term weight in final OOD score')
    parser.add_argument('--eps', default=1e-6, type=float, help='numerical epsilon in final OOD score')
    parser.add_argument('--max_iter', default=5, type=int, help='maximum GSP-OT refinement iterations')
    parser.add_argument('--tol_score', default=1e-4, type=float, help='early-stop threshold on OOD score change')
    parser.add_argument('--tol_proto', default=1e-4, type=float, help='early-stop threshold on prototype drift')
    parser.add_argument('--conf_threshold', default=0.55, type=float, help='student update confidence threshold')
    parser.add_argument('--margin_threshold', default=0.05, type=float, help='student update margin threshold')
    parser.add_argument('--ood_threshold', default=0.0, type=float, help='student update OOD-score threshold')
    parser.add_argument('--proto_ema', default=0.2, type=float, help='EMA factor for student prototype updates')
    parser.add_argument('--proto_min_samples', default=1, type=int,
                        help='minimum selected samples per class for student update')
    parser.add_argument('--text_proto_cluster_k', default=4, type=int,
                        help='number of clustered text prototypes per ID class')
    parser.add_argument('--k', default=10, type=int, help='kNN size')
    parser.add_argument('--local_method', default='lp_softmax', choices=['lp_softmax'],
                        help='local branch method (final model): propagation softmax')
    parser.add_argument('--local_score_variant', default='a0', choices=['a0', 'a1', 'a2', 'a3'],
                        help='local score shaping variant: a0=top1, a1=top1*conc, a2=top1*margin, a3=top1*margin*conc')
    parser.add_argument('--local_score_soft_enable', dest='local_score_soft_enable', action='store_true',
                        help='enable soft shaping for a1/a2/a3: weak penalty + entropy trigger + high-confidence protection')
    parser.add_argument('--no_local_score_soft_enable', dest='local_score_soft_enable', action='store_false',
                        help='disable soft shaping and use direct multiplicative a1/a2/a3')
    parser.set_defaults(local_score_soft_enable=False)
    parser.add_argument('--local_score_soft_lambda', default=0.75, type=float,
                        help='soft baseline factor lambda in s=top1*(lambda+(1-lambda)*factor)')
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
    parser.add_argument('--B', default=10, type=int, help='number of noisy graph views')
    parser.add_argument('--T_lp', default=5, type=int, help='label propagation steps')
    parser.add_argument('--alpha', default=0.5, type=float, help='LP injection coefficient')
    parser.add_argument('--stable_pi', default=0.6, type=float, help='stable-edge frequency threshold')
    parser.add_argument('--temp', default=0.07, type=float, help='stable graph edge temperature')
    parser.add_argument('--seed_temp', default=0.07, type=float, help='seed affinity temperature')
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

    parser.add_argument('--K_over', default=150, type=int, help='over-clustering K')
    parser.add_argument('--kmeans_niter', default=100, type=int, help='FAISS kmeans iterations')

    parser.add_argument('--T_p', default=5, type=int, help='dynamic prototype update steps')
    parser.add_argument('--eta', default=0.2, type=float, help='prototype update momentum')
    parser.add_argument('--global_score_mode', default='maxcos', choices=['maxcos', 'ot', 'oodd'],
                        help='global score type: max cosine, OT-based, or OOD dictionary score')
    parser.add_argument('--ot_alpha', default=0.5, type=float,
                        help='OT score blend alpha: S_ot=alpha*S_sem+(1-alpha)*S_dist')
    parser.add_argument('--ot_reg', default=0.05, type=float,
                        help='Sinkhorn entropic regularization epsilon for OT scoring')
    parser.add_argument('--ot_iters', default=80, type=int,
                        help='Sinkhorn iterations for OT scoring')
    parser.add_argument('--oodd_id_topk_per_class', default=2, type=int,
                        help='top-k per class to build ID dictionary (zero-shot)')
    parser.add_argument('--oodd_inlier_k', default=5, type=int,
                        help='k-th nearest ID-key similarity as latent inlier score')
    parser.add_argument('--oodd_ood_pool_frac', default=0.15, type=float,
                        help='candidate fraction for OOD dictionary queue')
    parser.add_argument('--oodd_ood_dict_size', default=64, type=int,
                        help='max size of OOD dictionary')
    parser.add_argument('--oodd_beta', default=0.5, type=float,
                        help='repulsion strength for OOD dictionary in global score')
    parser.add_argument('--oodd_use_local_intersection', dest='oodd_use_local_intersection', action='store_true',
                        help='build OOD dictionary from low-Sin and low-local intersection')
    parser.add_argument('--no_oodd_use_local_intersection', dest='oodd_use_local_intersection', action='store_false',
                        help='build OOD dictionary from low-Sin only')
    parser.set_defaults(oodd_use_local_intersection=True)
    parser.add_argument('--oodd_use_fewshot_id_dict', dest='oodd_use_fewshot_id_dict', action='store_true',
                        help='in few-shot mode, use few-shot prototypes as ID dictionary keys')
    parser.add_argument('--no_oodd_use_fewshot_id_dict', dest='oodd_use_fewshot_id_dict', action='store_false',
                        help='always build ID dictionary from test top-k even in few-shot mode')
    parser.set_defaults(oodd_use_fewshot_id_dict=True)
    parser.add_argument('--glo_update_topk_per_proto', default=0, type=int,
                        help='if >0, each update keeps only top-k highest-rank seed points per prototype component')
    parser.add_argument('--glo_ot_proto_weight', dest='glo_ot_proto_weight', action='store_true',
                        help='enable OT-based per-point/per-component weights in global prototype update')
    parser.add_argument('--no_glo_ot_proto_weight', dest='glo_ot_proto_weight', action='store_false',
                        help='disable OT-based per-point/per-component weights in global prototype update')
    parser.set_defaults(glo_ot_proto_weight=False)
    parser.add_argument('--glo_ot_proto_alpha', default=1.0, type=float,
                        help='blend factor for OT prototype weights vs stability weights')
    parser.add_argument('--glo_ot_proto_reg', default=0.05, type=float,
                        help='entropic regularization for OT prototype component weights')
    parser.add_argument('--glo_ot_proto_iters', default=80, type=int,
                        help='number of Sinkhorn iterations for OT prototype component weights')
    parser.add_argument('--q_frac', default=0.3, type=float, help='NR top-q fraction')
    parser.add_argument('--tau_GI', default=0.0, type=float, help='DGIS margin threshold')
    parser.add_argument('--lambda_cons', default=15.0, type=float, help='consistency weighting scale')
    parser.add_argument('--rbo_p', default=0.9, type=float, help='RBO persistence parameter')
    parser.add_argument('--glo_keep_use_dpam', dest='glo_keep_use_dpam', action='store_true',
                        help='enable DPAM gate in global prototype update keep-mask')
    parser.add_argument('--no_glo_keep_use_dpam', dest='glo_keep_use_dpam', action='store_false',
                        help='disable DPAM gate in global prototype update keep-mask')
    parser.set_defaults(glo_keep_use_dpam=True)
    parser.add_argument('--glo_keep_use_nr', dest='glo_keep_use_nr', action='store_true',
                        help='enable NR gate in global prototype update keep-mask')
    parser.add_argument('--no_glo_keep_use_nr', dest='glo_keep_use_nr', action='store_false',
                        help='disable NR gate in global prototype update keep-mask')
    parser.set_defaults(glo_keep_use_nr=True)
    parser.add_argument('--glo_keep_use_dgis', dest='glo_keep_use_dgis', action='store_true',
                        help='enable DGIS gate in global prototype update keep-mask')
    parser.add_argument('--no_glo_keep_use_dgis', dest='glo_keep_use_dgis', action='store_false',
                        help='disable DGIS gate in global prototype update keep-mask')
    parser.set_defaults(glo_keep_use_dgis=True)
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
    parser.add_argument('--fewshot_train_label_eq', default=-1, type=int,
                        help='if >=0, only keep train/support samples with this raw label id in few-shot/full-shot')
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
                        help='target alpha in conformal calibration (0.05 corresponds to TPR~95% thresholding)')
    parser.add_argument('--conformal_prior_lambda', default=0.02, type=float,
                        help='prior shrinkage towards pseudo w in conformal w-search')
    parser.add_argument('--fewshot_proto_cluster_k', default=1, type=int,
                        help='cluster count for shot prototype debiasing (1 disables clustering)')
    parser.add_argument('--fewshot_proto_cluster_mode', default='fixed', choices=['fixed', 'auto_dp', 'dbscan_balance'],
                        help='few-shot prototype clustering mode: fixed, DP-means auto-k, or DBSCAN density-balancing')
    parser.add_argument('--fewshot_proto_auto_max_k', default=4, type=int,
                        help='maximum cluster count for auto_dp few-shot clustering')
    parser.add_argument('--fewshot_proto_dp_lambda', default=0.18, type=float,
                        help='DP-means distance threshold (1-cos) for spawning new clusters')
    parser.add_argument('--fewshot_proto_dp_iters', default=20, type=int,
                        help='maximum iterations for DP-means few-shot clustering')
    parser.add_argument('--fewshot_proto_min_cluster_size', default=1, type=int,
                        help='minimum cluster size after auto_dp (small clusters are merged)')
    parser.add_argument('--fewshot_proto_center_weight', default='equal', choices=['equal', 'count'],
                        help='how to merge cluster centers into one class prototype')
    parser.add_argument('--fewshot_proto_dbscan_trigger_n', default=40, type=int,
                        help='minimum class sample count to trigger DBSCAN density-balancing')
    parser.add_argument('--fewshot_proto_dbscan_eps_k', default=8, type=int,
                        help='k for kNN-distance based DBSCAN eps estimation')
    parser.add_argument('--fewshot_proto_dbscan_eps_q', default=0.60, type=float,
                        help='quantile of kNN distance used as DBSCAN eps')
    parser.add_argument('--fewshot_proto_dbscan_min_samples_ratio', default=0.03, type=float,
                        help='DBSCAN min_samples ratio relative to class sample size')
    parser.add_argument('--fewshot_proto_dbscan_min_samples_floor', default=4, type=int,
                        help='DBSCAN min_samples lower bound')
    parser.add_argument('--fewshot_proto_dbscan_gamma', default=0.70, type=float,
                        help='density debias strength, sample weight ~ 1/(cluster_size^gamma)')
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
    parser.add_argument('--neg_hard_enable', dest='neg_hard_enable', action='store_true',
                        help='merge a hard-negative branch (high global score + low global margin) into neg pool')
    parser.add_argument('--no_neg_hard_enable', dest='neg_hard_enable', action='store_false',
                        help='disable hard-negative branch for neg pool')
    parser.set_defaults(neg_hard_enable=False)
    parser.add_argument('--neg_hard_frac', default=0.08, type=float,
                        help='fraction of samples used by hard-negative branch')
    parser.add_argument('--neg_hard_alpha', default=0.5, type=float,
                        help='merge weight for hard score vs base suspicion score')
    parser.add_argument('--neg_hard_use_disagree', dest='neg_hard_use_disagree', action='store_true',
                        help='upweight hard-negative branch by semantic/cluster disagreement')
    parser.add_argument('--no_neg_hard_use_disagree', dest='neg_hard_use_disagree', action='store_false',
                        help='disable disagreement upweight in hard-negative branch')
    parser.set_defaults(neg_hard_use_disagree=True)
    parser.add_argument('--neg_pool_cap_mult', default=2.0, type=float,
                        help='cap merged neg-pool size by multiplier of base k_pool')
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
    parser.add_argument('--neg_min_pool_fit', default=2, type=int,
                        help='minimum neg pool size to fit negative centers')
    parser.add_argument('--glo_num_proto_per_class', default=1, type=int,
                        help='number of global prototypes per class (1 disables mixture)')
    parser.add_argument('--oracle_true_id_proto', dest='oracle_true_id_proto', action='store_true',
                        help='diagnostic only: construct class prototypes from true ID labels on eval set')
    parser.add_argument('--no_oracle_true_id_proto', dest='oracle_true_id_proto', action='store_false',
                        help='disable oracle true-ID prototype construction')
    parser.set_defaults(oracle_true_id_proto=False)
    parser.add_argument('--oracle_true_ood_neg_bank', dest='oracle_true_ood_neg_bank', action='store_true',
                        help='diagnostic only: construct negative bank from true OOD samples on eval set')
    parser.add_argument('--no_oracle_true_ood_neg_bank', dest='oracle_true_ood_neg_bank', action='store_false',
                        help='disable oracle true-OOD negative-bank construction')
    parser.set_defaults(oracle_true_ood_neg_bank=False)
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
    parser.add_argument('--use_geo_signal', dest='use_geo_signal', action='store_true',
                        help='enable raw-geometry descriptor signal and fuse as third branch')
    parser.add_argument('--no_use_geo_signal', dest='use_geo_signal', action='store_false',
                        help='disable geometry descriptor branch')
    parser.set_defaults(use_geo_signal=False)
    parser.add_argument('--geo_subsample_points', default=192, type=int,
                        help='number of points used per cloud for geometry descriptor extraction')
    parser.add_argument('--geo_knn', default=16, type=int,
                        help='kNN size in xyz space for local geometry statistics')
    parser.add_argument('--geo_iso_k', default=8, type=int,
                        help='kNN size in descriptor space for geometry isolation scoring')
    parser.add_argument('--geo_fusion_weight', default=0.20, type=float,
                        help='weight of geometry ID score in final fusion')
    parser.add_argument('--geo_support_source', default='fewshot', choices=['fewshot', 'train_id'],
                        help='reference pool used to compute geometry isolation in few-shot')
    parser.add_argument('--geo_dist_chunk', default=2048, type=int,
                        help='chunk size when computing descriptor-space kNN distances')
    parser.add_argument('--geo_adaptive_weight', dest='geo_adaptive_weight', action='store_true',
                        help='use confidence-adaptive per-sample geometry fusion weight')
    parser.add_argument('--no_geo_adaptive_weight', dest='geo_adaptive_weight', action='store_false',
                        help='disable adaptive geometry fusion weight and use fixed scalar')
    parser.set_defaults(geo_adaptive_weight=True)
    parser.add_argument('--geo_adaptive_power', default=1.0, type=float,
                        help='power on (1-conf) when computing adaptive geo weight')
    parser.add_argument('--geo_adaptive_floor', default=0.0, type=float,
                        help='minimum fraction of geo_fusion_weight for adaptive geo weight')

    return parser


def _l2_normalize_torch(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _l2_normalize_np(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def _dp_means_centers(x, lambda_dist=0.18, max_k=4, n_iter=20, seed=0):
    """
    DP-means style adaptive clustering on normalized features.
    Distance is cosine distance: d(x, c) = 1 - cos(x, c).
    """
    x = _l2_normalize_np(np.asarray(x, dtype=np.float32))
    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    max_k = int(max(1, min(int(max_k), n)))
    n_iter = int(max(1, n_iter))
    lambda_dist = float(max(0.0, lambda_dist))

    rng = np.random.default_rng(int(seed))
    first_idx = int(rng.integers(0, n))
    centers = [x[first_idx].copy()]
    prev_labels = None

    for _ in range(n_iter):
        ctr = np.stack(centers, axis=0).astype(np.float32)
        sims = x @ ctr.T
        dists = 1.0 - sims
        labels = np.argmin(dists, axis=1).astype(np.int64)
        min_dist = dists[np.arange(n), labels]

        if len(centers) < max_k:
            add_idx = np.where(min_dist > lambda_dist)[0]
            if add_idx.size > 0:
                farthest = int(add_idx[np.argmax(min_dist[add_idx])])
                centers.append(x[farthest].copy())
                continue

        new_centers = []
        for ci in range(len(centers)):
            idx = np.where(labels == ci)[0]
            if idx.size == 0:
                new_centers.append(centers[ci])
                continue
            c = _l2_normalize_np(np.mean(x[idx], axis=0, keepdims=True))[0]
            new_centers.append(c.astype(np.float32))
        centers = new_centers

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

    centers = _l2_normalize_np(np.stack(centers, axis=0).astype(np.float32))
    labels = np.argmax(x @ centers.T, axis=1).astype(np.int64)
    return centers, labels


def _estimate_eps_from_knn_cosine(x, k=8, quantile=0.60):
    x = _l2_normalize_np(np.asarray(x, dtype=np.float32))
    n = int(x.shape[0])
    if n <= 2:
        return 0.20
    k_use = int(max(1, min(int(k), n - 1)))
    q_use = float(np.clip(quantile, 0.05, 0.95))

    dmat = 1.0 - np.clip(x @ x.T, -1.0, 1.0)
    np.fill_diagonal(dmat, np.inf)
    kth = np.partition(dmat, kth=(k_use - 1), axis=1)[:, (k_use - 1)]
    eps = float(np.quantile(kth, q_use))
    return float(np.clip(eps, 1e-4, 2.0))


def _dbscan_density_balanced_proto(
    x,
    trigger_n=40,
    eps_k=8,
    eps_q=0.60,
    min_samples_ratio=0.03,
    min_samples_floor=4,
    gamma=0.70,
):
    """
    DBSCAN-based density debiasing:
    keep all points; only reweight samples by inverse cluster size.
    """
    x = _l2_normalize_np(np.asarray(x, dtype=np.float32))
    n = int(x.shape[0])
    if n <= 1:
        proto = _l2_normalize_np(np.mean(x, axis=0, keepdims=True))[0]
        info = {"triggered": False, "eps": 0.0, "noise_frac": 0.0, "clusters": 1}
        return proto, np.zeros(n, dtype=np.int64), info

    trig_n = int(max(2, trigger_n))
    if n < trig_n:
        proto = _l2_normalize_np(np.mean(x, axis=0, keepdims=True))[0]
        info = {"triggered": False, "eps": 0.0, "noise_frac": 0.0, "clusters": 1}
        return proto, np.zeros(n, dtype=np.int64), info

    eps = _estimate_eps_from_knn_cosine(x, k=eps_k, quantile=eps_q)
    min_samples = int(round(float(np.clip(min_samples_ratio, 0.0, 1.0)) * n))
    min_samples = int(max(2, int(min_samples_floor), min_samples))
    min_samples = int(min(min_samples, n))

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(x).astype(np.int64)

    pos = [int(v) for v in np.unique(labels) if int(v) >= 0]
    if len(pos) == 0:
        proto = _l2_normalize_np(np.mean(x, axis=0, keepdims=True))[0]
        info = {"triggered": True, "eps": float(eps), "noise_frac": 1.0, "clusters": 1}
        return proto, labels, info

    gamma = float(max(0.0, gamma))
    counts = {c: int(np.sum(labels == c)) for c in pos}
    w = np.ones(n, dtype=np.float32)
    for i in range(n):
        lab = int(labels[i])
        if lab >= 0:
            w[i] = 1.0 / (float(counts[lab]) ** gamma)
        else:
            # Keep out-of-cluster samples; treat as singleton weight.
            w[i] = 1.0
    w = w / np.clip(float(np.sum(w)), 1e-12, None)
    proto = _l2_normalize_np(np.sum(x * w[:, None], axis=0, keepdims=True))[0]

    noise_frac = float(np.mean(labels < 0))
    n_clusters = int(len(pos) + (1 if noise_frac > 0.0 else 0))
    info = {"triggered": True, "eps": float(eps), "noise_frac": noise_frac, "clusters": n_clusters}
    return proto, labels, info


def _resolve_labels(args, dataset_obj):
    labels_path_candidates = []
    if args.dataset_name == 'ScanObjectNN15':
        labels_path_candidates.append(os.path.join('./data/SR', 'labels.json'))
    elif args.dataset_name == 'ShapeNetCore54':
        labels_path_candidates.append(os.path.join('./data/SN', 'labels.json'))
    elif args.dataset_name == 'ModelNet40':
        labels_path_candidates.append(os.path.join('./data/MN', 'labels.json'))
    labels_path_candidates.append(os.path.join('./data', 'labels.json'))

    all_labels = None
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

    if args.dataset_name == 'ScanObjectNN15':
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
        shapenet_split_indices = {
            'SN1': np.array([36, 29, 5, 53, 31, 49, 20, 13, 7, 27, 10, 8, 47, 6, 21, 19, 18, 37], dtype=np.int64),
            'SN2': np.array([22, 28, 17, 38, 48, 30, 32, 3, 24, 12, 46, 41, 40, 33, 50, 4, 2, 1], dtype=np.int64),
            'SN3': np.array([14, 34, 45, 23, 51, 25, 39, 26, 52, 0, 9, 15, 44, 43, 42, 16, 11, 35], dtype=np.int64),
        }
        if args.dataset_split in shapenet_split_indices:
            label_indices = shapenet_split_indices[args.dataset_split]
        else:
            raise ValueError(f'Unsupported ShapeNetCore54 split: {args.dataset_split}')
        id_class_names = [all_labels[i] for i in label_indices]
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


def _build_text_prototypes(model, tokenizer, args, class_names, device, return_clustered=False):
    with open(os.path.join('./data', 'templates.json'), 'r') as f:
        templates_all = json.load(f)
    if args.validate_dataset_prompt not in templates_all:
        raise ValueError(f'Prompt key {args.validate_dataset_prompt} not found in data/templates.json')
    templates = templates_all[args.validate_dataset_prompt]

    protos = []
    clustered = []
    k_text = int(max(1, getattr(args, 'text_proto_cluster_k', 4)))
    with torch.no_grad():
        for class_idx, cname in enumerate(class_names):
            texts = [t.format(cname) for t in templates]
            tokenized = tokenizer(texts).to(device, non_blocking=True)
            if tokenized.ndim == 1:
                tokenized = tokenized[None, ...]
            text_feats = model.encode_text(tokenized)
            text_feats = _l2_normalize_torch(text_feats)
            proto = _l2_normalize_torch(text_feats.mean(dim=0, keepdim=True))[0]
            protos.append(proto.detach().cpu().numpy())

            if return_clustered:
                x = text_feats.detach().cpu().numpy().astype(np.float32)
                x = _l2_normalize_np(x)
                if x.shape[0] <= 1 or k_text <= 1:
                    ctr = np.repeat(proto.detach().cpu().numpy()[None, :], k_text, axis=0).astype(np.float32)
                else:
                    k_use = int(max(1, min(k_text, x.shape[0])))
                    km = KMeans(
                        n_clusters=k_use,
                        random_state=int(getattr(args, 'seed', 0)) + int(class_idx),
                        n_init=10,
                    )
                    km.fit(x)
                    ctr = _l2_normalize_np(km.cluster_centers_.astype(np.float32))
                    if k_use < k_text:
                        pad = np.repeat(_l2_normalize_np(proto.detach().cpu().numpy()[None, :]), k_text - k_use, axis=0)
                        ctr = np.concatenate([ctr, pad.astype(np.float32)], axis=0)
                clustered.append(_l2_normalize_np(ctr.astype(np.float32)))

    protos = np.stack(protos, axis=0).astype(np.float32)
    protos = _l2_normalize_np(protos)
    if not return_clustered:
        return protos

    clustered = np.stack(clustered, axis=0).astype(np.float32)  # [C, K, D]
    return protos, clustered


def _cache_file_path(args, split_tag='test'):
    key_raw = (
        f"{args.model}|{args.dataset_name}|{args.dataset_split}|{split_tag}|"
        f"{args.npoints}|{args.test_ckpt_addr}|geo={int(bool(getattr(args, 'use_geo_signal', False)))}|"
        f"gsub={int(getattr(args, 'geo_subsample_points', 192))}|"
        f"gk={int(getattr(args, 'geo_knn', 16))}|"
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


def _compute_geo_descriptor_single(pc_xyz, args):
    xyz = np.asarray(pc_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        return np.zeros((8,), dtype=np.float32)
    xyz = xyz[:, :3]
    n = int(xyz.shape[0])
    if n < 8:
        return np.zeros((8,), dtype=np.float32)

    m = int(max(16, min(int(getattr(args, 'geo_subsample_points', 192)), n)))
    if m < n:
        # Deterministic downsampling to keep descriptor stable.
        idx = np.linspace(0, n - 1, m, dtype=np.int64)
        xyz = xyz[idx]
    m = int(xyz.shape[0])
    if m < 4:
        return np.zeros((8,), dtype=np.float32)

    k = int(max(3, min(int(getattr(args, 'geo_knn', 16)), m - 1)))

    diff = xyz[:, None, :] - xyz[None, :, :]
    dist2 = np.sum(diff * diff, axis=2).astype(np.float32)
    np.fill_diagonal(dist2, np.inf)
    nbr_idx = np.argpartition(dist2, kth=(k - 1), axis=1)[:, :k]
    nbr_d = np.sqrt(np.clip(np.take_along_axis(dist2, nbr_idx, axis=1), 0.0, None))
    local_d = np.mean(nbr_d, axis=1).astype(np.float32)
    density = (1.0 / np.clip(local_d, 1e-6, None)).astype(np.float32)

    curv = np.zeros(m, dtype=np.float32)
    ratio_mid = np.zeros(m, dtype=np.float32)
    ratio_small = np.zeros(m, dtype=np.float32)
    normals = np.zeros((m, 3), dtype=np.float32)

    eye3 = np.eye(3, dtype=np.float32)
    for i in range(m):
        neigh = xyz[nbr_idx[i]]
        center = neigh - np.mean(neigh, axis=0, keepdims=True)
        cov = (center.T @ center) / max(1, k)
        # numerical stabilization
        eigvals, eigvecs = np.linalg.eigh(cov + 1e-6 * eye3)
        eigvals = np.clip(eigvals.astype(np.float32), 1e-9, None)
        # eigh returns ascending: [small, mid, large]
        lam_s, lam_m, lam_l = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
        curv[i] = lam_s / max(lam_s + lam_m + lam_l, 1e-9)
        ratio_mid[i] = lam_m / max(lam_l, 1e-9)
        ratio_small[i] = lam_s / max(lam_l, 1e-9)
        normals[i] = eigvecs[:, 0].astype(np.float32)

    cz = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
    hist, _ = np.histogram(cz, bins=8, range=(0.0, 1.0))
    normal_entropy = _entropy_from_hist(hist)

    q10, q50, q90 = np.quantile(density, [0.10, 0.50, 0.90]).astype(np.float32)
    density_ratio = float(q90 / max(q10, 1e-6))
    desc = np.array(
        [
            float(np.mean(curv)),
            float(np.std(curv)),
            float(normal_entropy),
            float(density_ratio),
            float(np.mean(ratio_mid)),
            float(np.mean(ratio_small)),
            float(np.mean(local_d)),
            float(np.std(local_d)),
        ],
        dtype=np.float32,
    )
    return desc


def _compute_geo_descriptors_batch(pc_batch, args):
    pcs = np.asarray(pc_batch, dtype=np.float32)
    if pcs.ndim != 3:
        return np.zeros((0, 8), dtype=np.float32)
    out = [_compute_geo_descriptor_single(pcs[i], args) for i in range(pcs.shape[0])]
    return np.stack(out, axis=0).astype(np.float32)


def _knn_mean_distance(query, ref, k=8, chunk=2048, exclude_self=False):
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    if q.size == 0 or r.size == 0:
        return np.zeros((q.shape[0],), dtype=np.float32)

    nq = int(q.shape[0])
    nr = int(r.shape[0])
    k_use = int(max(1, min(int(k), nr - (1 if exclude_self and nq == nr else 0))))
    k_use = max(1, k_use)
    chunk = int(max(64, chunk))
    d_out = np.zeros((nq,), dtype=np.float32)

    for st in range(0, nq, chunk):
        ed = min(st + chunk, nq)
        qb = q[st:ed]
        dist2 = (
            np.sum(qb * qb, axis=1, keepdims=True)
            + np.sum(r * r, axis=1)[None, :]
            - 2.0 * (qb @ r.T)
        ).astype(np.float32)
        dist2 = np.clip(dist2, 0.0, None)
        if exclude_self and nq == nr:
            rr = np.arange(st, ed, dtype=np.int64)
            dist2[np.arange(ed - st), rr] = np.inf
        part = np.partition(dist2, kth=(k_use - 1), axis=1)[:, :k_use]
        d_out[st:ed] = np.mean(np.sqrt(np.clip(part, 0.0, None)), axis=1).astype(np.float32)
    return d_out


def _compute_geo_id_scores(query_desc, ref_desc, args, exclude_self=False):
    q = np.asarray(query_desc, dtype=np.float32)
    r = np.asarray(ref_desc, dtype=np.float32)
    if q.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if r.size == 0:
        return np.zeros((q.shape[0],), dtype=np.float32)

    med = np.median(r, axis=0, keepdims=True).astype(np.float32)
    mad = np.median(np.abs(r - med), axis=0, keepdims=True).astype(np.float32)
    scale = np.clip(mad, 1e-6, None)
    qn = (q - med) / scale
    rn = (r - med) / scale

    d = _knn_mean_distance(
        qn,
        rn,
        k=int(getattr(args, 'geo_iso_k', 8)),
        chunk=int(getattr(args, 'geo_dist_chunk', 2048)),
        exclude_self=bool(exclude_self),
    )
    return (1.0 - _minmax01(d)).astype(np.float32)


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
            geo_desc = data['geo_desc'] if ('geo_desc' in data.files and bool(getattr(args, 'use_geo_signal', False))) else None
            if bool(getattr(args, 'use_geo_signal', False)) and geo_desc is None:
                # Re-extract if geometry branch is enabled but missing in cache.
                pass
            else:
                return data['features'], data['targets'], geo_desc, cache_path
    else:
        cache_path = ''

    feats = []
    tgts = []
    geo_all = [] if bool(getattr(args, 'use_geo_signal', False)) else None

    model.eval()
    with torch.no_grad():
        for pc, target in data_loader:
            pc = pc.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if geo_all is not None:
                geo_b = _compute_geo_descriptors_batch(pc.detach().cpu().numpy(), args)
                geo_all.append(geo_b)

            h = model.encode_pc(pc)
            h = _l2_normalize_torch(h)

            feats.append(h.detach().cpu().numpy())
            tgts.append(target.detach().cpu().numpy())

    features = np.concatenate(feats, axis=0).astype(np.float32)
    targets = np.concatenate(tgts, axis=0).astype(np.int64)
    geo_desc = np.concatenate(geo_all, axis=0).astype(np.float32) if geo_all is not None else None

    if args.cache_features:
        if geo_desc is None:
            np.savez_compressed(cache_path, features=features, targets=targets)
        else:
            np.savez_compressed(cache_path, features=features, targets=targets, geo_desc=geo_desc)

    return features, targets, geo_desc, cache_path


def _build_fewshot_prototypes(
    features,
    targets,
    id_label_indices,
    shot,
    seed,
    fallback_text_prototypes,
    cluster_k=1,
    cluster_mode='fixed',
    auto_max_k=4,
    dp_lambda=0.18,
    dp_iters=20,
    min_cluster_size=1,
    center_weight='equal',
    dbscan_trigger_n=40,
    dbscan_eps_k=8,
    dbscan_eps_q=0.60,
    dbscan_min_samples_ratio=0.03,
    dbscan_min_samples_floor=4,
    dbscan_gamma=0.70,
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
            n_clusters = 0
        else:
            take = min(int(shot), total)
            pick = rng.choice(idx, size=take, replace=False)
            x_pick = h[pick]
            db_info = {"triggered": False, "eps": 0.0, "noise_frac": 0.0, "clusters": 1}
            if str(cluster_mode) == 'auto_dp' and x_pick.shape[0] >= 2:
                ctr, labels = _dp_means_centers(
                    x_pick,
                    lambda_dist=float(dp_lambda),
                    max_k=int(auto_max_k),
                    n_iter=int(dp_iters),
                    seed=int(seed) + int(raw_label),
                )
                min_sz = int(max(1, min_cluster_size))
                if min_sz > 1 and ctr.shape[0] > 1:
                    cnt = np.bincount(labels, minlength=ctr.shape[0])
                    keep = np.where(cnt >= min_sz)[0]
                    if keep.size >= 1 and keep.size < ctr.shape[0]:
                        sims_keep = x_pick @ ctr[keep].T
                        relabel = np.argmax(sims_keep, axis=1).astype(np.int64)
                        merged = []
                        for ki in range(keep.size):
                            idx_k = np.where(relabel == ki)[0]
                            if idx_k.size == 0:
                                continue
                            ck = _l2_normalize_np(np.mean(x_pick[idx_k], axis=0, keepdims=True))[0]
                            merged.append(ck.astype(np.float32))
                        if len(merged) > 0:
                            ctr = _l2_normalize_np(np.stack(merged, axis=0))
                            labels = np.argmax(x_pick @ ctr.T, axis=1).astype(np.int64)

                if ctr.shape[0] > 1:
                    if str(center_weight) == 'count':
                        cnt = np.bincount(labels, minlength=ctr.shape[0]).astype(np.float32)
                        w = cnt / np.clip(float(np.sum(cnt)), 1e-12, None)
                        proto = _l2_normalize_np(np.sum(ctr * w[:, None], axis=0, keepdims=True))[0]
                    else:
                        proto = _l2_normalize_np(np.mean(ctr, axis=0, keepdims=True))[0]
                    n_clusters = int(ctr.shape[0])
                else:
                    proto = _l2_normalize_np(np.mean(x_pick, axis=0, keepdims=True))[0]
                    n_clusters = 1
            elif str(cluster_mode) == 'dbscan_balance' and x_pick.shape[0] >= 2:
                proto, _, db_info = _dbscan_density_balanced_proto(
                    x_pick,
                    trigger_n=int(dbscan_trigger_n),
                    eps_k=int(dbscan_eps_k),
                    eps_q=float(dbscan_eps_q),
                    min_samples_ratio=float(dbscan_min_samples_ratio),
                    min_samples_floor=int(dbscan_min_samples_floor),
                    gamma=float(dbscan_gamma),
                )
                n_clusters = int(db_info.get("clusters", 1))
            else:
                k_proto = int(max(1, cluster_k))
                if k_proto > 1 and x_pick.shape[0] >= 2:
                    k_use = min(k_proto, x_pick.shape[0])
                    km = KMeans(n_clusters=k_use, random_state=int(seed), n_init=10)
                    km.fit(x_pick)
                    ctr = _l2_normalize_np(km.cluster_centers_.astype(np.float32))
                    if str(center_weight) == 'count':
                        cnt = np.bincount(km.labels_.astype(np.int64), minlength=k_use).astype(np.float32)
                        w = cnt / np.clip(float(np.sum(cnt)), 1e-12, None)
                        proto = _l2_normalize_np(np.sum(ctr * w[:, None], axis=0, keepdims=True))[0]
                    else:
                        proto = _l2_normalize_np(np.mean(ctr, axis=0, keepdims=True))[0]
                    n_clusters = int(k_use)
                else:
                    proto = _l2_normalize_np(np.mean(x_pick, axis=0, keepdims=True))[0]
                    n_clusters = 1
            used = int(take)
        protos.append(proto)
        class_stats[int(raw_label)] = {
            "total": total,
            "used": used,
            "clusters": int(n_clusters),
            "mode": str(cluster_mode),
            "db_triggered": bool(db_info.get("triggered", False)) if total > 0 else False,
            "db_eps": float(db_info.get("eps", 0.0)) if total > 0 else 0.0,
            "db_noise_frac": float(db_info.get("noise_frac", 0.0)) if total > 0 else 0.0,
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


def _sample_fewshot_indices(targets, id_label_indices, shot, seed):
    if int(shot) <= 0:
        return np.zeros((0,), dtype=np.int64)
    y = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    picked = []
    for raw_label in id_label_indices.tolist():
        idx = np.where(y == int(raw_label))[0]
        if idx.size == 0:
            continue
        take = min(int(shot), int(idx.size))
        sel = rng.choice(idx, size=take, replace=False)
        picked.extend(sel.tolist())
    if len(picked) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(set(picked)), dtype=np.int64)


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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    text_prototypes, text_proto_clusters = _build_text_prototypes(
        model,
        tokenizer,
        args,
        id_class_names,
        device,
        return_clustered=True,
    )

    test_features, test_targets, test_geo_desc, test_cache_path = _extract_or_load_features(
        model, test_loader, device, args, split_tag='test'
    )

    fewshot_prototypes = None
    fewshot_stats = {}
    support_learned_w = None
    support_raw_w = None
    pseudo_test_w = None
    support_weight_stats = {}
    support_results = None
    geo_score_test = None
    geo_score_train = None
    train_geo_desc = None
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
        train_features, train_targets, train_geo_desc, train_cache_path = _extract_or_load_features(
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
        label_eq = int(getattr(args, 'fewshot_train_label_eq', -1))
        if label_eq >= 0:
            keep_mask = (train_targets.astype(np.int64) == label_eq)
            keep_n = int(np.sum(keep_mask))
            total_n = int(train_targets.shape[0])
            if keep_n > 0:
                train_features = train_features[keep_mask]
                train_targets = train_targets[keep_mask]
                if train_geo_desc is not None:
                    train_geo_desc = train_geo_desc[keep_mask]
                print(f"fewshot_train_label_filter: label=={label_eq} kept={keep_n}/{total_n}")
            else:
                print(f"fewshot_train_label_filter: label=={label_eq} kept=0/{total_n}, skip filter")
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
            cluster_k=int(getattr(args, 'fewshot_proto_cluster_k', 1)),
            cluster_mode=str(getattr(args, 'fewshot_proto_cluster_mode', 'fixed')),
            auto_max_k=int(getattr(args, 'fewshot_proto_auto_max_k', 4)),
            dp_lambda=float(getattr(args, 'fewshot_proto_dp_lambda', 0.18)),
            dp_iters=int(getattr(args, 'fewshot_proto_dp_iters', 20)),
            min_cluster_size=int(getattr(args, 'fewshot_proto_min_cluster_size', 1)),
            center_weight=str(getattr(args, 'fewshot_proto_center_weight', 'equal')),
            dbscan_trigger_n=int(getattr(args, 'fewshot_proto_dbscan_trigger_n', 40)),
            dbscan_eps_k=int(getattr(args, 'fewshot_proto_dbscan_eps_k', 8)),
            dbscan_eps_q=float(getattr(args, 'fewshot_proto_dbscan_eps_q', 0.60)),
            dbscan_min_samples_ratio=float(getattr(args, 'fewshot_proto_dbscan_min_samples_ratio', 0.03)),
            dbscan_min_samples_floor=int(getattr(args, 'fewshot_proto_dbscan_min_samples_floor', 4)),
            dbscan_gamma=float(getattr(args, 'fewshot_proto_dbscan_gamma', 0.70)),
        )

        if bool(getattr(args, 'use_geo_signal', False)) and train_geo_desc is not None and test_geo_desc is not None:
            if str(getattr(args, 'geo_support_source', 'fewshot')) == 'fewshot':
                support_idx = _sample_fewshot_indices(
                    targets=train_targets,
                    id_label_indices=id_label_indices,
                    shot=args.shot,
                    seed=args.fewshot_seed,
                )
            else:
                support_idx = np.where(np.isin(train_targets, id_label_indices))[0].astype(np.int64)
            if support_idx.size == 0:
                support_idx = np.arange(train_geo_desc.shape[0], dtype=np.int64)
            ref_geo = train_geo_desc[support_idx]
            geo_score_train = _compute_geo_id_scores(train_geo_desc, ref_geo, args, exclude_self=False)
            geo_score_test = _compute_geo_id_scores(test_geo_desc, ref_geo, args, exclude_self=False)

        if args.fewshot_weight_source == 'support':
            support_results = run_logofuse(
                train_features,
                train_targets,
                text_prototypes,
                args,
                fewshot_prototypes=fewshot_prototypes,
                geo_score=geo_score_train,
            )
    else:
        train_cache_path = ''
        if bool(getattr(args, 'use_geo_signal', False)) and test_geo_desc is not None:
            geo_score_test = _compute_geo_id_scores(test_geo_desc, test_geo_desc, args, exclude_self=True)

    if args.shot > 0 and args.fewshot_weight_source == 'support' and support_results is not None:
        pseudo_preview = run_logofuse(
            test_features,
            test_targets,
            text_prototypes,
            args,
            fewshot_prototypes=fewshot_prototypes,
            geo_score=geo_score_test,
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

        if args.method == 'gsp_ot':
            results = run_gsp_ot(
                test_features,
                test_targets,
                text_prototypes,
                text_proto_clusters,
                args,
                fewshot_prototypes=fewshot_prototypes,
                fusion_weight_override=support_learned_w,
                geo_score=geo_score_test,
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
                geo_score=geo_score_test,
                train_anchor_features=train_id_anchor_features,
            )
    else:
        if args.method == 'gsp_ot':
            results = run_gsp_ot(
                test_features,
                test_targets,
                text_prototypes,
                text_proto_clusters,
                args,
                fewshot_prototypes=fewshot_prototypes,
                fusion_weight_override=support_learned_w,
                geo_score=geo_score_test,
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
                geo_score=geo_score_test,
                train_anchor_features=train_id_anchor_features,
            )

    print('===== LoGo-Fuse Results =====')
    print(f"dataset: {args.dataset_name} split: {args.dataset_split}")
    print(f"method: {args.method}")
    print(f"local_method: {results.get('local_method', getattr(args, 'local_method', 'lp_softmax'))}")
    print(f"local_score_variant: {results.get('local_score_variant', getattr(args, 'local_score_variant', 'a0'))}")
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
    print(f"shot: {args.shot}")
    print(f"global_update: T_p={int(getattr(args, 'T_p', 0))} topk_per_proto={int(getattr(args, 'glo_update_topk_per_proto', 0))}")
    print(
        "global_score: "
        f"mode={results.get('global_score_mode', getattr(args, 'global_score_mode', 'maxcos'))}"
    )
    if str(results.get('global_score_mode', getattr(args, 'global_score_mode', 'maxcos'))) == 'ot':
        print(
            "global_score_ot: "
            f"alpha={results.get('ot_alpha', getattr(args, 'ot_alpha', 0.5)):.6f} "
            f"reg={results.get('ot_reg', getattr(args, 'ot_reg', 0.05)):.6f} "
            f"iters={int(results.get('ot_iters', getattr(args, 'ot_iters', 80)))} "
            f"sem_mean={results.get('ot_sem_mean', 0.0):.6f} "
            f"dist_mean={results.get('ot_dist_mean', 0.0):.6f}"
        )
    if str(results.get('global_score_mode', getattr(args, 'global_score_mode', 'maxcos'))) == 'oodd':
        print(
            "global_score_oodd: "
            f"id_dict={results.get('oodd_id_dict_size', 0)} "
            f"id_topk={results.get('oodd_id_topk_per_class', 0)} "
            f"inlier_k={results.get('oodd_inlier_k', 0)} "
            f"ood_pool={results.get('oodd_ood_pool_size', 0)} "
            f"ood_dict={results.get('oodd_ood_dict_size', 0)} "
            f"beta={results.get('oodd_beta', 0.0):.6f} "
            f"use_loc_inter={results.get('oodd_use_local_intersection', False)} "
            f"neg_max_mean={results.get('oodd_neg_max_mean', 0.0):.6f}"
        )
    if bool(results.get('gsp_ot_used', False)):
        print(
            "gsp_ot: "
            f"ablation={results.get('gsp_ot_ablation', 'full')} "
            f"ot_mode={results.get('ot_mode_effective', 'softmin_proto')} "
            f"token={results.get('ot_token_available', False)} "
            f"iter={results.get('iter_used', 0)}/{results.get('max_iter', 0)} "
            f"w=(cos:{results.get('w_cos', 0.0):.3f},ot:{results.get('w_ot', 0.0):.3f},gsp:{results.get('w_gsp', 0.0):.3f},stu:{results.get('w_stu', 0.0):.3f})"
        )
        if results.get('ot_fallback_reason', ''):
            print(f"gsp_ot_fallback: {results.get('ot_fallback_reason', '')}")
        proto_hist = np.asarray(results.get('proto_drift_history', []), dtype=np.float32).reshape(-1)
        conf_hist = np.asarray(results.get('conf_change_history', []), dtype=np.float32).reshape(-1)
        if proto_hist.size > 0:
            print(f"gsp_ot_proto_drift_last: {float(proto_hist[-1]):.6f}")
        if conf_hist.size > 0:
            print(f"gsp_ot_conf_change_last: {float(conf_hist[-1]):.6f}")
    if bool(results.get('glo_ot_proto_weight_enabled', False)):
        print(
            "glo_ot_weight: "
            f"enabled={results.get('glo_ot_proto_weight_enabled', False)} "
            f"alpha={results.get('glo_ot_proto_alpha', 0.0):.6f} "
            f"reg={results.get('glo_ot_proto_reg', 0.0):.6f} "
            f"iters={results.get('glo_ot_proto_iters', 0)} "
            f"mean={results.get('glo_ot_proto_weight_mean', 0.0):.6f}"
        )
    print(f"AUROC: {results['auroc']:.6f}" if not np.isnan(results['auroc']) else 'AUROC: nan')
    print(f"FPR@TPR95: {results['fpr95']:.6f}" if not np.isnan(results['fpr95']) else 'FPR@TPR95: nan')
    print(f"w*: {results['w_star']:.6f}")
    if 'geo_weight' in results:
        print(
            "geo_branch: "
            f"enabled={results.get('geo_used', False)} "
            f"weight={results.get('geo_weight', 0.0):.6f} "
            f"mean={results.get('geo_weight_mean', results.get('geo_weight', 0.0)):.6f} "
            f"std={results.get('geo_weight_std', 0.0):.6f}"
        )
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
            f"source={results.get('neg_source_mode', 'na')} "
            f"pool={results.get('neg_pool_size', 0)} "
            f"k={results.get('neg_k_used', 0)} "
            f"ess={results.get('neg_ess', 0.0):.3f} "
            f"marg_eff={results.get('neg_margin_effective', 0.0):.4f}"
        )
        if 'oracle_true_id_proto' in results or 'oracle_true_ood_neg_bank' in results:
            print(
                "oracle_diag: "
                f"id_proto={results.get('oracle_true_id_proto', False)} "
                f"ood_neg_bank={results.get('oracle_true_ood_neg_bank', False)}"
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
        if 'neg_pool_ood_ratio' in results:
            print(
                "neg_label_check: "
                f"pool_ood={results.get('neg_pool_ood_ratio', 0.0):.6f} "
                f"fit_ood={results.get('neg_fit_ood_ratio', 0.0):.6f} "
                f"fit_n={results.get('neg_fit_size', 0)} "
                f"single_pool_ood={results.get('single_neg_pool_ood_ratio', 0.0):.6f}"
            )
        if 'neg_hard_used' in results:
            print(
                "neg_hard: "
                f"used={results.get('neg_hard_used', False)} "
                f"count={results.get('neg_hard_count', 0)} "
                f"ood_ratio={results.get('neg_hard_ood_ratio', 0.0):.6f}"
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
    if results.get('glo_mixture_used', False):
        print(
            "glo_mixture: "
            f"num_proto_per_class={results.get('glo_num_proto_per_class', 1)} "
            f"class_multi_count={results.get('glo_multi_class_count', 0)}"
        )
    if 'glo_gate_keep_rate' in results:
        print(
            "glo_gate_rates: "
            f"dpam={results.get('glo_gate_dpam_rate', 0.0):.6f} "
            f"nr={results.get('glo_gate_nr_rate', 0.0):.6f} "
            f"dgis={results.get('glo_gate_dgis_rate', 0.0):.6f} "
            f"keep={results.get('glo_gate_keep_rate', 0.0):.6f}"
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
        print(f"fewshot_proto_cluster_mode: {str(getattr(args, 'fewshot_proto_cluster_mode', 'fixed'))}")
        print(f"fewshot_proto_cluster_k: {int(getattr(args, 'fewshot_proto_cluster_k', 1))}")
        if str(getattr(args, 'fewshot_proto_cluster_mode', 'fixed')) == 'auto_dp':
            print(
                "fewshot_proto_auto: "
                f"max_k={int(getattr(args, 'fewshot_proto_auto_max_k', 4))} "
                f"dp_lambda={float(getattr(args, 'fewshot_proto_dp_lambda', 0.18)):.4f} "
                f"min_cluster_size={int(getattr(args, 'fewshot_proto_min_cluster_size', 1))} "
                f"center_weight={str(getattr(args, 'fewshot_proto_center_weight', 'equal'))}"
            )
        if str(getattr(args, 'fewshot_proto_cluster_mode', 'fixed')) == 'dbscan_balance':
            print(
                "fewshot_proto_dbscan: "
                f"trigger_n={int(getattr(args, 'fewshot_proto_dbscan_trigger_n', 40))} "
                f"eps_k={int(getattr(args, 'fewshot_proto_dbscan_eps_k', 8))} "
                f"eps_q={float(getattr(args, 'fewshot_proto_dbscan_eps_q', 0.60)):.3f} "
                f"min_samples_ratio={float(getattr(args, 'fewshot_proto_dbscan_min_samples_ratio', 0.03)):.3f} "
                f"min_samples_floor={int(getattr(args, 'fewshot_proto_dbscan_min_samples_floor', 4))} "
                f"gamma={float(getattr(args, 'fewshot_proto_dbscan_gamma', 0.70)):.3f}"
            )
        if args.fewshot_align_labels and label_mapping:
            print(f"fewshot_label_mapping(train->test): {label_mapping}")
        if support_learned_w is not None:
            print(f"fewshot_weight_source: {args.fewshot_weight_source}")
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
        save_payload = {
            'ood_score': results['ood_score'],
            'id_score': results['id_score'],
            's_loc': results.get('s_loc', None),
            's_glo': results.get('s_glo', None),
            's_geo': results.get('s_geo', None),
            'y_true_ood': results['y_true_ood'],
            'targets': test_targets,
            'features': test_features,
        }
        if 'fused_scores_cls' in results:
            save_payload['fused_scores_cls'] = results['fused_scores_cls']
        if 's_cos_cls' in results:
            save_payload['s_cos_cls'] = results['s_cos_cls']
        if 's_ot_cls' in results:
            save_payload['s_ot_cls'] = results['s_ot_cls']
        if 's_gsp_cls' in results:
            save_payload['s_gsp_cls'] = results['s_gsp_cls']
        if 's_stu_cls' in results:
            save_payload['s_stu_cls'] = results['s_stu_cls']
        if 'proto_drift_history' in results:
            save_payload['proto_drift_history'] = results['proto_drift_history']
        if 'conf_change_history' in results:
            save_payload['conf_change_history'] = results['conf_change_history']
        if 'score_change_history' in results:
            save_payload['score_change_history'] = results['score_change_history']
        np.savez_compressed(args.save_scores, **save_payload)
        print(f"saved_scores: {args.save_scores}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP + LoGo-Fuse evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
