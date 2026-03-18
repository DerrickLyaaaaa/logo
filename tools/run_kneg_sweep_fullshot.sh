#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

K_START="${1:-1}"
K_END="${2:-15}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_DEFAULT="${TEST_CKPT_ADDR:-$ROOT_DIR/ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt}"
TS=$(date +%Y%m%d_%H%M%S)
OD="${OD:-logs_kneg_fullshot_sr123_k${K_START}_k${K_END}_${TS}}"
mkdir -p "$OD"
printf "neg_k\tsplit\tAUROC\tFPR95\tw_star\n" > "$OD/results.tsv"

for neg_k in $(seq "$K_START" "$K_END"); do
  for split in SR1 SR2 SR3; do
    log="$OD/k${neg_k}_${split}.log"
    echo "[RUN] neg_k=$neg_k split=$split"
    "$PYTHON_BIN" main_logofuse.py \
      --model ULIP_PointBERT \
      --method logofuse \
      --evaluate_3d \
      --dataset_name ScanObjectNN15 \
      --dataset_split "$split" \
      --npoints 2048 \
      --validate_dataset_prompt shapenet_64 \
      --test_ckpt_addr "$CKPT_DEFAULT" \
      --fewshot_seed 0 \
      --batch-size 8 \
      --workers 0 \
      --cache_features \
      --feature_cache_dir ./outputs/feature_cache_logofuse \
      --fewshot_weight_source support \
      --fewshot_support_importance test_affinity \
      --fewshot_weight_solver map \
      --fewshot_weight_cap 0.3 \
      --local_method lp_softmax \
      --glo_num_proto_per_class 1 \
      --no_glo_triggered_mixture \
      --glo_use_neg_bank \
      --neg_k "$neg_k" \
      --neg_min_pool 32 \
      --neg_rank_use_joint \
      --neg_use_soft_weight_ess \
      --neg_adaptive_margin \
      --tta_views 8 \
      --no_tta_filter_neg_pool \
      --fewshot_proto_cluster_mode fixed \
      --fewshot_proto_cluster_k 1 \
      --neg_rank_shot_aware \
      --neg_rank_ref_shot 5 \
      --neg_rank_r0 0.2 \
      --neg_tta_soft_only \
      --q_frac 0.3 \
      --T_p 5 \
      --glo_update_topk_per_proto 2 \
      --k 7 \
      --no_glo_use_iterative_revisit \
      --no_neg_stab_enable \
      --fusion_weight_solver mse \
      --global_score_mode maxcos \
      --shot 999999 > "$log" 2>&1

    "$PYTHON_BIN" - "$log" "$neg_k" "$split" >> "$OD/results.tsv" <<'PY'
import re,sys
log,neg_k,split=sys.argv[1:4]
s=open(log,encoding='utf-8',errors='ignore').read()
def g(p,d='nan'):
    m=re.search(p,s,re.M)
    return m.group(1) if m else d
print('\t'.join([neg_k,split,g(r'^AUROC:\s*([0-9.]+)$'),g(r'^FPR@TPR95:\s*([0-9.]+)$'),g(r'^w\*:\s*([0-9.]+)$')]))
PY
  done
done

"$PYTHON_BIN" - "$OD/results.tsv" > "$OD/summary_by_k.tsv" <<'PY'
import csv,sys
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1],encoding='utf-8'), delimiter='\t'))
acc=defaultdict(lambda:[0.0,0.0,0])
for r in rows:
    k=r['neg_k']; acc[k][0]+=float(r['AUROC']); acc[k][1]+=float(r['FPR95']); acc[k][2]+=1
print('neg_k\tmacro_AUROC\tmacro_FPR95')
for k in sorted(acc,key=lambda x:int(x)):
    au,fp,n=acc[k]
    print(f'{k}\t{au/n:.6f}\t{fp/n:.6f}')
PY

echo "RESULT_DIR=$OD"
cat "$OD/summary_by_k.tsv"
