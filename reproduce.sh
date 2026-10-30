#!/bin/bash
# Train every model of the paper, then compute the paper's numbers, tables and figure data.
#
#   bash reproduce.sh <data root> <output folder> [model ...]
#
# With no model names, all 19 are trained (about 15 hours on one GPU, most of it for the
# Noise2Void and the four aligned-pair models). Each checkpoint is written to
# <output>/checkpoints/<name>/model.pt; with the package versions of requirements.txt and the GPU
# type stated in the README it is byte-identical to the released one (checkpoints/SHA256SUMS).
set -euo pipefail
DATA=$1
OUT=$2
shift 2

declare -A ARGS=(
  [n2n_mixsigma_s1]="--model n2n --seed 1"
  [n2n_mixsigma_s2]="--model n2n --seed 2"
  [n2n_mixsigma_s3]="--model n2n --seed 3"
  [n2n_mixsigma_s4]="--model n2n --seed 4"
  [n2n_mixsigma_s5]="--model n2n --seed 5"
  [n2n_noaug]="--model n2n --noise-aug-max 0 --seed 0"
  [n2n_gap2]="--model n2n --pair-gap 2 --align 1 --seed 0"
  [n2n_gap3]="--model n2n --pair-gap 3 --align 1 --seed 0"
  [n2n_gap6_s1]="--model n2n --pair-gap 6 --align 1 --seed 0"
  [n2n_gap6_s2]="--model n2n --pair-gap 6 --align 1 --seed 2"
  [n2n_gap6_s3]="--model n2n --pair-gap 6 --align 1 --seed 3"
  [n2n_depth1]="--model n2n --levels 1 --seed 0"
  [n2n_depth3]="--model n2n --levels 3 --seed 0"
  [n2n_w16]="--model n2n --base 16 --seed 0"
  [n2n_w64]="--model n2n --base 64 --seed 0"
  [n2v]="--model n2v"
  [supervised]="--model supervised"
  [supervised_mixsigma]="--model supervised --noise-aug-max 21"
  [park2020]="--model park2020"
)
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(n2n_mixsigma_s1 n2n_mixsigma_s2 n2n_mixsigma_s3 n2n_mixsigma_s4 n2n_mixsigma_s5
  n2n_noaug n2n_gap2 n2n_gap3 n2n_gap6_s1 n2n_gap6_s2 n2n_gap6_s3 n2n_depth1 n2n_depth3 n2n_w16 n2n_w64
  n2v supervised supervised_mixsigma park2020)

cd "$(dirname "$0")"
for name in "${NAMES[@]}"; do
  echo "== $name: python train.py ${ARGS[$name]}"
  python train.py ${ARGS[$name]} --data-root "$DATA" --out-dir "$OUT/checkpoints/$name"
done
if [ $# -eq 0 ]; then
  python analysis.py --data-root "$DATA" --ckpt-dir "$OUT/checkpoints" --out-dir "$OUT"
fi
