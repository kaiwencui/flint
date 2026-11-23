#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <model_path_or_pack> [bf16|w4a16|fp8] [seq_len]}"
QUANT="${2:-w4a16}"
SEQ_LEN="${3:-4096}"

echo "=== Nsight Compute (ncu) Roofline Capture: model=$MODEL quant=$QUANT seq=$SEQ_LEN ==="
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__bytes.sum.per_second,\
  lts__t_sector_hit_rate.pct,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  launch__registers_per_thread,\
  gpu__time_duration.sum \
  --kernel-name-base demangled \
  --launch-skip 40 --launch-count 60 \
  ./flint-bench --model "$MODEL" --decode-step --seq-len "$SEQ_LEN" --quant "$QUANT"
