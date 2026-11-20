#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
evidence_dir="$(cd "$(dirname "$0")" && pwd)"
out_dir="${1:?Pass a new output directory}"
mode="${2:-graph}"
mkdir "$out_dir"
: "${PYTHON:=python}"
: "${NCU:=/usr/local/cuda/bin/ncu}"
for experiment in '1024 1' '1024 8' '4096 1'; do
    read -r rows pairs <<< "$experiment"
    "$NCU" --csv --nvtx --nvtx-include 'measure/' --cache-control none --clock-control none --graph-profiling "$mode" \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active \
        "$PYTHON" "$evidence_dir/ncu_probe.py" "$rows" "$pairs" "$mode" \
        > "$out_dir/ncu_${rows}x4096_pairs${pairs}.csv" 2>&1
done
