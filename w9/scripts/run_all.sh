#!/usr/bin/env bash
# Run serially: GPU measurements must not overlap one another.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
run_dir="${1:-results/run_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing results: $run_dir" >&2
    exit 1
fi
mkdir -p "$run_dir"
"$PYTHON" scripts/environment.py > "$run_dir/environment.json"
"$PYTHON" -m pip freeze > "$run_dir/environment.freeze.txt"
make -B all > "$run_dir/build.log" 2>&1
make test PYTHON="$PYTHON" > "$run_dir/tests.log" 2>&1
# Invalid input checks must fail before launching GPU work.
for arg in 0 -1 abc 123abc 67108865; do
    if ./build/vector_bench "$arg" >> "$run_dir/invalid_inputs.log" 2>&1; then
        echo "ERROR: accepted invalid input $arg" >&2
        exit 1
    fi
    echo "Rejected $arg PASS" >> "$run_dir/invalid_inputs.log"
done
if command -v compute-sanitizer >/dev/null 2>&1; then
    if compute-sanitizer --tool memcheck --error-exitcode 99 ./build/vector_add_check > "$run_dir/memcheck_integer.log" 2>&1; then
        compute-sanitizer --tool memcheck --error-exitcode 99 ./build/vector_bench 1025 1 2 > "$run_dir/memcheck_float.log" 2>&1
    elif grep -q 'Failed to initialize WDDM debugger interface' "$run_dir/memcheck_integer.log"; then
        echo "NOT VERIFIED: WDDM debugger interface unavailable; float memcheck not attempted" > "$run_dir/memcheck_unavailable.txt"
    else
        echo "Memcheck failed; inspect log" >&2
        exit 1
    fi
else
    echo "NOT RUN: compute-sanitizer unavailable" > "$run_dir/memcheck_unavailable.txt"
fi
"$PYTHON" benchmarks/attention_bench.py --output "$run_dir/attention.csv" > "$run_dir/attention.log" 2>&1
for n in 1024 262144 2097152 16777216; do
    ./build/vector_bench "$n" 5 100 > "$run_dir/vector_$n.csv" 2>> "$run_dir/vector_correctness.log"
done
"$PYTHON" scripts/profile_attention.py --output-dir "$run_dir/traces" > "$run_dir/profile.log" 2>&1
"$PYTHON" scripts/environment.py > "$run_dir/environment_after.json"
"$PYTHON" scripts/summarize.py "$run_dir" > "$run_dir/summary_console.txt"
sha256sum src/*.* benchmarks/*.py scripts/*.py scripts/*.sh tests/*.py Makefile > "$run_dir/source_sha256.txt"
printf 'Complete: %s\n' "$run_dir"
