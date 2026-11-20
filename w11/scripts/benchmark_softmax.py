"""W11: correctness-gated, reproducible eager Softmax comparison on one GPU.

Method reference: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
The student's W10 kernel is preserved locally in src/w11_softmax/softmax.py.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.utils.benchmark as benchmark
import triton
from w11_softmax.softmax import triton_softmax

MINIMUM = [(1, 128), (4096, 128), (1024, 1000), (1024, 4096)]
FULL = MINIMUM + [(32, 128), (32, 4096), (1024, 1024),
                  (1024, 1025), (4096, 4096)]


def naive_softmax(x):
    # FP16 extension: explicitly use FP32 intermediates, matching W10's contract.
    # FP32 .float() is a no-op; FP16 promotion and final cast are timed.
    z = x.float()
    shifted = z - z.amax(dim=1, keepdim=True)
    numerator = shifted.exp()
    return (numerator / numerator.sum(dim=1, keepdim=True)).to(x.dtype)


def torch_softmax(x):
    return torch.softmax(x, dim=1)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def capture(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        return f"exit_code={p.returncode}\n{p.stdout}{p.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"capture unavailable: {exc}\n"


def check(x, fn):
    actual = fn(x)
    expected = torch.softmax(x, dim=1)
    fp16 = x.dtype == torch.float16
    rtol, atol, sum_atol = (1e-3, 1e-5, 1e-3) if fp16 else (1e-5, 1e-6, 1e-6)
    
    if (actual.shape, actual.dtype, actual.device) != (x.shape, x.dtype, x.device):
        raise AssertionError("output shape/dtype/device mismatch")
    
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    
    if not torch.isfinite(actual).all().item() or (actual < 0).any().item():
        raise AssertionError("nonfinite or negative output")
    
    sums = actual.float().sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=sum_atol)
    torch.cuda.synchronize()
    return dict(max_abs_error=(actual.float() - expected.float()).abs().max().item(),
                max_row_sum_error=(sums - 1).abs().max().item(),
                rtol=rtol, atol=atol, row_sum_atol=sum_atol)


def summarize(records):
    groups = {}
    for row in records:
        key = (row["M"], row["N"], row["dtype"], row["provider"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (m, n, dtype, provider), group in groups.items():
        medians = [r["median_us"] for r in group]
        center = statistics.median(medians)
        paired = []
        for r in group:
            ref = next(v for v in records if (v["M"], v["N"], v["dtype"], v["round"], v["provider"])
                       == (m, n, dtype, r["round"], "torch"))
            paired.append(ref["median_us"] / r["median_us"])
        summaries.append(dict(M=m, N=n, dtype=dtype, provider=provider,
                              rounds=len(group), median_us=center,
                              min_round_us=min(medians), max_round_us=max(medians),
                              repeat_range_pct=100 * (max(medians) - min(medians)) / center,
                              max_within_iqr_pct=max(100 * r["iqr_us"] / r["median_us"] for r in group),
                              speedup_vs_torch=statistics.median(paired),
                              effective_GBps=2 * m * n * (2 if dtype == "fp16" else 4) / (center * 1e3)))
    return summaries


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["minimum", "full"], default="full")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--note", default="unspecified; record power mode and other GPU workloads")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args()
    if args.rounds < 3 or not math.isfinite(args.min_run_time) or args.min_run_time <= 0 or args.warmup < 1:
        parser.error("Use rounds >= 3, min-run-time > 0, warmup >= 1")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; use the working WSL W10 Python environment")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai"))
    out = args.output_dir / stamp.strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {out}", flush=True)
    props = torch.cuda.get_device_properties(0)
    env = dict(timestamp=stamp.isoformat(), command=sys.argv, python=sys.version,
               executable=sys.executable, platform=platform.platform(), torch=torch.__version__,
               triton=triton.__version__, cuda_build=torch.version.cuda, gpu=props.name,
               gpu_memory_bytes=props.total_memory, capability=list(torch.cuda.get_device_capability()),
               args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               method="torch.utils.benchmark.Timer.blocked_autorange; synchronized host wall time",
               allocation="all providers allocate outputs inside timed function",
               cache="same resident input reused; no explicit cache flush",
               grad="torch.inference_mode", num_threads=1, profiler=False,
               schema_version=2, source_sha256={})
    sources = {"benchmark_softmax.py": Path(__file__),
               "softmax.py": ROOT / "src/w11_softmax/softmax.py",
               "__init__.py": ROOT / "src/w11_softmax/__init__.py"}
    (out / "source").mkdir()
    for name, path in sources.items():
        content = path.read_bytes()
        env["source_sha256"][name] = hashlib.sha256(content).hexdigest()
        (out / "source" / name).write_bytes(content)
    save_json(out / "environment.json", env)
    (out / "nvidia-smi-before.txt").write_text(capture(["nvidia-smi"]), encoding="utf-8")
    (out / "pip-freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    providers = {"naive": naive_softmax, "torch": torch_softmax, "triton": triton_softmax}
    shapes = MINIMUM if args.suite == "minimum" else FULL
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    checks, records = [], []
    x = None
    try:
        with torch.inference_mode():
            for m, n in shapes:
                x = torch.randn((m, n), device="cuda", dtype=dtype)
                # Check all implementations before timing this shape, including the naive baseline.
                for name, fn in providers.items():
                    metrics = check(x, fn)
                    checks.append(dict(M=m, N=n, dtype=args.dtype, provider=name, status="PASS", **metrics))
                    save_json(out / "correctness.json", checks)
                    for _ in range(args.warmup):
                        fn(x)
                    torch.cuda.synchronize()
                order = list(providers)
                random.Random(args.seed + m + n).shuffle(order)
                for round_id in range(args.rounds):
                    shifted = order[round_id % 3:] + order[:round_id % 3]
                    for position, name in enumerate(shifted):
                        measurement = benchmark.Timer(stmt="fn(x)",
                            globals={"fn": providers[name], "x": x}, num_threads=1,
                        ).blocked_autorange(min_run_time=args.min_run_time)
                        row = dict(M=m, N=n, dtype=args.dtype, provider=name, round=round_id,
                                   order=position, median_us=measurement.median * 1e6,
                                   iqr_us=measurement.iqr * 1e6,
                                   number_per_run=measurement.number_per_run,
                                   sample_blocks=len(measurement.raw_times),
                                   raw_block_seconds=measurement.raw_times)
                        records.append(row)
                        save_json(out / "raw.json", records)
                    print(f"{m:5d}x{n:<4d} {args.dtype} round {round_id + 1}/{args.rounds} done", flush=True)
        summaries = summarize(records)
        save_csv(out / "summary.csv", summaries)
        save_json(out / "summary.json", summaries)
        lines = ["# 自动汇总：同步主机调用耗时", "", "这是实测数据摘要；归因和学习验收仍需完成。",
                 "同一输入重复使用；包括 Python 封装、输出分配和 GPU 完成时间；不等于纯 kernel 时间。", "",
                 "| M×N | dtype | 实现 | 中位数 μs | 各轮最小～最大 μs | 相对 torch 倍率 | 轮内最大 IQR/中位数 |",
                 "|---|---|---|---:|---:|---:|---:|"]
        for r in summaries:
            lines.append(f"| {r['M']}×{r['N']} | {r['dtype']} | {r['provider']} | {r['median_us']:.3f} | "
                         f"{r['min_round_us']:.3f}～{r['max_round_us']:.3f} | {r['speedup_vs_torch']:.3f} | "
                         f"{r['max_within_iqr_pct']:.1f}% |")
        lines += ["", "倍率为同轮 torch 耗时 / 当前实现耗时，再对各轮取中位数；>1 表示当前实现更快。",
                  "波动大的条件应单独复核；这里的分位数来自计时块均值，不是线上请求尾延迟。"]
        (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        save_json(out / "status.json", dict(status="PASS", correctness_checks=len(checks),
                                            timed_groups=len(records), shapes=len(shapes)))
    except Exception:
        save_json(out / "status.json", dict(status="FAIL", traceback=traceback.format_exc(),
                                            completed_checks=len(checks), completed_timings=len(records)))
        if x is not None:
            try:
                torch.save(x.cpu(), out / "failure_input.pt")
            except Exception:
                pass
        traceback.print_exc()
        return 1
    finally:
        (out / "nvidia-smi-after.txt").write_text(capture(["nvidia-smi"]), encoding="utf-8")
    print(f"PASS: {len(checks)} correctness checks; {len(records)} timing groups\n{out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
