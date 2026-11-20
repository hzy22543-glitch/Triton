"""Generate repeatable tables from CSV and raw trace events (no profiler-table summation)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics as stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir
    groups = defaultdict(list)
    positions = defaultdict(list)
    rows = list(csv.DictReader((root / "attention.csv").open()))
    for row in rows:
        key = (int(row["s"]), row["variant"])
        groups[key].append(float(row["avg_us"]))
        positions[(int(row["s"]), int(row["group"]), row["variant"])].append(float(row["avg_us"]))
    out = ["# 自动生成的实验数据汇总", "", "数据源：本目录 CSV；单位 μs。每个样本是一轮连续多次前向的平均时间，不是单请求延迟。", "",
           "## Attention（无 profiler）", "",
           "| S | 版本 | 样本数 | 中位数 μs | 最小 μs | 最大 μs |",
           "|---:|---|---:|---:|---:|---:|"]
    for (s, name), values in sorted(groups.items()):
        out.append(f"| {s} | {name} | {len(values)} | {stats.median(values):.3f} | {min(values):.3f} | {max(values):.3f} |")
    out += ["", "| S | baseline/inplace | 耗时下降 | 各组 baseline/inplace |", "|---:|---:|---:|---|"]
    sizes = sorted({key[0] for key in groups})
    for s in sizes:
        baseline, inplace = [stats.median(groups[(s, name)]) for name in ("baseline", "inplace")]
        ratios = [stats.median(positions[(s, g, "baseline")]) / stats.median(positions[(s, g, "inplace")])
                  for g in sorted({key[1] for key in positions if key[0] == s})]
        out.append(f"| {s} | {baseline/inplace:.3f}× | {(1-inplace/baseline)*100:.2f}% | " + ", ".join(f"{r:.3f}×" for r in ratios) + " |")
    out += ["", "## Vector add（无 profiler，全部元素正确性在计时外检查）", "",
            "| N | 轮数 | 中位数 μs | 最小 μs | 最大 μs | 有效带宽中位数 GB/s |",
            "|---:|---:|---:|---:|---:|---:|"]
    for path in sorted(root.glob("vector_*.csv"), key=lambda p: int(p.stem.split("_")[-1])):
        data = list(csv.DictReader(path.open()))
        values = [float(r["avg_us"]) for r in data]
        bw = [float(r["effective_gbps"]) for r in data]
        out.append(f"| {data[0]['n']} | {len(values)} | {stats.median(values):.3f} | {min(values):.3f} | {max(values):.3f} | {stats.median(bw):.2f} |")
    trace_rows = []
    for name in ("baseline", "inplace"):
        path = root / "traces" / f"{name}.json"
        trace = json.loads(path.read_text())
        events = trace["traceEvents"]
        # Only actual complete GPU activity records; exclude CPU annotated ranges.
        gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
        origin = min(e["ts"] for e in gpu)
        for e in sorted(gpu, key=lambda e: e["ts"]):
            trace_rows.append(dict(variant=name, category=e["cat"], name=e["name"],
                                   relative_start_us=e["ts"]-origin, duration_us=e["dur"]))
    with (root / "trace_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    out += ["", "## 单次 profile 的 GPU 事件（不能代替性能基准）", "",
            "| 版本 | kernel 数 | DtoD 次数 | DtoD μs | mask kernel μs |",
            "|---|---:|---:|---:|---:|"]
    for name in ("baseline", "inplace"):
        events = [r for r in trace_rows if r["variant"] == name]
        copies = [r for r in events if r["category"] == "gpu_memcpy" and "DtoD" in r["name"]]
        masks = [r for r in events if r["category"] == "kernel" and "masked_fill" in r["name"]]
        out.append(f"| {name} | {sum(r['category']=='kernel' for r in events)} | {len(copies)} | {sum(r['duration_us'] for r in copies):.3f} | {sum(r['duration_us'] for r in masks):.3f} |")
    out += ["", "全部事件及其相对起点见 trace_events.csv；CPU 提交与同步等待需结合原始 JSON 的 cuda_runtime 轨道读取。",
            "有效带宽是逻辑读写量/时间；同一缓冲区反复使用，不能当作物理 DRAM 带宽。", ""]
    (root / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()

