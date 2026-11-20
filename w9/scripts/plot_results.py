"""Optional figures from saved data only; requires matplotlib."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    root = args.run_dir
    fig_dir = root / "figures"
    fig_dir.mkdir(exist_ok=True)
    rows = list(csv.DictReader((root / "attention.csv").open()))
    sizes = sorted({int(r["s"]) for r in rows})
    fig, axes = plt.subplots(1, len(sizes), figsize=(11, 3.8))
    for ax, s in zip(axes, sizes):
        values = [[float(r["avg_us"]) for r in rows if int(r["s"]) == s and r["variant"] == name]
                  for name in ("baseline", "inplace")]
        ax.boxplot(values, tick_labels=["baseline", "inplace"], showfliers=True)
        for x, sample in enumerate(values, 1):
            ax.scatter([x + ((i % 7)-3)*.018 for i in range(len(sample))], sample,
                       s=12, color="#2563eb", alpha=.55)
        ax.set_title(f"S={s}: {statistics.median(values[0])/statistics.median(values[1]):.3f}x")
        ax.set_ylabel("Mean time per round (us)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Attention: 30 round means / variant; 100 forwards / round (no profiler)")
    fig.tight_layout()
    fig.savefig(fig_dir / "attention.png", dpi=160)
    plt.close(fig)

    colors = dict(QK="#2563eb", Scale="#8b5cf6", DtoD="#ef4444", Mask="#f59e0b",
                  Softmax="#10b981", AV="#0891b2", Launch="#64748b", CopyAPI="#fb7185", Sync="#94a3b8")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for ax, variant in zip(axes, ("baseline", "inplace")):
        events = json.loads((root / "traces" / f"{variant}.json").read_text())["traceEvents"]
        gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
        cpu = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and
               e["name"] in ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaDeviceSynchronize")]
        origin = min(e["ts"] for e in gpu + cpu)
        for e in gpu:
            name = e["name"]
            label = ("DtoD" if e["cat"] == "gpu_memcpy" else "QK" if "sgemm_128x128_tn" in name else
                     "AV" if "sgemm_128x128_nn" in name else "Mask" if "masked_fill" in name else
                     "Softmax" if "softmax" in name else "Scale")
            start = e["ts"]-origin
            ax.broken_barh([(start, e["dur"])], (0, .65), color=colors[label])
            ax.text(start+e["dur"]/2, .32, label, ha="center", va="center", fontsize=8, color="white")
        for e in cpu:
            label = "Sync" if "Synchronize" in e["name"] else "CopyAPI" if "Memcpy" in e["name"] else "Launch"
            ax.broken_barh([(e["ts"]-origin, e["dur"])], (1, .65), color=colors[label])
        ax.set_yticks([.325, 1.325], ["GPU", "CPU API"])
        ax.set_ylim(-.2, 1.9)
        ax.set_title(f"{variant}: single profiled forward (independent time origin)", loc="left")
        ax.set_xlabel("Time since first selected event (us)")
        ax.grid(axis="x", alpha=.15)
    fig.legend(handles=[Patch(color=colors[k], label=k) for k in ("Launch", "CopyAPI", "Sync")],
               loc="upper right", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(fig_dir / "timeline.png", dpi=160)
    plt.close(fig)
    print(fig_dir)


if __name__ == "__main__":
    main()
