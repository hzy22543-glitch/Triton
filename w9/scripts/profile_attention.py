"""Capture the same implementation and input as the benchmark, separately."""
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attention import attention, make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_inputs(1024, seed=42)
    for inplace in (False, True):
        name = "inplace" if inplace else "baseline"
        for _ in range(100):
            attention(*inputs, inplace=inplace)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            attention(*inputs, inplace=inplace, annotate=True)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output_dir / f"{name}.json"))
        (args.output_dir / f"{name}_table.txt").write_text(
            prof.key_averages().table(sort_by="self_device_time_total", row_limit=30), encoding="utf-8")
        print(f"Captured {name}")


if __name__ == "__main__":
    main()
