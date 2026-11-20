"""ABBA comparison: same inputs, per-round CSV, no profiler."""
import argparse
import csv
from pathlib import Path
import statistics
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attention import attention, make_inputs


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=positive, default=[128, 512, 1024])
    parser.add_argument("--groups", type=positive, default=3)
    parser.add_argument("--rounds", type=positive, default=5)
    parser.add_argument("--repeats", type=positive, default=100)
    parser.add_argument("--warmup", type=positive, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if max(args.sizes) > 4096:
        parser.error("this FP32 learning benchmark supports S <= 4096")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as f:
        fields = ["s", "dtype", "seed", "group", "position", "variant", "round",
                  "repeats", "warmup", "avg_us"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in args.sizes:
            inputs = make_inputs(s, args.seed)
            torch.testing.assert_close(attention(*inputs), attention(*inputs, inplace=True),
                                       rtol=1e-5, atol=1e-6)
            print(f"S={s} correctness PASS", flush=True)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for group in range(1, args.groups + 1):
                for position, inplace in enumerate((False, True, True, False), 1):
                    name = "inplace" if inplace else "baseline"
                    for _ in range(args.warmup):
                        attention(*inputs, inplace=inplace)
                    torch.cuda.synchronize()
                    samples = []
                    for round_id in range(1, args.rounds + 1):
                        start.record()
                        for _ in range(args.repeats):
                            attention(*inputs, inplace=inplace)
                        end.record()
                        end.synchronize()
                        us = start.elapsed_time(end) * 1000 / args.repeats
                        samples.append(us)
                        writer.writerow(dict(s=s, dtype="float32", seed=args.seed, group=group,
                                             position=position, variant=name, round=round_id,
                                             repeats=args.repeats, warmup=args.warmup, avg_us=us))
                    f.flush()
                    print(f"S={s} group={group} position={position} {name} "
                          f"median={statistics.median(samples):.3f} us "
                          f"range=[{min(samples):.3f}, {max(samples):.3f}]", flush=True)


if __name__ == "__main__":
    main()

