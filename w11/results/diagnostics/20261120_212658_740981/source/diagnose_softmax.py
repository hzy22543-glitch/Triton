"""Diagnose W11 with preallocated outputs, CUDA Graphs and launch traces."""
import ctypes
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark_softmax import FULL, ROOT, capture, check, save_json, torch, triton
from w11_softmax.softmax import softmax_kernel

SHAPES = FULL + [(256, 4096), (512, 4096), (2048, 4096),
                 (1024, 2048), (1024, 3072)]
NODES = 64
SAMPLES = 15
ROTATING_PAIRS = 8


def l2_bytes():
    # CUDA driver enum CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE in cuda.h.
    value = ctypes.c_int()
    status = ctypes.CDLL("libcuda.so.1").cuDeviceGetAttribute(ctypes.byref(value), 38, 0)
    if status:
        raise RuntimeError(f"cuDeviceGetAttribute(L2_CACHE_SIZE) failed: {status}")
    return value.value


def torch_out(x, y):
    return torch.softmax(x, dim=-1, out=y)


def triton_out(warps):
    def run(x, y):
        m, n = x.shape
        softmax_kernel[(m,)](x, y, n, BLOCK_SIZE=triton.next_power_of_2(n), num_warps=warps)
        return y
    return run


def capture_graph(fn, pairs):
    for x, y in pairs:
        fn(x, y)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for idx in range(NODES):
            fn(*pairs[idx % len(pairs)])
    return graph


def measure(graph, start, end):
    # Warm with the same buffer pattern immediately before taking the sample.
    graph.replay()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / NODES


def trace_launch(out, name, m, n, fn, pair):
    fn(*pair)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn(*pair)
        torch.cuda.synchronize()
    path = out / f"trace_{m}x{n}_{name}.json"
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if len(kernels) != 3:
        raise RuntimeError(f"Expected one kernel per call, got {len(kernels)} events in {path.name}")
    fields = ("grid", "block", "registers per thread", "shared memory", "occupancy")
    return dict(M=m, N=n, provider=name, trace=path.name,
                kernels=[dict(name=e["name"], **{k: e["args"].get(k) for k in fields}) for e in kernels])


@torch.inference_mode()
def main():
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.cuda.synchronize()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai"))
    out = ROOT / "results/diagnostics" / stamp.strftime("%Y%m%d_%H%M%S_%f")
    (out / "source").mkdir(parents=True, exist_ok=False)
    print("Evidence:", out, flush=True)
    hashes = {}
    for path in (Path(__file__), ROOT / "scripts/benchmark_softmax.py",
                 ROOT / "src/w11_softmax/softmax.py", ROOT / "src/w11_softmax/__init__.py"):
        content = path.read_bytes()
        hashes[path.name] = hashlib.sha256(content).hexdigest()
        (out / "source" / path.name).write_bytes(content)
    save_json(out / "environment.json", dict(timestamp=stamp.isoformat(), command=sys.argv,
              gpu=torch.cuda.get_device_name(), l2_bytes=l2_bytes(),
              sms=torch.cuda.get_device_properties(0).multi_processor_count,
              torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
              nodes=NODES, samples=SAMPLES, rotating_pairs=ROTATING_PAIRS,
              seed=42, dtype="fp32", source_sha256=hashes))
    (out / "nvidia-smi-before.txt").write_text(capture(["nvidia-smi"]), encoding="utf-8")
    records, traces = [], []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for m, n in SHAPES:
        pairs = []
        for _ in range(ROTATING_PAIRS):
            x = torch.randn((m, n), device="cuda", dtype=torch.float32)
            pairs.append((x, torch.empty_like(x)))
        providers = {"torch": torch_out, "triton4": triton_out(4)}
        if (m, n) == (1024, 4096):
            providers.update({f"triton{w}": triton_out(w) for w in (8, 16, 32)})
        jobs = {}
        for name, fn in providers.items():
            inputs = pairs if name in ("torch", "triton4") else pairs[:1]
            checks = [check(x, lambda _, x=x, y=y: fn(x, y)) for x, y in inputs]
            for mode, buffers in (("reuse", inputs[:1]), ("rotate", inputs)):
                if mode == "rotate" and len(inputs) == 1:
                    continue
                row = dict(M=m, N=n, provider=name, mode=mode, pairs=len(buffers),
                           logical_footprint_bytes=len(buffers) * m * n * 8,
                           correctness=checks[:len(buffers)], samples_us=[])
                jobs[(name, mode)] = (capture_graph(fn, buffers), row)
        rng = random.Random(m * 10000 + n)
        for _ in range(SAMPLES):
            order = list(jobs)
            rng.shuffle(order)
            for key in order:
                graph, row = jobs[key]
                row["samples_us"].append(measure(graph, start, end))
        for _, row in jobs.values():
            row["median_us"] = statistics.median(row["samples_us"])
            records.append(row)
        save_json(out / "graph_timings.json", records)
        if (m, n) in ((32, 128), (4096, 128), (1024, 1000), (1024, 1024),
                      (1024, 1025), (32, 4096), (1024, 4096), (4096, 4096)):
            traces.extend(trace_launch(out, name, m, n, fn, pairs[0])
                          for name, fn in providers.items())
            save_json(out / "launches.json", traces)
        print(f"{m}x{n}: " + ", ".join(
            f"{name}/{mode}={row['median_us']:.3f} us" for (name, mode), (_, row) in jobs.items()), flush=True)
    (out / "nvidia-smi-after.txt").write_text(capture(["nvidia-smi"]), encoding="utf-8")
    save_json(out / "status.json", dict(status="PASS", shapes=len(SHAPES), timing_groups=len(records),
                                         trace_groups=len(traces), samples_per_group=SAMPLES))


if __name__ == "__main__":
    main()

