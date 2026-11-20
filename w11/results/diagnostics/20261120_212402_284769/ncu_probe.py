"""Profile 3 launches per provider, using identical buffer layouts for both."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from diagnose_softmax import capture_graph, torch, torch_out, triton_out

with torch.inference_mode():
    m, pairs = map(int, sys.argv[1:3])
    mode = sys.argv[3]
    torch.manual_seed(42)
    buffers = []
    for _ in range(pairs):
        x = torch.randn((m, 4096), device="cuda", dtype=torch.float32)
        buffers.append((x, torch.empty_like(x)))
    for name, fn in (("torch", torch_out), ("triton4", triton_out(4))):
        print(f"provider={name} shape={m}x4096 pairs={pairs} mode={mode}", flush=True)
        if mode == "graph":
            graph = capture_graph(fn, buffers)
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            for _ in range(3):
                with torch.cuda.nvtx.range("measure"):
                    graph.replay()
                torch.cuda.synchronize()
        else:
            for i in range(40):
                fn(*buffers[i % pairs])
            torch.cuda.synchronize()
            for i in range(3):
                with torch.cuda.nvtx.range("measure"):
                    fn(*buffers[i % pairs])
            torch.cuda.synchronize()

