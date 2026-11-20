import math
import statistics
import torch

torch.manual_seed(42)

def initqkv(S):
    B, H, D = 1, 8, 64
    q = torch.randn(B, H, S, D, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.ones(S, S, device="cuda", dtype=torch.bool).triu(1)
    return q, k, v, mask, D          # ← 返回出去

@torch.no_grad()
def attention(q, k, v, mask, D, inplace=False):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(D)

    if inplace:
        scores.masked_fill_(mask, float("-inf"))   # 原地修改，省一次分配
    else:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v


q, k, v, mask, D = initqkv(512)
baseline  = attention(q, k, v, mask, D, inplace=False)
optimized = attention(q, k, v, mask, D, inplace=True)
torch.testing.assert_close(baseline, optimized)
print("Correctness: PASS")


def bench(q, k, v, mask, D, inplace, repeats=100, rounds=5):
    for _ in range(20):                       # 预热
        attention(q, k, v, mask, D, inplace)
    torch.cuda.synchronize()

    results = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            attention(q, k, v, mask, D, inplace)
        end.record()
        end.synchronize()
        results.append(start.elapsed_time(end) * 1000 / repeats)
    return results

for S in [128, 512, 1024]:
    for i in range(3):
        for name, inplace in [
            ("baseline", False),
            ("inplace", True),
            ("inplace", True),
            ("baseline", False),
        ]:
            
            q, k, v, mask, D = initqkv(S)
            samples = bench(q, k, v, mask, D, inplace)
            print(f"S={S} group={i + 1} "
                f"{name} round_us={samples} "
                f"median_us={statistics.median(samples):.3f}")