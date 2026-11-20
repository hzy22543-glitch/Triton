"""Short traces for kernel counts only; never used as benchmark timings."""
import hashlib
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from benchmark_softmax import ROOT, capture, naive_softmax, torch_softmax, triton_softmax, torch, save_json


def main():
    stamp = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = ROOT / 'results/profiles' / stamp.strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out / 'source').mkdir()
    hashes = {}
    for path in [Path(__file__), ROOT/'scripts/benchmark_softmax.py', ROOT/'src/w11_softmax/softmax.py',
                 ROOT/'src/w11_softmax/__init__.py']:
        content = path.read_bytes()
        (out/'source'/path.name).write_bytes(content)
        hashes[path.name] = hashlib.sha256(content).hexdigest()
    save_json(out/'environment.json', dict(timestamp=stamp.isoformat(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), command=sys.argv, source_sha256=hashes,
        purpose='kernel count only; profiled durations must not be used in the benchmark table'))
    (out/'nvidia-smi.txt').write_text(capture(['nvidia-smi']), encoding='utf-8')
    records = []
    with torch.inference_mode():
        for m,n in [(32,128), (1024,4096)]:
            x = torch.randn((m,n),device='cuda')
            for name,fn in [('naive',naive_softmax),('torch',torch_softmax),('triton',triton_softmax)]:
                for _ in range(25):
                    fn(x)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function(f'{name}_{m}x{n}'):
                        for _ in range(5):
                            fn(x)
                        torch.cuda.synchronize()
                trace = f'{name}_{m}x{n}.json'
                prof.export_chrome_trace(str(out/trace))
                events = json.loads((out/trace).read_text())['traceEvents']
                kernels = [e for e in events if e.get('cat') == 'kernel']
                records.append(dict(M=m,N=n,provider=name,calls=5,kernel_events=len(kernels),
                                    kernels_per_call=len(kernels)/5,
                                    kernel_names=sorted({e['name'] for e in kernels}),trace=trace))
                print(name,m,n,'kernel_events=',len(kernels))
    save_json(out/'summary.json',records)
    save_json(out/'status.json',dict(status='PASS' if all(r['kernel_events']>0 for r in records) else 'UNAVAILABLE',
                                   scope='kernel count evidence only'))
    print('Evidence:',out)
    return 0 if all(r['kernel_events']>0 for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
