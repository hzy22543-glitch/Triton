"""Local W11 correctness suite; no sibling W10 directory required."""
import argparse
import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark_softmax import ROOT, capture, check, save_json, torch, triton, triton_softmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = ROOT / 'results/correctness' / stamp.strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    env = dict(timestamp=stamp.isoformat(), python=sys.version, platform=platform.platform(),
               torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(), seed=args.seed, command=sys.argv, source_sha256={})
    (out / 'source').mkdir()
    for path in [Path(__file__), ROOT / 'scripts/benchmark_softmax.py',
                 ROOT / 'src/w11_softmax/softmax.py', ROOT / 'src/w11_softmax/__init__.py']:
        data = path.read_bytes()
        env['source_sha256'][path.name] = hashlib.sha256(data).hexdigest()
        (out / 'source' / path.name).write_bytes(data)
    save_json(out / 'environment.json', env)
    (out / 'nvidia-smi.txt').write_text(capture(['nvidia-smi']), encoding='utf-8')
    records = []

    def run(name, x, error=None, message=None):
        row = dict(name=name, shape=list(x.shape), dtype=str(x.dtype))
        try:
            if error:
                try:
                    triton_softmax(x)
                except error as exc:
                    if message not in str(exc):
                        raise AssertionError(f'unexpected rejection: {exc}')
                    row['reason'] = str(exc)
                else:
                    raise AssertionError('invalid input accepted')
            elif x.numel() == 0:
                torch.testing.assert_close(triton_softmax(x), torch.softmax(x, dim=1))
            else:
                row.update(check(x, triton_softmax))
            torch.cuda.synchronize()
            row['status'] = 'PASS'
        except Exception:
            row.update(status='FAIL', traceback=traceback.format_exc())
            try:
                torch.save(x.cpu(), out / f'failure_{len(records):03d}.pt')
            except Exception:
                pass
        records.append(row)
        save_json(out / 'cases.json', records)
        print(row['status'], name, tuple(x.shape), x.dtype)

    with torch.inference_mode():
        for dtype in [torch.float32, torch.float16]:
            for shape in [(1, 1), (2, 3), (3, 5), (7, 255), (7, 256), (7, 257),
                          (32, 1000), (16, 4096), (1024, 257), (0, 5)]:
                run('random_or_empty', torch.randn(shape, device='cuda', dtype=dtype))
            for name, values in [
                ('large_shift', [[1, 2, 3], [1001, 1002, 1003]]),
                ('negative', [[-5, -4, -3, -2, -1]]),
                ('equal', [[7] * 5] * 2),
                ('extreme_finite', [[-10000, 0, 10000], [10000, 10000, -10000]]),
            ]:
                run(name, torch.tensor(values, device='cuda', dtype=dtype))
        for name, x, error, message in [
            ('zero_width', torch.empty((2, 0), device='cuda'), ValueError, '行宽'),
            ('too_wide', torch.empty((2, 4097), device='cuda'), ValueError, '行宽'),
            ('noncontiguous', torch.empty((3, 5), device='cuda').T, ValueError, '连续'),
            ('one_dimensional', torch.empty(5, device='cuda'), ValueError, '二维'),
            ('cpu', torch.empty((3, 5)), ValueError, 'GPU'),
            ('float64', torch.empty((3, 5), device='cuda', dtype=torch.float64), TypeError, 'FP32'),
        ]:
            run(name, x, error, message)
    failed = sum(r['status'] == 'FAIL' for r in records)
    summary = dict(total=len(records), passed=len(records)-failed, failed=failed)
    save_json(out / 'summary.json', summary)
    print(json.dumps(summary), '\nEvidence:', out)
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
