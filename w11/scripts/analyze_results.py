"""Audit raw benchmark evidence and regenerate comparisons without running the GPU.

Only --plots requires matplotlib. All statistics are recomputed from raw blocks,
independently of benchmark_softmax.summarize; historical records are never edited.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ('naive', 'torch', 'triton')
MINIMUM = [(1, 128), (4096, 128), (1024, 1000), (1024, 4096)]
FULL = MINIMUM + [(32, 128), (32, 4096), (1024, 1024), (1024, 1025), (4096, 4096)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    require(math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-7), f'statistic mismatch: {a} != {b}')


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def audit_run(path):
    env, status = read(path / 'environment.json'), read(path / 'status.json')
    require(status['status'] == 'PASS', 'run did not pass')
    for name, digest in env['source_sha256'].items():
        require(hashlib.sha256((path / 'source' / name).read_bytes()).hexdigest() == digest,
                f'source hash mismatch: {name}')
    args = env['args']
    require(args['suite'] in ('minimum', 'full'), 'unknown suite')
    require(args['dtype'] in ('fp32', 'fp16'), 'unknown dtype')
    require(type(args['rounds']) is int and args['rounds'] >= 3, 'invalid rounds')
    shapes = MINIMUM if args['suite'] == 'minimum' else FULL
    rounds, dtype = args['rounds'], args['dtype']
    raw, checks = read(path / 'raw.json'), read(path / 'correctness.json')
    expected = {(m, n, dtype, p, r) for m, n in shapes for p in PROVIDERS for r in range(rounds)}
    keyed = {(r['M'], r['N'], r['dtype'], r['provider'], r['round']): r for r in raw}
    require(set(keyed) == expected and len(raw) == len(expected), 'missing/duplicate timing groups')
    check_keys = {(r['M'], r['N'], r['dtype'], r['provider']) for r in checks}
    require(check_keys == {k[:4] for k in expected} and len(checks) == len(shapes) * 3,
            'missing/duplicate correctness checks')
    require(all(r['status'] == 'PASS' for r in checks), 'failed correctness check')
    for r in checks:
        for field in ('max_abs_error', 'max_row_sum_error', 'rtol', 'atol', 'row_sum_atol'):
            require(math.isfinite(r[field]) and r[field] >= 0, f'invalid correctness metric: {field}')
        require(r['max_row_sum_error'] <= r['row_sum_atol'], 'row sum check exceeds tolerance')
    require(status['shapes'] == len(shapes) and status['timed_groups'] == len(raw)
            and status['correctness_checks'] == len(checks), 'status counts mismatch')
    # No actual output tensors were saved; we validate recorded checks, not re-prove allclose.
    for r in raw:
        blocks, calls = r['raw_block_seconds'], r['number_per_run']
        require(isinstance(calls, int) and calls > 0 and blocks, 'invalid block size')
        require(len(blocks) == r['sample_blocks'], 'block count mismatch')
        require(all(math.isfinite(t) and t > 0 for t in blocks), 'invalid raw timing')
        times = [t / calls * 1e6 for t in blocks]
        close(statistics.median(times), r['median_us'])
        close(quantile(times, .75) - quantile(times, .25), r['iqr_us'])
    stored = read(path / 'summary.json')
    stored_keys = {(r['M'], r['N'], r['dtype'], r['provider']): r for r in stored}
    require(set(stored_keys) == check_keys and len(stored) == len(check_keys), 'summary coverage mismatch')
    calculated, comparisons = [], []
    for m, n in shapes:
        for provider in PROVIDERS:
            group = [keyed[(m, n, dtype, provider, r)] for r in range(rounds)]
            times = [r['median_us'] for r in group]
            center = statistics.median(times)
            pair = [keyed[(m, n, dtype, 'torch', r)]['median_us'] / times[r] for r in range(rounds)]
            summary = dict(M=m, N=n, dtype=dtype, provider=provider, rounds=rounds, median_us=center,
                           min_round_us=min(times), max_round_us=max(times),
                           repeat_range_pct=100*(max(times)-min(times))/center,
                           max_within_iqr_pct=max(100*r['iqr_us']/r['median_us'] for r in group),
                           speedup_vs_torch=statistics.median(pair),
                           effective_GBps=2*m*n*(2 if dtype == 'fp16' else 4)/(center*1e3))
            saved = stored_keys[(m, n, dtype, provider)]
            for field, value in summary.items():
                if isinstance(value, (int, float)):
                    close(saved[field], value)
                else:
                    require(saved[field] == value, f'summary field mismatch: {field}')
            calculated.append(summary)
        t = {p: [keyed[(m, n, dtype, p, r)]['median_us'] for r in range(rounds)] for p in PROVIDERS}
        comparisons.append(dict(run=path.name, dtype=dtype, M=m, N=n,
            naive_us=statistics.median(t['naive']), torch_us=statistics.median(t['torch']),
            triton_us=statistics.median(t['triton']),
            vs_naive=statistics.median(a/b for a,b in zip(t['naive'], t['triton'])),
            vs_torch=statistics.median(a/b for a,b in zip(t['torch'], t['triton'])),
            triton_min_us=min(t['triton']), triton_max_us=max(t['triton']),
            max_iqr_pct=max(s['max_within_iqr_pct'] for s in calculated if s['M']==m and s['N']==n),
            max_repeat_range_pct=max(s['repeat_range_pct'] for s in calculated if s['M']==m and s['N']==n),
            min_sample_blocks=min(r['sample_blocks'] for r in raw if r['M']==m and r['N']==n)))
    with (path / 'summary.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == len(stored), 'CSV row count mismatch')
    csv_keys = set()
    for row in rows:
        key = (int(row['M']), int(row['N']), row['dtype'], row['provider'])
        require(key not in csv_keys, 'duplicate CSV row')
        csv_keys.add(key)
        for field, value in stored_keys[key].items():
            if isinstance(value, (int, float)):
                close(row[field], value)
            else:
                require(row[field] == value, 'CSV value mismatch')
    return env, comparisons


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f"{r['M']}x{r['N']}" for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for offset, (name, color) in enumerate([('naive', '#718096'), ('torch', '#2563eb'), ('triton', '#d97706')]):
        axes[0].bar([i+(offset-1)*.25 for i in range(len(rows))], [r[name+'_us'] for r in rows],
                    width=.24, label=name, color=color)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Amortized host time (us, log scale)')
    axes[0].legend()
    axes[0].set_title(f"Softmax FP32 | run {rows[0]['run']} | warmed, same input reused")
    axes[1].bar(range(len(rows)), [r['vs_torch'] for r in rows], color='#d97706')
    axes[1].axhline(1, color='#111827', linestyle='--', label='torch baseline')
    axes[1].set_ylabel('torch time / Triton time')
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(range(len(rows)), labels, rotation=25)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[1].set_xlabel('M x N | median of per-round ratios; no confidence intervals implied')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--plot-run', help='Explicit full FP32 run for the figure; default latest')
    args = parser.parse_args()
    lines = ['# 原始数据核验与跨运行对照', '',
             '由 scripts/analyze_results.py 生成；不修改原始记录。倍率 >1 表示 Triton 更快。',
             '核验覆盖源码快照哈希、形状/实现/轮数完整性、正确性记录、原始块统计、JSON/CSV 汇总一致性。',
             '未保存完整输出张量，离线核验不等于重新执行数值测试。', '',
             '| 运行 | dtype / suite | 检查 | 计时组 | 状态 |', '|---|---|---:|---:|---|']
    all_rows, candidates = [], {}
    for path in sorted((ROOT / 'results/runs').iterdir()):
        if not path.is_dir():
            continue
        try:
            env, rows = audit_run(path)
        except Exception as exc:
            raise SystemExit(f'AUDIT FAILED {path.name}: {exc}')
        all_rows.extend(rows)
        a = env['args']
        lines.append(f"| [{path.name}](../results/runs/{path.name}/summary.md) | {a['dtype']} / {a['suite']} | {len(rows)*3} | {len(rows)*3*a['rounds']} | PASS |")
        if a['dtype'] == 'fp32' and a['suite'] == 'full':
            candidates[path.name] = rows
    require(bool(all_rows), 'no runs')
    lines += ['', '## 全量对照（保留异常和退化）', '',
              'max 波动列取该形状三种实现中的最大值；10% 仅作排查提示，不是显著性检验。', '',
              '| 运行 | dtype | M×N | naive μs | torch μs | Triton μs | vs naive | vs torch | max轮间跨度 | max轮内IQR | 最少块数 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in all_rows:
        lines.append(f"| {r['run']} | {r['dtype']} | {r['M']}×{r['N']} | {r['naive_us']:.3f} | {r['torch_us']:.3f} | {r['triton_us']:.3f} | {r['vs_naive']:.3f} | {r['vs_torch']:.3f} | {r['max_repeat_range_pct']:.1f}% | {r['max_iqr_pct']:.1f}% | {r['min_sample_blocks']} |")
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    (out / '数据核验与对照.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out / 'comparisons.json').write_text(json.dumps(all_rows, indent=2), encoding='utf-8')
    if args.plots:
        run = args.plot_run or sorted(candidates)[-1]
        plot(candidates[run], out / 'figures/softmax_fp32.png')
        (out / 'figures/source.json').write_text(json.dumps(dict(run=run,
            source=f'results/runs/{run}/raw.json', statistic='median of per-round medians and ratios'), indent=2), encoding='utf-8')
    print(f'PASS: {sum(p.is_dir() for p in (ROOT / "results/runs").iterdir())} runs, {len(all_rows)} shape/dtype comparisons audited')


if __name__ == '__main__':
    main()
