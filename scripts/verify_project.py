"""Read-only evidence audit, optionally followed by serial GPU correctness checks.

Run from WSL with the existing environment. Never mixes W9 Event timing and
W11 host timing. Does not launch full performance benchmarks or overwrite data.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'w11/scripts'))
from analyze_results import audit_run


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_docs():
    paths = [ROOT / 'README.md'] + list((ROOT / 'docs').glob('*.md'))
    for path in (ROOT/'scripts').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    for week in ('w9', 'w10', 'w11'):
        base = ROOT / week
        paths += list(base.glob('*.md')) + list((base/'docs').glob('*.md')) + list((base/'reports').glob('*.md'))
        for folder in ('src', 'scripts', 'tests', 'benchmarks'):
            for path in (base/folder).rglob('*.py'):
                ast.parse(path.read_text(encoding='utf-8-sig'))
    for path in paths:
        text = path.read_text(encoding='utf-8-sig')
        require(text.count('```') % 2 == 0, f'unclosed code fence: {path}')
        text = re.sub(r'```.*?```', '', text, flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            require((path.parent / target.split('#')[0]).exists(), f'broken link: {path}: {target}')
    return {'documents_checked': len(paths)}


def audit_w9():
    run = ROOT / 'w9/results/final_20261104_2'
    with (run/'attention.csv').open() as f:
        rows = list(csv.DictReader(f))
    expected = {(s,g,p,r) for s in (128,512,1024) for g in (1,2,3) for p in (1,2,3,4) for r in range(1,6)}
    actual = {(int(r['s']),int(r['group']),int(r['position']),int(r['round'])) for r in rows}
    require(actual == expected and len(rows) == len(expected), 'W9 ABBA coverage mismatch')
    summary = (run/'summary.md').read_text()
    for r in rows:
        require(r['variant'] == ('baseline' if int(r['position']) in (1,4) else 'inplace'), 'W9 ABBA order')
        require(r['dtype']=='float32' and int(r['seed'])==42 and int(r['repeats'])==100
                and int(r['warmup'])==100, 'W9 metadata mismatch')
        require(math.isfinite(float(r['avg_us'])) and float(r['avg_us'])>0, 'invalid W9 timing')
    for s in (128,512,1024):
        for name in ('baseline','inplace'):
            values=[float(r['avg_us']) for r in rows if int(r['s'])==s and r['variant']==name]
            line=f'| {s} | {name} | 30 | {statistics.median(values):.3f} | {min(values):.3f} | {max(values):.3f} |'
            require(line in summary, 'W9 attention summary mismatch')
    for n in (1024,262144,2097152,16777216):
        with (run/f'vector_{n}.csv').open() as f:
            values=list(csv.DictReader(f))
        require(len(values)==5 and {int(r['round']) for r in values}==set(range(1,6)), 'W9 vector rounds')
        for r in values:
            require(int(r['n'])==n and int(r['repeats'])==100 and int(r['warmup'])==100
                    and int(r['block_size'])==256, 'W9 vector metadata')
            us=float(r['avg_us'])
            require(math.isfinite(us) and us>0, 'invalid W9 vector time')
            # CSV stores rounded microseconds and bandwidth, allow rounding error.
            require(math.isclose(float(r['effective_gbps']),3*n*4/(us*1000),rel_tol=2e-6,abs_tol=1e-6),
                    'W9 effective bandwidth mismatch')
    # Regenerate summaries in isolation; immutable original CSV/trace never modified.
    import shutil
    with tempfile.TemporaryDirectory(prefix='w9-audit-') as tmp:
        temp=Path(tmp)
        for path in run.glob('*.csv'):
            shutil.copy2(path,temp/path.name)
        shutil.copytree(run/'traces',temp/'traces')
        subprocess.run([sys.executable,str(ROOT/'w9/scripts/summarize.py'),str(temp)],
                       check=True,stdout=subprocess.DEVNULL)
        for name in ('summary.md','trace_events.csv'):
            require((temp/name).read_text()==(run/name).read_text(), f'W9 regenerated {name} mismatch')
    limitations=[]
    if (run/'memcheck_unavailable.txt').exists():
        limitations.append((run/'memcheck_unavailable.txt').read_text().strip())
    require('Ran 3 tests' in (run/'tests.log').read_text() and '\nOK' in (run/'tests.log').read_text(),
            'W9 historical test log incomplete')
    return dict(attention_samples=len(rows),vector_samples=20,summary_and_trace='MATCH',limitations=limitations)


def audit_correctness(base):
    summaries=[]
    for path in sorted(base.glob('*/summary.json')):
        cases=read(path.parent/'cases.json'); summary=read(path)
        passed=sum(r.get('status')=='PASS' for r in cases)
        require(all(r.get('status') in ('PASS','FAIL') for r in cases), 'unknown test status')
        require(summary==dict(total=len(cases),passed=passed,failed=len(cases)-passed), f'counts mismatch: {path}')
        # Historical failures remain visible; current regression is executed separately.
        summaries.append(dict(run=path.parent.name,**summary))
    require(bool(summaries), f'no correctness evidence: {base}')
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true',help='Also rebuild W9 and run current correctness suites serially')
    args=parser.parse_args()
    stamp=datetime.now(ZoneInfo('Asia/Shanghai'))
    out=ROOT/'reports/checks'/stamp.strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    checks=[]
    report=dict(timestamp=stamp.isoformat(),python=sys.executable,gpu_requested=args.gpu,checks=checks)

    def checkpoint(name,fn):
        try:
            details=fn()
            checks.append(dict(name=name,status='PASS',details=details))
        except Exception as exc:
            checks.append(dict(name=name,status='FAIL',error=str(exc)))
            print(f'FAIL {name}: {exc}',flush=True)
        else:
            print(f'PASS {name}',flush=True)
        (out/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')

    def command(name,cmd,cwd):
        p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True)
        (out/f'{name}.log').write_text(p.stdout+p.stderr,encoding='utf-8')
        require(p.returncode==0,f'exit {p.returncode}; see {name}.log')
        return dict(command=cmd,cwd=str(cwd.relative_to(ROOT)),log=f'{name}.log')

    checkpoint('docs_and_syntax',audit_docs)
    checkpoint('w9_archived_evidence',audit_w9)
    checkpoint('w10_archived_correctness',lambda:audit_correctness(ROOT/'w10/results/runs'))
    checkpoint('w11_archived_correctness',lambda:audit_correctness(ROOT/'w11/results/correctness'))

    def w11_audit():
        paths=sorted(p for p in (ROOT/'w11/results/runs').iterdir() if p.is_dir())
        require(bool(paths),'no W11 runs')
        results=[audit_run(p)[1] for p in paths]
        return dict(runs=len(paths),shape_dtype_conditions=sum(map(len,results)),
                    timing_groups=sum(len(read(p/'raw.json')) for p in paths))

    checkpoint('w11_archived_benchmarks',w11_audit)
    checkpoint('w11_auditor_tests',lambda:command('w11_auditor_tests',
               [sys.executable,'-m','unittest','discover','-s','tests','-v'],ROOT/'w11'))
    if args.gpu:
        jobs=[('w9_rebuild_correctness',['make','-B','test',f'PYTHON={sys.executable}'],'w9'),
              ('w10_correctness',[sys.executable,'scripts/run_experiments.py'],'w10'),
              ('w10_optimized_python',[sys.executable,'-O','scripts/run_experiments.py','--suite','vector_add'],'w10'),
              ('w11_correctness',[sys.executable,'scripts/check_correctness.py'],'w11')]
        for name,cmd,week in jobs:
            checkpoint(name,lambda name=name,cmd=cmd,week=week:command(name,cmd,ROOT/week))
        def finite_cli():
            for value in ('nan','inf','-1'):
                p=subprocess.run([sys.executable,'scripts/benchmark_softmax.py','--min-run-time',value],
                                 cwd=ROOT/'w11',text=True,capture_output=True)
                require(p.returncode==2 and 'Use rounds' in p.stderr, f'accepted invalid duration: {value}')
            return dict(rejected=['nan','inf','-1'],exit_code=2)
        checkpoint('w11_cli_invalid_duration',finite_cli)
    report['status']='PASS' if all(r['status']=='PASS' for r in checks) else 'FAIL'
    report['scope']='Historical evidence audit plus serial current correctness' if args.gpu else 'Offline evidence audit only'
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for week in ('w9','w10','w11') for folder in ('src','scripts','tests','benchmarks')
        for p in (ROOT/week/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.sh')}
    for path in [Path(__file__), ROOT/'w9/Makefile']:
        report['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{report["status"]}: {out}',flush=True)
    return report['status']!='PASS'


if __name__=='__main__':
    raise SystemExit(main())
