"""W10 自动化正确性测试：对照参考实现，检查边界，保存结果。

通过 scripts/run_experiments.py 启动，而非直接执行本文件。
阅读顺序：main 的参数解析 → 最下方用例循环 → check → 环境与文件留档。
这不是性能基准，也不是断点调试器；测试失败提供进一步调试的线索。
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import triton
from w10_triton.softmax import triton_softmax
from w10_triton.vector_add import triton_add

# __file__ 是本文件路径；parents[0] 是 tests，parents[1] 是项目 w10。
# 使用文件位置定位项目，不依赖用户从哪个目录启动脚本。
ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    # ① 解析命令行配置。argv=None 时 argparse 读取进程命令行参数；
    # 若从 Python 调用，也可传入 ["--suite", "softmax"] 这样的字符串列表。
    parser = argparse.ArgumentParser(description=__doc__)
    # choices 限制可输入值，default 指省略此参数时选什么。
    parser.add_argument('--suite', choices=['all', 'softmax', 'vector_add'], default='all')
    # 命令行本来都是字符串，type=int 将 seed 转成整数。
    parser.add_argument('--seed', type=int, default=42)
    # --output-dir 解析后通过 args.output_dir 访问：连字符变为下划线。
    # Path 支持用 / 拼接路径；不代表数学除法。
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'runs')
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        # 配置/环境不满足条件时提前退出，不能把没有跑测试算成通过。
        parser.error('CUDA is required; activate the WSL GPU environment.')

    # ② 准备本次实验。时间戳子目录隔离每次结果，避免覆盖旧数据。
    stamp = datetime.now(ZoneInfo('Asia/Shanghai'))
    out = args.output_dir / stamp.strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    # 同环境、同调用顺序下便于重现随机输入；不是让所有测试使用同一张量。
    torch.manual_seed(args.seed)

    # ③ 环境留档（工程辅助，可在首次学习时跳过）。
    # 当前实验按单卡 cuda:0 组织；这些信息用来解释在哪个环境跑过。
    props = torch.cuda.get_device_properties(0)
    env = dict(timestamp=stamp.isoformat(), python=sys.version, platform=platform.platform(),
               torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
               gpu=props.name, memory_bytes=props.total_memory,
               capability=list(torch.cuda.get_device_capability(0)), seed=args.seed,
               suite=args.suite, command=sys.argv, source_sha256={})
    for name in ['src/w10_triton/__init__.py', 'src/w10_triton/softmax.py',
                 'src/w10_triton/vector_add.py', 'tests/correctness.py',
                 'scripts/run_experiments.py']:
        # 源码内容指纹：代码或注释改变都会改变哈希，不代表正确性评分。
        env['source_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    # dumps 把字典变成 JSON 文本，write_text 才真正写入磁盘。
    (out / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    for filename, command in [('nvidia-smi.txt', ['nvidia-smi']),
                              ('pip-freeze.txt', [sys.executable, '-m', 'pip', 'freeze'])]:
        # 在子进程中执行命令并捕获输出。sys.executable 保证 pip 属于当前 Python。
        # check=False 不因命令非零退出码直接抛异常；这里只采集环境信息。
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (out / filename).write_text(result.stdout + result.stderr, encoding='utf-8')
    records = []  # 每完成一个用例，就追加一个结果字典。

    def check(name, x, operation, reference=None, expected_error=None, message=None, other=None):
        # ④ 单个用例的公共检查流程。
        # operation/reference 是“稍后调用的函数”，不是已经算好的张量。
        # x 用于 shape/dtype/device 检查和失败保存；other 保存 add 的第二个输入。
        # expected_error/message 用于非法输入：要求正确类型和原因的异常。
        # 定义在 main 内部，能直接使用外层的 out 和 records（闭包）。
        record = dict(name=name, shape=list(x.shape), dtype=str(x.dtype),
                      kind='rejection' if expected_error else 'numerical')
        try:
            if expected_error:
                # 分支 A：负向测试。这里“按预期报错”才是 PASS。
                try:
                    operation()
                except expected_error as exc:
                    # 例如应因行宽被拒绝，不能因另一个无关原因碰巧报错。
                    if message and message not in str(exc):
                        raise AssertionError(f'Wrong rejection reason: {exc}')
                    record['rejection_reason'] = str(exc)
                else:
                    # try/except/else：try 没有抛异常时才进入 else。
                    # 非法输入居然被接受，所以主动制造一次测试失败。
                    raise AssertionError('Invalid input was accepted')
            else:
                # 分支 B：正向测试。执行算子与参考，检查实际数值。
                actual = operation()
                expected = reference()
                # 元信息也属于接口约定，不能只有数值相近但类型/设备错误。
                if (actual.shape, actual.dtype, actual.device) != (x.shape, x.dtype, x.device):
                    raise AssertionError('Output metadata mismatch')
                rtol, atol = (1e-3, 1e-5) if x.dtype == torch.float16 else (1e-5, 1e-6)
                record.update(rtol=rtol, atol=atol)
                # 对每个元素要求 |actual-expected| <= atol + rtol*|expected|。
                # 不满足时 assert_close 抛异常，交给外层 except 留档。
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                if actual.numel():
                    # numel 为 0 时不能做 max。float() 转 FP32 后算差；
                    # item() 将单元素张量取成 Python 数字，方便存进 JSON。
                    record['max_abs_error'] = (actual.float() - expected.float()).abs().max().item()
                    if name.startswith('softmax'):
                        # 本项目按用例名称前缀开启 Softmax 专用检查。
                        # 新增 Softmax 用例也要保留 softmax 前缀。
                        # all/any 将逐元素条件归约；概率应有限且非负。
                        if not torch.isfinite(actual).all().item() or (actual < 0).any().item():
                            raise AssertionError('Nonfinite or negative probabilities')
                        # FP32 累加，更准确地观察 FP16 输出中已经存在的舍入偏差。
                        sums = actual.float().sum(dim=1)
                        sum_atol = 1e-3 if x.dtype == torch.float16 else 1e-6
                        record.update(row_sum_atol=sum_atol,
                                      max_row_sum_error=(sums - 1).abs().max().item())
                        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=sum_atol)
                else:
                    # 空输出已在上面完成元信息和参考对照，只跳过数值归约。
                    record['kind'] = 'empty'
            # 等待 GPU 完成，尽量把异步执行错误归到当前用例；这里不计时。
            torch.cuda.synchronize()
            record['status'] = 'PASS'
        except Exception:
            # 外层捕获两类测试的失败，记下完整堆栈，随后继续其他用例。
            # 所以终端继续打印不代表全部通过，最终还要检查 failed 和退出码。
            record.update(status='FAIL', traceback=traceback.format_exc())
            # 保存到 CPU，便于之后重现。严重 CUDA 错误下此复制也可能失败。
            # :03d 表示编号至少 3 位，如 failure_005.pt。
            try:
                torch.save({'x': x.cpu(), 'other': other.cpu() if other is not None else None},
                           out / f'failure_{len(records):03d}.pt')
            except Exception as exc:
                # A device failure must not prevent the original failure from being logged.
                record['input_save_error'] = str(exc)
        records.append(record)
        print(f"{record['status']:4s} {name:38s} {str(tuple(x.shape)):14s} {x.dtype}")
        # 每完成一例就写出当前累计结果；ensure_ascii=False 保留中文错误原因。
        (out / 'cases.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')

    # ⑤ 构造具体用例并立即交给 check；先读这部分，再回头看 check 更容易理解。
    if args.suite in ('all', 'vector_add'):
        # 0 测空输入，255/256/257 测块边界前后。
        for n in [0, 1, 10, 255, 256, 257, 1000, 4096]:
            x = torch.randn(n, device='cuda')
            y = torch.randn_like(x)
            # lambda 延迟执行，让算子错误发生在 check 的 try 内部。
            check(f'vector_add_{n}', x, lambda: triton_add(x, y), lambda: x + y, other=y)

        # Explicit interface rejection remains enabled under python -O.
        valid = torch.empty(4, device='cuda')
        for name, x, y, error, message in [
            ('cpu', torch.empty(4), torch.empty(4), ValueError, 'GPU'),
            ('mixed_device', valid, torch.empty(4), ValueError, 'GPU'),
            ('dtype', valid.half(), valid.half(), TypeError, 'FP32'),
            ('dimension', valid.reshape(2, 2), valid.reshape(2, 2), ValueError, '一维'),
            ('shape', valid, torch.empty(3, device='cuda'), ValueError, '形状'),
            ('noncontiguous', torch.empty(8, device='cuda')[::2], valid, ValueError, '连续'),
        ]:
            check(f'vector_add_reject_{name}', x, lambda: triton_add(x, y),
                  expected_error=error, message=message, other=y)

    if args.suite in ('all', 'softmax'):
        # 同一组形状分别验证 FP32、FP16，不能只在接口声明支持两种类型。
        for dtype in [torch.float32, torch.float16]:
            for shape in [(1, 1), (2, 3), (3, 5), (7, 255), (7, 256), (7, 257),
                          (32, 1000), (16, 4096), (1024, 257), (0, 5)]:
                x = torch.randn(shape, device='cuda', dtype=dtype)
                check('softmax_random', x, lambda: triton_softmax(x), lambda: torch.softmax(x, 1))
            # 随机数据之外，用有针对性的输入检查稳定化和 padding。
            for name, values in [
                ('large_shift', [[1, 2, 3], [1001, 1002, 1003]]),
                ('negative', [[-5, -4, -3, -2, -1]]),
                ('equal', [[7] * 5] * 2),
                ('extreme_finite', [[-10000, 0, 10000], [10000, 10000, -10000]]),
            ]:
                x = torch.tensor(values, device='cuda', dtype=dtype)
                check(f'softmax_{name}', x, lambda: triton_softmax(x), lambda: torch.softmax(x, 1))
        # 每个元组依次为：名称、非法输入、预期异常类型、异常信息关键词。
        # 无须给非法输入 reference，因为目标是确认它被拒绝。
        invalid = [
            ('zero_width', torch.empty((2, 0), device='cuda'), ValueError, '行宽'),
            ('too_wide', torch.empty((2, 4097), device='cuda'), ValueError, '行宽'),
            ('noncontiguous', torch.empty((3, 5), device='cuda').T, ValueError, '连续'),
            ('one_dimensional', torch.empty(5, device='cuda'), ValueError, '二维'),
            ('cpu', torch.empty((3, 5)), ValueError, 'GPU'),
            ('float64', torch.empty((3, 5), device='cuda', dtype=torch.float64), TypeError, 'FP32'),
        ]
        for name, x, error, message in invalid:
            check(f'softmax_reject_{name}', x, lambda: triton_softmax(x),
                  expected_error=error, message=message)
    # ⑥ 汇总。布尔值在 sum 中按 True=1、False=0 计数。
    failed = sum(r['status'] == 'FAIL' for r in records)
    summary = dict(total=len(records), passed=len(records) - failed, failed=failed)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), '\nEvidence:', out)
    return 1 if failed else 0  # 进程约定：0 表示成功，非 0 表示未成功。


if __name__ == '__main__':
    # 将 main 的返回值变成 shell 可读取的进程退出码。
    raise SystemExit(main())
