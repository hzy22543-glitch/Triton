# W11 · Softmax 性能对比实验

[返回 CUDA / Triton 算子实验总览](../README.md)

**状态：算子实验与工程交付已完成。** 比较朴素 PyTorch 表达式、torch.softmax 与自己实现的 Triton Softmax；保留正确性、原始计时、重复实验、收益/退化和来源说明。

在 RTX 4080 Laptop GPU 的连续 FP32 eager 测试中，1024×4096 相对 Torch 的三次完整复测为 **1.991 / 1.962 / 2.100 倍**。七个较轻的形状由 Torch 获胜，4096×4096 基本持平。预分配诊断显示，1024×4096 的优势在轮换八对缓冲后缩小到约 1 倍。报告包含 warp 数消融及 L2、DRAM 计数器证据。

## 阅读入口

- [正式实验报告](reports/W11实验报告.md)：结论、形状选型、warp 数消融、缓存复用和硬件计数器。
- [实验复现指南](docs/W11实验指导.md)：从环境到测量、核验、图表与排错。
- [学习文档](docs/W11学习文档.md)：融合、同步计时、统计与自测题。
- [数据核验与完整对照](reports/数据核验与对照.md)：所有运行的重算结果，包括异常轮。
- [首次试跑记录（归档）](reports/W11首次试跑记录.md)：保留最初结果，不作为当前完成状态。

## 快速复现

在 **WSL Ubuntu / Bash**，复用已有环境：

```bash
cd w11  # from the repository root
python scripts/check_correctness.py
python scripts/benchmark_softmax.py --suite minimum --dtype fp32 --note '填写实际电源模式与后台负载'
python scripts/analyze_results.py
python -m unittest discover -s tests -v
```

预期正确性 34/34；minimum 基准 12 项输出检查、36 个计时组。完整形状使用 `--suite full --min-run-time 1`，共 27 项检查、81 个计时组。计时前按形状验证；失败会保留错误并退出非零，不输出完整成功报告。

每次写入独立时间目录，已有原始结果不会覆盖。主表使用预热后的重复调用计时，分配包含在各自函数内，输入创建与校验在计时外。复用同一个输入且不主动清缓存，不是冷缓存或纯 kernel 测试。

## 已完成的验证

| 验收项 | 结果 |
|---|---|
| 本地算子正确性 | 34/34，FP32/FP16、边界、特殊有限值、空输入与拒绝用例 |
| 用户独立复跑 | 11/14 两次 minimum，全量保留 |
| 完整 FP32 复核 | A/B/R 各 9 类形状 × 3 实现 × 3 轮，分别独立进程 |
| 原始证据核验 | 8 次性能运行、52 个形状/dtype 条件、468 个计时组全部一致 |
| 数据核验测试 | 当前 7 项，覆盖原有 5 项及未知 suite/dtype 拒绝 |
| 短 trace | 两种形状中，naive/torch/Triton 每调用分别 5/1/1 个 kernel |
| 设备侧诊断 | 两次 Graph 运行，各 14 个形状、59 个计时组，另有 warp 数消融和三组整图硬件计数器 |
| 报告与图表 | 已完成，包含收益、退化、异常和不可推广的边界 |

当前项目独立保存源自 W10 的算子，不再依赖相邻 W10 目录。历史快照仍保留旧路径和版本。已删除空白报告模板，保留有证据价值的历史记录；不用删除异常数据制造漂亮结论。

## 目录职责

```text
src/w11_softmax/   算子实现，保留原计算逻辑与注释
scripts/          正确性、基准、离线分析、短 trace、设备侧诊断
tests/            数据完整性测试
docs/             学习材料与复现指南
reports/          正式报告、完整对照、图表和历史记录
results/          不改写的原始证据与源码快照
```

运行版本见 [requirements.txt](requirements.txt) 和每次 `environment.json`。环境为 Python 3.12、PyTorch 2.13.0+cu132、Triton 3.7.1；没有新建环境安装验证，CUDA wheel 来源需与实际机器兼容。画图可使用本机已有 matplotlib 3.10.9，不生成图片时无需它。

只重建现有结果和图，不重新跑 GPU：

```bash
python scripts/analyze_results.py --plots --plot-run 20261120_212155_885328
```

机制观察可单独运行 `python scripts/profile_softmax.py`。Profiler 用来观察启动结构。设备侧归因可运行 `python scripts/diagnose_softmax.py`，使用预分配缓冲和 CUDA Graph，结果见报告第 4～5 节。

## 支持范围与完成边界

CUDA 二维连续 FP32/FP16、行宽 1～4096、前向、有限值验证。零行返回空输出，不用于性能比较。NaN/Inf、非连续布局、梯度、更宽行、自动调优、persistent 调度与 vLLM 集成不在本次交付范围。

FP16 已做正确性及最低 4 类形状冒烟；未进行完整 FP16 性能扫描。笔记本电源/显示负载未完全控制，跨运行小差异不作强结论。

## 参考资料

算子沿用 W10 的按行 Softmax。计时与对照设计参考 PyTorch 和 Triton 的官方教程。[PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) · [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
