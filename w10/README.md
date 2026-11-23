# W10 · Triton Softmax 正确性实验

[返回 CUDA / Triton 算子实验总览](../README.md)

实现 Triton vector add 与按行 Softmax，对照 PyTorch 验证边界、数值稳定性和 FP32/FP16 输出。W10 的代码实验与报告已完成；本项目尚未测量性能，不声明加速比。

## 快速复现

在 **WSL Ubuntu 的 Bash 终端**执行：

```bash
cd w10  # from the repository root
python scripts/run_experiments.py
```

预期：`total=48, passed=48, failed=0`，退出码为 0。每次执行创建新的 `results/runs/<时间>/`，不覆盖历史结果。

2026-11-15 回归：[全套 48/48](results/runs/20261115_225210_984586/summary.json)，[`python -O` 下 vector add 14/14](results/runs/20261115_225215_199546/summary.json)。

```bash
python scripts/run_experiments.py --suite softmax     # 34 项
python scripts/run_experiments.py --suite vector_add  # 14 项（8 数值 + 6 拒绝）
python scripts/run_experiments.py --seed 123          # 补充种子，独立留档
```

统一使用上面的脚本入口，原先根目录三个 Python 文件已迁移。脚本自动定位项目根目录并导入 `src/`，无需安装本项目。全套与单项测试消耗随机数的顺序不同，误差不要求逐位相同。

## 文件组织

```text
w10/
├── README.md
├── requirements.txt
├── src/w10_triton/
│   ├── __init__.py
│   ├── softmax.py           # 算子与调用封装
│   └── vector_add.py
├── tests/correctness.py     # 用例、检查与证据保存
├── scripts/run_experiments.py # 唯一命令行入口
├── docs/W10巩固学习.md       # 概念、代码拆解、练习与答案
├── reports/W10实验报告.md    # 固定运行的结果与结论
└── results/
    ├── legacy/             # 用户早期文本记录
    └── runs/<时间>/         # 每次运行的独立证据
```

先阅读 [巩固学习文档](docs/W10巩固学习.md)，对照 [Softmax 源码](src/w10_triton/softmax.py)，再查 [实验报告](reports/W10实验报告.md)。测试逻辑见 [correctness.py](tests/correctness.py)。

报告仍引用 `results/runs/20261105_020646_816504/` 的实测值。历史 JSON 中的命令、文件名与哈希反映当时布局，不回填成新版；新运行记录当前相对路径和源码哈希。

目录整理后的全套回归：[`20261105_021721_544992/summary.json`](results/runs/20261105_021721_544992/summary.json)，42/42 通过。

## 支持范围

- Softmax：CUDA、二维连续张量、FP32/FP16、行宽 1～4096；零行输入返回空张量。
- 输出与输入 shape、dtype、device 一致；中间计算为 FP32。
- 错误设备、维度、布局、行宽抛出 `ValueError`，错误 dtype 抛出 `TypeError`。这些检查不依赖 Python `assert`。
- 数值实验针对有限值；NaN/Inf、梯度、非连续布局、多 GPU 和其他 dtype 未验证。不要作为支持 autograd 的算子使用。
- Vector add：CUDA、同设备同形状的一维连续 FP32 张量；使用显式异常检查输入，在 `python -O` 下仍保留输入保护；新增 6 个拒绝用例。

## 环境与可复现性

实测：Python 3.12.3、PyTorch 2.13.0+cu132、Triton 3.7.1、WSL2、RTX 4080 **Laptop GPU**（约 12 GiB）。硬件以实测为准，不沿用学习计划中桌面 4080 的假设。

优先复用已经通过验证的 `.venv`。`requirements.txt` 记录版本，不保证这些 CUDA wheel 在默认 PyPI 上可用；迁移机器时需选择能提供对应 wheel 的来源，或使用兼容环境后重新测试。此次没有在全新环境执行安装验证。每次运行的 `pip-freeze.txt` 保存已有环境的完整版本快照，包含其他项目依赖，不要求全部安装。

结果目录包含 `cases.json`（逐例指标和容限）、`summary.json`、`environment.json`（版本、种子、设备、源码 SHA256）、`nvidia-smi.txt` 和 `pip-freeze.txt`。若失败，退出码为 1，`cases.json` 保存堆栈，`failure_*.pt` 保存 CPU 输入；不要通过扩大容限掩盖失败。

## 参考资料

实现参考 Triton 官方教程的索引、边界掩码和逐行归约方法。当前实现采用一行一个 program。[Triton Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) · [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

后续 [W11](../w11/README.md) 已完成性能对比；其计时与结论以 W11 报告为准，后续性能改进集中在 W11。

