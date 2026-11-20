# W9 CUDA 与 Attention 性能剖析实验

所属项目：[W9～W11 总入口](../README.md)。本次规范化与回归见 [检查记录](../docs/W9-W11检查与整合记录.md)。历史性能与源码快照保持原样。

一个可复现的单卡学习项目：用 CUDA vector add 学习线程索引、边界和 Event 计时；用因果 Attention 对照普通与原地 mask，练习从 trace 提出假设，再用无 profiler 的重复测量验证。

支持范围：Linux/WSL2、NVIDIA CUDA GPU；Attention 为二维序列维度上的多头 FP32 前向，输入 `[B,H,S,D]`，不提供反向传播或服务集成。报告基准固定 `B=1,H=8,D=64`、`S=128/512/1024`。

## 先读这些

- [核心代码与性能分析巩固](docs/W9_核心代码与性能分析巩固.md)：重点概念、关键代码与参数逐项解释，含练习和参考答案。
- [完整实验报告](reports/W9_实验报告.md)：结果、证据、未完成项与限制。
- [检查反馈与修改说明](docs/检查反馈与收尾说明.md)：你已完成的部分及此次补齐的内容。
- [实测数据汇总](results/final_20261104_2/summary.md)：由原始 CSV 自动生成。
- [学习与实验指南](docs/W9_学习与实验指南.md)：先前学习步骤；代码路径已按本 README 调整。

## 项目布局

```text
src/                  CUDA 实现与共用 Attention 前向
benchmarks/           无 profiler 的 Attention 对照基准
tests/                独立参考、因果性、各形状等价性检查
scripts/              环境记录、完整运行、profile、汇总
reports/              完成的实验报告
docs/                 反馈与项目说明
results/w9_review/    用户原始日志及 trace 备份，保留失败记录
results/final_20261104/ 首次收尾运行，在 sanitizer 环境错误处中止
results/final_20261104_2/ 完整收尾运行，含数据、环境、测试、trace
archive/pre_cleanup_20261104/ 整理前源码与历史 trace
build/                本机构建输出，不纳入版本管理
Makefile              构建与最小测试入口
```

## 环境

本次验证：RTX 4080 Laptop GPU（约 12 GiB）、WSL2 Ubuntu、CUDA toolkit 12.8、Python 3.12.3、PyTorch 2.13.0+cu132。准确版本及驱动见结果目录的 `environment.json`。

需要 `nvcc`、GNU Make、Python 和支持本机 GPU 的 PyTorch。复用已有可用环境即可；不要为复现本项目盲目升级。完整环境快照 `environment.freeze.txt` 供排障，不是项目最小安装清单。核心 Python 依赖只有 PyTorch，测试使用标准库 unittest，数据汇总使用标准库。

默认编译架构 `sm_89` 对应本机；其他 GPU 使用 `make CUDA_ARCH=<实际架构>`，不要照搬硬件参数。完整运行脚本可通过 `export CUDA_ARCH=<实际架构>` 传给 Make。

## 快速复现（在 WSL Bash 中）

```bash
cd w9  # from the repository root
make test
bash scripts/run_all.sh
```

完整运行会自动创建时间戳结果目录，依次执行：记录环境 → 强制重编译 → 正确性与参数检查 → 尝试 memcheck → 无 profiler 基准 → 独立 profile → 汇总 → 源码 SHA256 留档。GPU 任务顺序运行；请关闭你不需要的后台 GPU 工作，不同时启动另一组实验。

也可指定一个**尚不存在**的目录：

```bash
bash scripts/run_all.sh results/my_reproduction
```

脚本拒绝覆盖已有目录。计算错误、测试失败及未知 sanitizer 错误会导致非零退出；已知 WDDM 调试接口初始化失败会留下 `memcheck_unavailable.txt` 后继续，但这不代表 memcheck 通过。

## 单独运行

```bash
make all
./build/vector_add_check
# N / rounds / repeats；CSV 写 stdout，正确性写 stderr
./build/vector_bench 1025 5 100
PYTHONPATH=src python -m unittest discover -s tests -v
python benchmarks/attention_bench.py --output results/my_attention.csv
python scripts/profile_attention.py --output-dir results/my_traces
# 汇总完整运行目录，而非单个文件
python scripts/summarize.py results/final_20261104_2
# 可选绘图，需要 matplotlib（本次使用 3.10.9），不重新运行 GPU 实验
python scripts/plot_results.py results/final_20261104_2
```

Attention benchmark 输出 CSV 必须不存在；profile 输出目录必须不存在。汇总脚本只重新生成 `summary.md`、`summary_console.txt`（完整脚本重定向时）与 `trace_events.csv`，不修改原始 CSV/trace。

绘图脚本从现有 CSV/JSON 生成 `figures/attention.png` 和 `figures/timeline.png`。它是可选后处理，不属于完整运行脚本的必需依赖；报告中的图片已生成并检查。

## 数据口径

- Attention：同一 S 的两版共用输入；种子 42，关闭 TF32；每组 ABBA 顺序、3 组、每位置 5 轮、每轮 100 次、每位置预热 100 次。每版每形状 30 个“轮均值”样本。
- Vector add：非零输入、逐元素校验在计时外；256 线程/block，预热 100 次、5 轮、每轮 100 次。
- CUDA Event 测量流上区间，可能含主机提交不及时造成的空闲；不是纯 kernel 时间之和，不是端到端服务延迟。
- profile 仅用于归因；普通和原地版本的 JSON 可载入 trace 查看器。`trace_events.csv` 摘录实际 GPU kernel/拷贝事件，不混加 CPU 标记与嵌套统计行。
- 有效带宽按 `3*N*sizeof(float)/时间` 计算。重复访问同一缓冲区可能命中缓存，不代表 DRAM 实测流量。

## 来源与贡献范围

本项目整理自用户的 W9 练习。用户完成整数边界检查、基本 Event 基准、Attention 两版实现与 trace 采集；本次收尾由 AI 辅助完成结构整理、浮点输出检查、共享输入对照、独立参考与因果性测试、运行与汇总脚本及报告。报告数字来自保存的实际运行，不是教程数字。

参考：[PyTorch CUDA 异步执行](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)、[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。以本机安装版本为准；不要求阅读完整手册。

这是学习项目，尚未集成 vLLM，不能据此宣称服务吞吐或延迟收益。编译产物与本地 profiler 二进制导出保留在本机。


