# W10 Triton Softmax 初学者学习手册

从向量加法到按行 Softmax　整理日期 2026 年 10 月 31 日

这份手册整理了你本周实际完成的学习和实验，帮助你在离开对话后，仍然能够解释代码、重新运行实验并定位错误。重点是 Triton 的执行方式、内存地址与 mask、数值稳定性、FP16 与 FP32、正确性验证。新语法和 API 都附有解释，完整代码集中放在后半部分，避免只留下不能独立运行的修改片段。

你已经跑通向量加法，以及 FP32 和 FP16 的按行 Softmax。现有结果支持“在已测试输入上正确”，还不支持“比 PyTorch 快”或“已经加速模型服务”。性能测量属于 W11。

## 1 学习范围与复习路线

建议第一次按第 2 至第 7 章复习概念，再保存第 8、9 章的完整脚本。第 10 章记录你的实测结果，第 11 章用于排错，第 12 章用于验收。

| 学习任务 | 已完成内容 | 回查位置 |
| --- | --- | --- |
| 理解执行模型 | program 与数据块的关系 | 第 3 章 |
| 编写向量加法 | 地址计算与读写边界 | 第 3 和第 8 章 |
| 理解 Softmax | 减最大值与负无穷补齐 | 第 4 章 |
| 编写按行 Softmax | 一个 program 负责一行 | 第 5 和第 9 章 |
| 支持低精度输入 | FP16 存储与 FP32 计算 | 第 6 章 |
| 理解内存布局 | stride 与连续张量 | 第 7 章 |
| 验证结果 | 参考对比与行和检查 | 第 9 和第 10 章 |

本手册覆盖本周实际进行的算子主线。原计划另列的算法题为 200 岛屿数量、733 图像渲染、994 腐烂的橘子，选做 207 课程表，并复做两道堆题；本次对话没有验收这些题，不将它们记作已完成。

## 2 环境与运行方式

### 2.1 本次实际环境

以下来自你在终端提供的输出，不是要求所有机器必须安装的版本组合。

| 项目 | 实际值 |
| --- | --- |
| 虚拟环境 | .venv-vllm |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu132 |
| PyTorch 构建对应 CUDA | 13.2 |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU |
| Compute capability | 8.9 |
| Triton | 3.7.1 |
| CUDA available | True |
| 实验目录 | ~/ai-infra/cuda_learning/w10 |

torch.version.cuda 是 PyTorch 构建对应的 CUDA 版本，不一定等于系统 nvcc 的版本。Compute capability 是硬件计算能力编号，也不是 CUDA 软件版本。导入成功只是第一步；你后续成功运行了 kernel，才验证了实际编译与执行链路。

### 2.2 怎样运行与重新检查

在原来已激活的虚拟环境中执行。下面的 cd 进入你的实际实验目录；如果目录不存在，先用 mkdir -p 创建。

```bash
cd ~/ai-infra/cuda_learning/w10
python vector_add.py
python softmax.py
```

需要重新核对环境时，可在终端执行：

```bash
python - <<'PY'
import sys
import torch
import triton

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
print("Triton:", triton.__version__)
PY
```

python - 表示从标准输入读取代码；<<'PY' 是 shell 的 here-document 写法，中间的内容作为输入交给 Python。结束标记 PY 要独占一行，不要把终端提示符复制进去。Triton kernel 本身建议定义在 .py 文件中；这里的临时脚本只检查环境或导入已有 kernel。

## 3 Triton 如何把任务交给 GPU

### 3.1 host 与 kernel 以及 Triton

Triton 是一套用于编写 GPU 计算程序的编程语言和编译器。 它让你用接近 Python 的写法，描述 GPU 要执行的计算，然后把这些代码编译成 GPU 能执行的程序。

普通 Python 包装函数在 CPU 侧运行，负责检查输入、分配输出、决定启动网格并提交 GPU 工作，这一侧常叫 host。kernel 是实际交给 GPU 执行的计算程序。

@triton.jit 是装饰器。它告诉 Triton 编译下面的函数，使其中的 Triton 运算能够在 GPU 上执行。它并不是让普通 Python 解释器逐条执行这些 GPU 计算。首次遇到需要编译的配置时可能产生编译开销，因此第一轮耗时不能直接代表稳定性能。

| 名称 | 是什么 | 做什么 |
| - | --- | --- |
| Python| 通用编程语言                      | 组织程序、调用函数、打印结果 |
|PyTorch |	深度学习与张量计算框架	          |创建 GPU 张量、提供参考结果
Triton	 | 编写 GPU kernel 的语言与编译器	 | 编译你写的加法、Softmax kernel
CUDA	 | NVIDIA 的 GPU 计算平台与编程体系	 | 提供在 NVIDIA GPU 上运行计算所需的底层能力
GPU	     | 执行计算的硬件	                | 真正完成数据运算
 
### 3.2 program 不是一个线程

常见 CUDA 入门代码让一个线程处理一个元素，通过 blockIdx、blockDim 和 threadIdx 计算全局下标。Triton 入门写法则描述一个 program 要处理的一组元素，编译器负责将组内工作映射到 GPU 线程。

program 可以理解为 kernel 的一个执行实例。多个 program 执行同一份代码，但通过自己的编号选择不同数据。它们不保证按编号先后执行。BLOCK_SIZE 表示本例中一个 program 处理的数据位置数量，不是线程数量。

### 3.3 用十个元素看懂索引

假设向量长度为 10，每个 program 处理 4 个位置。需要向上取整得到 3 个 program。

| program 编号 | offsets | mask |
| --- | --- | --- |
| 0 | 0 1 2 3 | True True True True |
| 1 | 4 5 6 7 | True True True True |
| 2 | 8 9 10 11 | True True False False |

```python
# 获取一维启动网格中的 program 编号，从 0 开始。
pid = tl.program_id(0)

# arange 生成左闭右开的整数序列，例如 0、1、2、3。
# offsets 是一组元素下标，不是已经读取到的数据。
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# 逐位置比较，得到一组布尔值。
mask = offsets < n_elements
```

注意两个“轴”的含义不同：tl.program_id(0) 中的 0 是启动网格的维度；tl.max(row, axis=0) 中的 0 是 program 内局部张量的维度。

### 3.4 地址与读写

x_ptr 是输入数据的指针。x_ptr + offsets 计算一组地址；tl.load 才真正读取地址中的数值。指针加 1 表示移动一个元素，FP32 元素对应 4 字节，FP16 对应 2 字节，不需要手动再乘字节数。

```python
# 有效位置读取数据，无效位置不访问内存并取 0。
x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# result 是一组结果，store 只写出 mask 为 True 的位置。
tl.store(z_ptr + offsets, result, mask=mask)
```

读和写都需要边界保护。只保护读取，仍然可能越界写入；只保护写入，仍然可能越界读取。other 的选择还必须服从算法：向量加法中的无效位置彼此独立且不写回，填 0 即可；Softmax 中这些位置会参与归约，必须仔细选择。

### 3.5 启动语法与编译时常量

```python
# cdiv 是向上取整除法，257 / 256 向上取整为 2。
# 末尾逗号让这个表达式成为单元素元组。
grid = (triton.cdiv(n_elements, block_size),)

# 中括号指定启动网格，圆括号传 kernel 参数。
add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=block_size)
```

BLOCK_SIZE: tl.constexpr 表示该参数在编译时确定，编译器据此知道 tl.arange 生成的块形状。本例使用 2 的幂作为块大小。启动 kernel 后，CPU 通常不会自动等待 GPU 完成；正确性测试中的读取结果可能触发等待，但这不等于普通 CPU 计时天然正确。

## 4 从数学到数值稳定的 Softmax

### 4.1 每行独立归一化

对一行数据，Softmax 先给每个数取指数，再除以这一行所有指数的和。用计算表达式写就是：

```text
p[i] = exp(x[i]) / sum(exp(x[j]) for j in 当前行)
```

输入形状为 M 行 N 列时，每行得到一组权重；整行输出之和在数学上为 1。向量加法的每个输出只依赖同位置的两个输入；Softmax 的每个输出还依赖整行的分母，因此需要求最大值和求和这类归约。

归约是把一组数汇总成更少的数，例如 max 把一行变成一个最大值，sum 把一行变成一个总和。

### 4.2 为什么先减最大值

直接计算 exp(1001) 会超出 FP32 的表示范围。本次实验中，输入第二行 [1001, 1002, 1003] 的直接指数结果都是 inf。inf 表示正无穷；后续 inf / inf 无法得到正常概率。

稳定写法是先找这一行最大值 m，再计算：

```text
m = max(x)
shifted[i] = x[i] - m
numerator[i] = exp(shifted[i])
denominator = sum(numerator)
p[i] = numerator[i] / denominator
```

因为 exp(x[i] - m) = exp(x[i]) * exp(-m)，分子和分母具有同一个因子 exp(-m)，在理想数学中可以约掉。对有限输入减去最大值后，指数的输入不大于 0，最大指数为 1，从而避免正指数溢出。它仍可能产生下溢或舍入，不代表没有任何浮点误差。

| 步骤 | 第一行 | 第二行 |
| --- | --- | --- |
| 输入 | [1, 2, 3] | [1001, 1002, 1003] |
| 最大值 | 3 | 1003 |
| 减最大值 | [-2, -1, 0] | [-2, -1, 0] |
| 指数约值 | [0.1353, 0.3679, 1] | [0.1353, 0.3679, 1] |
| 分母约值 | 1.5032 | 1.5032 |
| 输出约值 | [0.0900, 0.2447, 0.6652] | [0.0900, 0.2447, 0.6652] |

### 4.3 PyTorch 中的维度与广播

```python
# x 的形状是 (M, N)。沿 dim=1 归约，就是每行分别计算。
# max(dim=...) 会返回最大值和索引，.values 只取最大值。
# keepdim=True 保留被归约的维度，结果形状为 (M, 1)。
row_max = x.max(dim=1, keepdim=True).values

# 广播：每行的所有元素减去该行那个最大值。
shifted = x - row_max
numerator = torch.exp(shifted)
denominator = numerator.sum(dim=1, keepdim=True)
result = numerator / denominator
```

如果去掉 keepdim=True，最大值形状会变为 (M,)。广播通常按尾部维度对齐，它不再明确表示“每行一个值”；可能报维度错误，也可能在某些巧合形状下算出错误结果。保留 (M, 1) 能清楚表达按行广播。

### 4.4 为什么补齐位置填负无穷

本实现将行宽补到大于或等于它的最小 2 的幂。实际 5 列对应内部 8 个位置，但输入张量本身不扩容。无效位置通过 mask 跳过内存访问，并在局部计算中使用 -inf。

- 最大值归约中，-inf 不会超过有效的有限数。
- 减去有限最大值后，它仍然是 -inf。
- exp(-inf) 等于 0，因此不影响分母。

不能简单认为“最后不写出补齐位置就没问题”。这些位置已经参与分母计算，会提前影响有效输出。

| 步骤 | 正确补齐 | 错误补齐 |
| --- | --- | --- |
| 真实输入 | [0, 0, 0] | [0, 0, 0] |
| 内部四个位置 | [0, 0, 0, -inf] | [0, 0, 0, 0] |
| 指数 | [1, 1, 1, 0] | [1, 1, 1, 1] |
| 分母 | 3 | 4 |
| 写回前三个位置 | [1/3, 1/3, 1/3] | [1/4, 1/4, 1/4] |
| 有效输出之和 | 1 | 0.75 |

你实际运行得到了 1.0 和 0.75，验证了这个原因。更一般地，填入 0 后的贡献是 exp(0 - m)，只有 m 等于 0 时才是 exp(0)=1；关键是它并非算法所要求的零贡献。

## 5 按行 Softmax 的工作分配

### 5.1 program 数量与块大小分开计算

```python
n_rows, n_cols = x.shape
grid = (n_rows,)
block_size = triton.next_power_of_2(n_cols)
```

grid 决定有多少个 program；block_size 决定每个 program 的内部数据块大小。这两个数不能混淆。

| 输入形状 | program 数量 | 有效列数 | BLOCK_SIZE |
| --- | --- | --- | --- |
| (3, 5) | 3 | 5 | 8 |
| (10, 256) | 10 | 256 | 256 |
| (10, 257) | 10 | 257 | 512 |
| (32, 1000) | 32 | 1000 | 1024 |

你曾把 (10, 257) 的块大小回答成 257。正确答案是 512，因为 256 小于 257，下一档是 512。每个 program 的下标 0～256 有效，257～511 是补齐位置。512 仍然不是线程数。

### 5.2 为什么 Triton 用 axis 0

PyTorch 对整个二维输入沿 dim=1 计算；Triton kernel 已经先选出一行，row 的局部形状是 (BLOCK_SIZE,)，只有一个维度，所以 tl.max(row, axis=0) 和 tl.sum(..., axis=0) 沿这个唯一维度归约。

axis 指当前张量的维度，不是固定对应原始二维输入的某个轴。最大值和分母成为局部标量，随后与整组数据相减或相除时发生广播。

### 5.3 融合计算的意义

朴素的多步 PyTorch 表达式可能产生多个 kernel 和中间张量。这里把读取、减最大值、取指数、求和、归一化和写回描述在一个 kernel 内，具有减少中间显存读写和启动次数的机会。

但 torch.softmax 本身也是专门实现的算子，不能把它当成未优化的逐步表达式。我们的代码是否更快，取决于形状、资源使用、调度、编译结果和基准测试。不能仅凭“用了 Triton”或“融合了”保证加速。

## 6 FP16 存储与 FP32 计算

### 6.1 三种类型分别由谁决定

| 对象 | 决定位置 | 当前 FP16 路径 |
| --- | --- | --- |
| 输入 x | 创建输入时的 dtype | FP16 |
| 局部 row 与 result | row.cast(tl.float32) | FP32 |
| 输出 y | torch.empty_like(x) | FP16 |

FP32 每个元素占 4 字节，FP16 占 2 字节。FP16 存储较省，但精度和数值范围更有限。把读取后的数据转换为 FP32，可以减少后续减法、指数和求和中的精度损失；已经在 FP16 输入中丢失的信息无法恢复。

```python
# 输出在 host 侧先分配，类型与 x 一致。
y = torch.empty_like(x)

# kernel 内仅转换局部计算数据，不修改 x 或 y 的类型。
row = row.cast(tl.float32)

# store 按输出指针的元素类型转换并写入。
# y 为 FP16 时，FP32 结果在这里舍入为 FP16。
tl.store(y_ptr + offsets, result, mask=mask)
```

如果希望输出 FP32，需要显式改变输出分配：

```python
y = torch.empty(
    x.shape,
    device=x.device,
    dtype=torch.float32,
)
```

你已正确回答：这时输出为 FP32，因为 y 是 FP32。这个变体改变了接口约定；若采用它，还必须同步修改原来要求输出与输入 dtype 相同的测试。本手册完整代码保留输入输出同类型的版本。

### 6.2 误差容限怎样判断

assert_close 的数值判断可写成：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

atol 是绝对容限，rtol 是相对容限。例如参考值为 0.1，atol=1e-5、rtol=1e-3 时，允许差异为 0.00011。参考值接近 0 时，绝对容限尤其重要。默认检查还涉及形状、dtype 等属性，不能只把它理解为一个数字比较。

本次学习使用 FP32 的 rtol=1e-5、atol=1e-6；FP16 使用 rtol=1e-3、atol=1e-5。这是当前输入范围下的学习测试配置，不是所有算子的通用标准。失败后先检查最大误差的位置、输入和公式，不要只放宽容限。

max_error 衡量两种实现的最大绝对差异；sum_error 衡量每行和偏离 1 的最大程度。两者不等价。两种实现可能舍入到完全相同的 FP16 输出，此时 max_error=0，但这些输出之和仍可能不等于 1。

只检查行和也不够：交换两个概率的位置不会改变行和。因此必须保留逐元素参考对比。用 actual.float() 再计算误差和行和，可减少测量过程本身再次使用 FP16 舍入的影响。

## 7 stride 与连续输入

stride 告诉我们沿某个维度前进一格，需要在内存中跨过多少个元素。单位是元素，不是字节。普通二维连续张量中，跨一列移动 1 个元素，跨一行通常移动 n_cols 个元素。

（如第一个，从x维度走一个跨越内存五个元素，按列走跨过一个）

你实际观察到：

| 对象 | shape | stride | 是否连续 |
| --- | --- | --- | --- |
| 原始 x | (3, 5) | (5, 1) | True |
| 转置 xt | (5, 3) | (1, 5) | False |
| xt.contiguous() | (5, 3) | (3, 1) | True |

.t() 是二维转置，通常只改变形状与步长并共享底层存储。.contiguous() 在必要时复制和重排数据，使逻辑张量按连续方式存储；它保持转置后的数值和形状，不会撤销转置。若输入本来就连续，它不一定产生新拷贝。

本实现使用 row_id * n_cols + cols 计算地址，因此只支持连续输入。对于 stride=(1, 5) 的转置结果，相邻列实际相隔 5 个元素，按相隔 1 个读取会取错数据。删除断言不能解决这个问题。

下面是已做实验的可重复版本：

```python
import torch
from softmax import triton_softmax

x = torch.randn(3, 5, device="cuda")
xt = x.t()
print("原始:", x.shape, x.stride(), x.is_contiguous())
print("转置:", xt.shape, xt.stride(), xt.is_contiguous())

# 捕获预期的 AssertionError；as error 为异常对象命名。
try:
    triton_softmax(xt)
except AssertionError as error:
    print("正确拒绝:", error)
else:
    # try 没有抛出异常时才执行 else。
    raise RuntimeError("非连续输入没有被拒绝")

xc = xt.contiguous()
print("连续化:", xc.shape, xc.stride(), xc.is_contiguous())
torch.testing.assert_close(
    triton_softmax(xc), torch.softmax(xc, dim=1),
    rtol=1e-5, atol=1e-6,
)
print("连续化后通过")
```

## 8 向量加法完整代码

保存为 vector_add.py。本例限定一维连续 FP32 张量；空输入直接返回。torch.empty_like 不清零，因为每个有效输出位置都会被 kernel 覆盖。代码保留了学习时的结构，并补充详细注释。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    # 当前 program 的编号；每个实例负责不同数据段。
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 指针加下标形成地址，load 才真正取出数据。
    # 无效位置跳过显存读取，并在局部用 0 代替。
    xv = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    yv = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = xv + yv
    tl.store(z_ptr + offsets, result, mask=mask)


def triton_add(x, y):
    # assert 条件不成立时抛出 AssertionError。
    assert x.is_cuda and y.is_cuda
    assert x.device == y.device
    assert x.dtype == y.dtype == torch.float32
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()

    z = torch.empty_like(x)
    n = x.numel()  # numel 返回元素总数。
    if n == 0:
        return z

    block_size = 256
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=block_size)
    return z


def main():
    # 固定随机种子便于在同一环境中复现实验。
    torch.manual_seed(42)
    for n in [1, 10, 255, 256, 257, 1000, 4096]:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        actual = triton_add(x, y)
        expected = x + y
        torch.testing.assert_close(
            actual, expected, rtol=1e-5, atol=1e-6,
        )
        # abs 取绝对值，max 找最大值。
        # item 把单元素张量转换为 Python 数值，可能等待 GPU。
        error = (actual - expected).abs().max().item()
        programs = triton.cdiv(n, 256)
        # :5d 表示整数占 5 位；:.3e 表示科学计数法保留 3 位小数。
        print(f"n={n:5d} | programs={programs:2d} "
              f"| max_error={error:.3e} | PASS")


# 直接运行本文件时执行 main；被 import 时不执行测试。
if __name__ == "__main__":
    main()
```

## 9 Softmax 完整代码与复现

### 9.1 支持范围

保存为 softmax.py。下面将对话中的 FP32 版本、FP16 修改和两套测试整合在一个脚本里。核心实现对应你已经跑通的版本；整合后的排版和测试组织在整理时做了语法检查，没有在你的 GPU 上重新执行。原始实测结果见第 10 章，不应当作整合脚本的新一轮测试结果。

| 条件 | 当前约定 |
| --- | --- |
| 设备 | 本次在单张 NVIDIA GPU 上验证 |
| 维度 | 二维 |
| 布局 | 连续 |
| dtype | FP32 或 FP16 |
| 行宽 | 1～4096，学习版本的约定上限 |
| 数值 | 有限值；未添加有限值扫描 |
| 输出 | 与输入相同形状、设备、dtype |
| 自动求导 | 未实现 backward，不用于训练 |
| 空行数 | 包装函数处理，但本次未实测 |

全行 -inf 会导致减最大值时出现 NaN；+inf、NaN 也不在本次支持约定内。极端有限值还可能在低精度转换或减法中产生数值问题。当前随机测试和少量构造测试不等于穷尽所有有限输入。

### 9.2 kernel 与包装函数

下面两个代码块按顺序放进同一个 softmax.py 文件，第二块紧接第一块。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    # 每个 program 处理一行，所以编号就是行号。
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 仅适用于本版本约定的二维连续输入。
    # 一行起点距整个输入起点 row_id * n_cols 个元素。
    offsets = row_id * n_cols + cols
    row = tl.load(
        x_ptr + offsets, mask=mask, other=-float("inf"),
    )

    # 局部转 FP32，不改变显存里的输入或输出类型。
    row = row.cast(tl.float32)
    # row 是一维块，因此沿 axis=0 归约。
    row_max = tl.max(row, axis=0)
    shifted = row - row_max
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # 只有有效列写回；自动转换为 y 的元素类型。
    tl.store(y_ptr + offsets, result, mask=mask)


def triton_softmax(x):
    assert x.is_cuda, "输入必须在 GPU 上"
    assert x.ndim == 2, "输入必须是二维张量"
    # 把条件放进括号跨行，避免反斜杠复制遗漏。
    assert (
        x.dtype in (torch.float32, torch.float16)
    ), "只支持 FP32 或 FP16"
    assert x.is_contiguous(), "当前版本只支持连续张量"

    n_rows, n_cols = x.shape
    assert 1 <= n_cols <= 4096, "行宽必须为 1 到 4096"
    y = torch.empty_like(x)
    if n_rows == 0:
        return y

    # next_power_of_2(257)=512，不是 257。
    block_size = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    softmax_kernel[grid](
        x, y, n_cols, BLOCK_SIZE=block_size,
    )
    return y
```

### 9.3 参考对比与测试入口

```python
def check_case(x, name):
    actual = triton_softmax(x)
    expected = torch.softmax(x, dim=1)
    assert actual.dtype == x.dtype

    # dtype 不同，允许的舍入误差也不同。
    if x.dtype == torch.float16:
        rtol, atol = 1e-3, 1e-5
        sum_atol = 1e-3
    else:
        rtol, atol = 1e-5, 1e-6
        sum_atol = 1e-5

    # 主检查：逐元素与参考结果比较。
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol,
    )
    # 辅助检查：用 FP32 测量误差和行和。
    actual32 = actual.float()
    expected32 = expected.float()
    max_error = (actual32 - expected32).abs().max().item()
    row_sums = actual32.sum(dim=1)
    sum_error = (row_sums - 1.0).abs().max().item()
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums),
        rtol=0.0, atol=sum_atol,
    )
    print(f"{name} | shape={tuple(x.shape)} | {x.dtype} "
          f"| max_error={max_error:.3e} "
          f"| sum_error={sum_error:.3e} | PASS")


def main():
    torch.manual_seed(42)
    shapes = [
        (1, 1), (2, 3), (3, 5), (7, 255),
        (7, 256), (7, 257), (32, 1000), (16, 4096),
    ]
    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        check_case(x, "FP32 random")

    x = torch.tensor(
        [[1.0, 2.0, 3.0], [1001.0, 1002.0, 1003.0]],
        device="cuda", dtype=torch.float32,
    )
    check_case(x, "large_values")
    print("两行示例:", triton_softmax(x))

    x = torch.tensor(
        [[-5.0, -4.0, -3.0, -2.0, -1.0]], device="cuda",
    )
    check_case(x, "negative_values")
    check_case(torch.full((2, 5), 7.0, device="cuda"),
               "equal_values")

    # 重新固定种子，对应独立运行的 FP16 实验起点。
    torch.manual_seed(42)
    for n_cols in [1, 3, 5, 255, 256, 257, 1000, 4096]:
        x = torch.randn(
            (7, n_cols), device="cuda", dtype=torch.float16,
        )
        check_case(x, "FP16 random")


if __name__ == "__main__":
    main()
```

复现时先运行 python softmax.py，再运行第 7 章的非连续输入实验。所有测试的打印顺序由 Python 循环决定，不代表不同 GPU program 按顺序执行。打印出来的耗时如果没有专门设计计时，也不能作为性能结果。

### 9.4 补充保存失败用例

原计划要求保存失败用例。你目前展示的数值测试都通过，因此没有现成的数值失败样本。下面是可选补充方法，尚未在本次对话中执行；可以用它替换 check_case 中的第一处 assert_close，便于以后定位失败。

```python
# 仅捕获参考对比失败；不会捕获所有编译或执行错误。
try:
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol,
    )
except AssertionError:
    # 字典保存同一次失败的输入和输出。
    # detach 去除梯度关联，cpu 将数据复制到 CPU。
    # 使用固定文件名会覆盖上一次失败；需要多份时改名。
    torch.save(
        {
            "input": x.detach().cpu(),
            "actual": actual.detach().cpu(),
            "expected": expected.detach().cpu(),
        },
        "softmax_failure.pt",
    )
    raise  # 重新抛出原异常，不能把失败打印成 PASS。
```

这一步只保存自己实验产生的数据。后续还应记录 shape、dtype、版本、随机种子和报错文本，才能区分代码缺陷、数值差异与环境问题。

## 10 你的实验结果记录

本章按你提供的终端输出整理。数值来自原始课堂实验，不是本手册生成环境中的 GPU 测试。

### 10.1 向量加法

| 元素数 | program 数量 | 最大误差 | 状态 |
| --- | --- | --- | --- |
| 1 | 1 | 0 | PASS |
| 10 | 1 | 0 | PASS |
| 255 | 1 | 0 | PASS |
| 256 | 1 | 0 | PASS |
| 257 | 2 | 0 | PASS |
| 1000 | 4 | 0 | PASS |
| 4096 | 16 | 0 | PASS |

255、256、257 分别覆盖比一个块少一个、恰好一个块、比一个块多一个的情况，有助于发现边界错误，但不能证明所有长度都正确。

### 10.2 FP32 Softmax

| 测试 | 形状 | 最大绝对误差 | 状态 |
| --- | --- | --- | --- |
| random | (1, 1) | 0.000e+00 | PASS |
| random | (2, 3) | 5.960e-08 | PASS |
| random | (3, 5) | 2.980e-08 | PASS |
| random | (7, 255) | 1.490e-08 | PASS |
| random | (7, 256) | 7.451e-09 | PASS |
| random | (7, 257) | 1.118e-08 | PASS |
| random | (32, 1000) | 3.725e-09 | PASS |
| random | (16, 4096) | 9.313e-10 | PASS |
| large_values | (2, 3) | 2.980e-08 | PASS |
| negative_values | (1, 5) | 5.960e-08 | PASS |
| equal_values | (2, 5) | 0.000e+00 | PASS |

这些测试中最大绝对误差约为 5.96e-08。两行相差 1000 的示例输出均约为 [0.0900, 0.2447, 0.6652]。

### 10.3 FP16 Softmax

所有输入均为 7 行，下面列出实际行宽。行和误差使用转换为 FP32 后的输出计算。

| 列数 | 最大绝对误差 | 最大行和误差 | 状态 |
| --- | --- | --- | --- |
| 1 | 0.000e+00 | 0.000e+00 | PASS |
| 3 | 0.000e+00 | 1.831e-04 | PASS |
| 5 | 0.000e+00 | 1.678e-04 | PASS |
| 255 | 0.000e+00 | 2.813e-05 | PASS |
| 256 | 1.526e-05 | 2.635e-05 | PASS |
| 257 | 7.629e-06 | 6.664e-05 | PASS |
| 1000 | 2.384e-07 | 1.591e-05 | PASS |
| 4096 | 1.192e-07 | 5.960e-06 | PASS |

不能仅凭这些差异判断哪种实现更接近数学真值。可能的影响包括指数近似、归约次序和舍入；当前实验没有进一步定位各项来源。

### 10.4 其他已验证现象

- 直接对 [1001, 1002, 1003] 取指数出现 inf，减最大值后得到正常结果。
- 正确负无穷补齐的三元素输出行和为 1，错误零补齐时为 0.75。
- 转置后 stride 为 (1, 5)，当前函数正确拒绝非连续输入。
- 调用 contiguous 后 stride 为 (3, 1)，数值对比通过。

未测量 kernel 耗时、带宽、端到端延迟或服务吞吐；没有接入 vLLM；没有 backward 实现。不能在报告或简历中将这些尚未完成的工作写成已验证成果。

## 11 排错与常用语法速查

### 11.1 这次遇到的 assert 换行错误

你遇到的报错行以逗号结束：assert x.dtype in (...),。逗号后缺少同一条语句中的消息表达式，因此出现 SyntaxError。可以写成一行：

```python
assert x.dtype in (torch.float32, torch.float16), "类型不支持"
```

长条件可像第 9 章一样将条件放进括号再换行。不要写 assert (condition, message)：这会构造一个非空元组，无法实现预期的条件检查。学习阶段使用 assert 清晰方便，但 Python 的优化模式可能移除断言；正式接口应使用显式 if 判断并抛出 ValueError 或 TypeError。

### 11.2 常见现象

| 现象 | 先检查什么 |
| --- | --- |
| 导入 softmax 失败 | 当前目录是否包含 softmax.py，函数名是否一致 |
| 非连续输入被拒绝 | 查看 stride；确认是否需要先 contiguous |
| 只有不规则列数出错 | mask、BLOCK_SIZE 和补齐值是否正确 |
| 行和小于 1 | 无效位置是否以非零指数参与分母 |
| 输出出现 NaN | 是否有 NaN、正无穷或全行负无穷 |
| FP16 对比失败 | 是否读取后转 FP32，输入是否相同，误差位置在哪里 |
| 第一次明显慢 | 是否包含编译和首次初始化，不能直接下性能结论 |
| 输出是 FP32 而预期 FP16 | 检查 y 的分配 dtype，而不只看 row.cast |

### 11.3 Python 与张量语法

| 写法 | 含义 |
| --- | --- |
| import triton.language as tl | 用 tl 作为模块的简短别名 |
| (n_rows,) | 单元素元组，逗号不能省略 |
| n_rows, n_cols = x.shape | 将两个形状值拆分赋给两个变量 |
| x.numel() | 张量元素总数 |
| x.ndim | 张量维度数 |
| x[:, :3] | 所有行，列下标 0 到 2 |
| x.float() | 转换为 FP32，不自动转到 CPU |
| tensor.item() | 从单元素张量取出 Python 数值 |
| torch.ones_like(x) | 创建形状、设备和类型对应的全一张量 |
| try 与 except | 尝试执行，并捕获指定异常 |
| if __name__ == "__main__" | 直接运行文件时执行，被导入时不执行该分支 |

## 12 验收与进入 W11 的条件

### 12.1 理解题与答案

1. 输入 (10, 257) 时启动多少个 program，块大小是多少？答案：10 个，块大小 512。每个 program 处理一行，行宽向上补到 2 的幂。
2. 为什么 FP16 输入、FP32 中间计算后仍输出 FP16？答案：empty_like 分配的 y 是 FP16，store 按它的类型写回。
3. 如果 y 改为显式 FP32，输出是什么？答案：FP32。你最后已经正确解释这一点。
4. 为什么 store 有 mask 还不能用 0 补齐 Softmax？答案：补齐位置已经参与求最大值和分母，写入屏蔽无法撤销其影响。
5. 为什么局部归约是 axis=0？答案：当前 row 是一维块，只有这个维度；原始二维 x 的 dim=1 不再直接适用。
6. 为什么 contiguous 不会撤销转置？答案：它保持逻辑形状和数值，只在必要时改变存储布局。
7. 为什么 PASS 不等于加速？答案：正确性测试没有提供公平、预热后的重复计时数据。

### 12.2 当前交付与尚未验证项目

核心学习已完成：能解释 program、mask、负无穷补齐和输出 dtype；能运行并对比 FP32、FP16 Softmax；能够说明连续输入限制。失败样本保存是新增的可选补充，空行数、极端值、更多随机种子和多设备情形没有在本次实验中验收。原计划中的算法题也另行完成。

### 12.3 W11 怎样接着做

下一周比较三种实现：逐步 PyTorch 表达式、torch.softmax 和自己的 Triton kernel。固定设备、输入和 dtype，排除编译与首次预热，使用能反映 GPU 完成时间的计时方式，重复测量典型耗时和波动，并覆盖小、中、大行宽与不同的行数。

先找出什么形状下有收益、什么形状下没有收益，再讨论原因。算子局部加速也不等于整个模型端到端加速；不要为了得到加速比而只挑弱基线或只展示最快的一次结果。

## 13 参考资料

代码按 Triton 官方入门教程的基本思路编写并作教学简化；使用一个 program 处理一行，未实现官方教程中的 persistent 调度与资源估算。官方 main 文档会更新，应结合本机版本和实际运行结果阅读。

- Triton Vector Addition：https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- Triton Fused Softmax：https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- Triton cast：https://triton-lang.org/main/python-api/generated/triton.language.cast.html
- Triton store：https://triton-lang.org/main/python-api/generated/triton.language.store.html
- 本次 19 周计划的 W10 与 W11 小节，以及你在对话中提供的实际终端输出。

复习时优先重新解释自己的地址公式、补齐值和 dtype 流转，再阅读教程中的高级调度部分。能够独立说清楚当前实现的正确性与限制，比记住整段代码更重要。
