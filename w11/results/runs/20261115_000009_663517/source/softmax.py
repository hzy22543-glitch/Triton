import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一行。
    row_id = tl.program_id(0)

    # 当前行内的列下标。
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 本版本要求二维连续输入。
    # 第 row_id 行的起点，距离整个张量起点 row_id * n_cols 个元素。
    offsets = row_id * n_cols + cols

    # 无效列不读取内存，用负无穷参与后续计算。
    # 先读取当前行。
    # 如果输入 x 是 FP16，读取的有效元素也是 FP16。
    # mask=False 的位置不访问显存，用 -inf 代替。
    row = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=-float("inf"),
    )

    # cast 表示数值类型转换。
    # 将 program 内这一组数据转成 FP32，
    # 让后面的减法、指数、求和和除法使用 FP32 数据。
    #
    # 这不会修改显存中的输入 x，也不会恢复输入已经损失的精度。
    # 如果输入本来就是 FP32，这一步仍然保持 FP32。
    row = row.cast(tl.float32)

    # 数值稳定的 Softmax。
    # 参数1：要计算的张量，参数2：沿哪个维度求最大值。因为 row 是一维的，所以 axis=0就是取这一行的最大值。
    row_max = tl.max(row, axis=0)

    # 广播：每行的所有元素减去该行那个最大值。
    shifted = row - row_max
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # 只写入实际存在的列，写入时转换为输出 dtype（与输入一致）。
    tl.store(y_ptr + offsets, result, mask=mask)


def triton_softmax(x):
    if not x.is_cuda:
        raise ValueError("输入必须在 GPU 上")
    if x.ndim != 2:
        raise ValueError("输入必须是二维张量")
    # “in (A, B)” 表示判断是否属于这两个允许的类型之一。
    if x.dtype not in (torch.float32, torch.float16):
        raise TypeError("当前版本只支持 FP32 或 FP16")
    if not x.is_contiguous():
        raise ValueError("当前版本只支持连续张量")

    n_rows, n_cols = x.shape
    if not 1 <= n_cols <= 4096:
        raise ValueError("当前版本要求行宽为 1～4096")

    # empty_like(x) 创建与 x 形状、类型和设备相同的输出。
    y = torch.empty_like(x)

    if n_rows == 0:
        return y

    block_size = triton.next_power_of_2(n_cols)

    # 有多少行，就启动多少个 program。
    grid = (n_rows,)

    softmax_kernel[grid](
        x,
        y,
        n_cols,
        BLOCK_SIZE=block_size,
    )

    return y
