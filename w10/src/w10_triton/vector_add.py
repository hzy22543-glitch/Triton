import torch
import triton
import triton.language as tl


# 这一部分由 Triton 编译，在 GPU 上执行。
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    # BLOCK_SIZE：变量名，代表当前这个 Kernel 实例（program）一次要处理的数据块大小（比如 256 个元素）。
    # :：Python 的类型注解符号。在普通 Python 里它只起提示作用，但在 @triton.jit 装饰的函数中，Triton 编译器会严格检查它。
    # tl.constexpr：来自 triton.language，意思是“常量表达式”（Constant Expression，概念类似于 C++ 中的 constexpr）。
    # 它告诉编译器：这个参数的值在编译成 GPU 机器码时必须是固定且已知的，而不是在 GPU 运行时才动态计算的变量。
    # 表示该参数在编译时确定
):
    # 当前 program 的编号。  
    # Triton 的 program = CUDA 的 Thread Block (CTA)，0 是启动网格的维度
    pid = tl.program_id(0)

    # 当前 program 负责的一组元素下标。
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 只有小于元素总数的位置才允许访问。
    mask = offsets < n_elements

    # 从 GPU 显存中读取输入。
    # 参数1：要读取的显存地址，参数2：mask，表示哪些位置是有效的，参数3：other，表示无效位置的默认值。
    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 逐元素相加。
    result = x_values + y_values

    # 将有效位置的结果写回 GPU 显存。只把有效位置的结果写回去。读和写都需要保护，不能只在读取时加 mask
    # 参数1：要写入的显存地址，参数2：要写入的值，参数3：掩码，表示哪些位置是有效的。
    tl.store(z_ptr + offsets, result, mask=mask)


# 这一部分是普通 Python，负责检查输入、分配输出和启动 kernel。
def triton_add(x, y):
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("输入必须在 GPU 上")
    if x.device != y.device:
        raise ValueError("输入必须在同一设备")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("当前版本只支持 FP32")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("输入必须是一维张量")
    if x.shape != y.shape:
        raise ValueError("输入形状必须相同")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("输入必须连续")

    z = torch.empty_like(x)
    n_elements = x.numel()

    if n_elements == 0:
        return z

    block_size = 256

    # 向上取整，保证所有元素都有 program 处理。
    grid = (triton.cdiv(n_elements, block_size),)

    # 中括号指定启动网格，圆括号传 kernel 参数。
    add_kernel[grid](
        x,
        y,
        z,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return z
