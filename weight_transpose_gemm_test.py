import torch
import torch.nn.functional as F
import time

M, K, N = 1, 4096, 11008

def benchmark(name, func, num_iters=100, warmup=20):
    # 1. Warmup (让GPU进入状态，分配缓存等)
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # 2. 正式计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        func()
    end_event.record()

    # 3. 等待所有CUDA核心完成
    torch.cuda.synchronize()

    # 计算平均耗时 (ms)
    avg_time = start_event.elapsed_time(end_event) / num_iters
    print(f"[{name}] 平均耗时: {avg_time:.4f} ms")
    print(f"FLOPS: {2 * M * K * N / (avg_time / 1000) / 1e12:.2f} TFLOPS")
    return avg_time

def run_test():
    if not torch.cuda.is_available():
        print("错误: 此脚本需要 CUDA GPU 环境。")
        return

    # === 配置参数 (模拟 LLaMA-7B/13B 级别的矩阵大小) ===
    # M: Batch Size * Seq Length (例如 32 * 512 = 16384)
    # K: In Features (Hidden Size, 例如 4096)
    # N: Out Features (Intermediate Size, 例如 11008)
    dtype = torch.float16 # 使用 FP16 以激活 Tensor Cores
    device = "cuda"

    print(f"=== Benchmark 配置 ===")
    print(f"Shape: M={M}, K={K}, N={N}")
    print(f"Dtype: {dtype}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("-" * 30)

    # === 1. 准备数据 ===
    # 输入数据 X: (M, K)
    x = torch.randn(M, K, device=device, dtype=dtype)

    # 权重 A (PyTorch 原生格式): (Out, In) -> (N, K)
    # 内存布局: Row-Major. 第k个元素和第k+1个元素是挨着的。
    w_native = torch.randn(N, K, device=device, dtype=dtype)

    # 权重 B (手动转置并连续化): (In, Out) -> (K, N)
    # 内存布局: Row-Major.
    # 警告：这里 K 是行。但在矩阵乘法 X * W 中，我们需要沿着 K 维度求和。
    # 对于 W_continuous 来说，沿着 K 走意味着要跨过 N 个元素（跨行读取）。
    w_t_c = w_native.transpose(0, 1).contiguous()

    # === 2. 定义操作 ===

    # Case 1: F.linear (利用 w_native)
    # 公式: Y = X @ W_native.T
    # 实际上由于 W_native 是 (N, K)，它的行就是 K 维连续的。
    # Tensor Core 可以直接把一行拉进来做点积。
    def op_native():
        return F.linear(x, w_native)

    # Case 2: torch.mm (利用 w_t_c)
    # 公式: Y = X @ W_t_c
    # W_t_c 是 (K, N)。矩阵乘法需要取 W 的“列”来和 X 的“行”做点积。
    # W_t_c 的“列”在物理内存中是不连续的（Stride = N）。
    def op_transposed_contiguous():
        return torch.mm(x, w_t_c)

    # Case 3: torch.matmul (利用 w_native.t()) - 对照组
    # 这应该和 Case 1 一样快，因为它利用了 View 机制，没有物理转置
    def op_view_transpose():
        return torch.matmul(x, w_native.t())

    # === 3. 运行测试 ===

    t1 = benchmark("1. PyTorch Native (F.linear)", op_native)
    t2 = benchmark("2. Transposed & Contiguous (mm)", op_transposed_contiguous)
    t3 = benchmark("3. View Transpose (matmul .t())", op_view_transpose)

    # === 4. 结果分析 ===
    print("-" * 30)
    diff = (t2 - t1) / t1 * 100
    print(f"性能差异: 'Transposed & Contiguous' 比 'Native' 慢了 {diff:.2f}%")

    if t2 > t1:
        print("\n结论: 验证成功。")
        print("原因: Case 2 导致矩阵乘法必须以 Strided 模式（跨步）读取权重矩阵的 K 维度，")
        print("      降低了 Tensor Core 的显存读取效率 (Memory Coalescing 失效)。")
    else:
        print("\n注意: 如果两者差异不大，可能是矩阵过小导致 Kernel Launch 开销占主导，")
        print("      或者当前 GPU 的 L2 Cache 极大掩盖了访存延迟。")

if __name__ == "__main__":
    run_test()
