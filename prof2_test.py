import torch
import threading
import time

def worker(name, delay):
    print(f"[{name}] 准备启动 Profiler...")
    try:
        # 模拟错峰启动，制造重叠时间窗口
        time.sleep(delay)

        # 尝试启动 profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True
        ) as p:
            print(f"[{name}] Profiler 已启动! 正在运行...")
            # 模拟推理工作
            x = torch.randn(100, 100)
            torch.mm(x, x)
            time.sleep(2)

        print(f"[{name}] Profiler 正常结束.")

    except RuntimeError as e:
        print(f"\n!!! [{name}] 启动失败，捕获到报错: {e} !!!\n")
    except Exception as e:
        print(f"[{name}] 未知错误: {e}")

# 线程 A 先启动
t1 = threading.Thread(target=worker, args=("Thread-A", 0))
# 线程 B 在 A 运行期间尝试启动 (0.5s 后)
t2 = threading.Thread(target=worker, args=("Thread-B", 0.5))

t1.start()
t2.start()

t1.join()
t2.join()
