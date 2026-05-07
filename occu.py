import torch
import time
import argparse
import sys
import threading

def occupy_and_utilize_gpu(device_id, mem_mb):
    """在指定的GPU设备上分配显存并执行计算任务"""
    try:
        # 验证设备是否可用
        if not torch.cuda.is_available():
            print("错误：CUDA 不可用，请检查PyTorch GPU版本和驱动")
            sys.exit(1)
        
        # 关键修复：确保device_id是整数，且在合法范围内
        if not isinstance(device_id, int):
            print(f"错误：GPU设备ID必须是整数，而非 {type(device_id)}")
            sys.exit(1)
        
        if device_id >= torch.cuda.device_count():
            print(f"错误：GPU ID {device_id} 超出范围，可用设备数：{torch.cuda.device_count()}")
            sys.exit(1)

        device = torch.device(f'cuda:{device_id}')
        num_elements = mem_mb * 1024 * 1024 // 4  # 每个float32元素占4字节
        print(f"在 GPU:{device_id} 上分配 {mem_mb} MB 显存...")
        
        # 分配显存
        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
        allocated_mem = torch.cuda.memory_allocated(device) / 1024 / 1024
        print(f"GPU:{device_id} 实际分配了 {allocated_mem:.2f} MB 显存")
        
        # 持续计算以提高GPU利用率
        while True:
            tensor = torch.sin(tensor)
            time.sleep(0.001)  # 避免CPU过度占用
            
    except torch.cuda.OutOfMemoryError:
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
        print(f"\nGPU:{device_id} 显存不足！尝试分配 {mem_mb} MB，总容量 {total_mem:.2f} MB")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n释放 GPU:{device_id} 资源...")
        sys.exit(0)
    except Exception as e:
        print(f"GPU:{device_id} 错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多GPU显存占用与利用率测试")
    parser.add_argument(
        '--gpu', 
        type=int, 
        nargs='+',  # 支持多个GPU ID（如 --gpu 0 1 2）
        default=list(range(8)),  # 默认使用0-7号GPU
        help='要使用的GPU设备ID（空格分隔，如0 1 2）'
    )
    parser.add_argument(
        '--mem', 
        type=int, 
        default=30000,
        help='每个GPU占用的显存大小（MB）'
    )
    
    args = parser.parse_args()
    
    # 为每个GPU启动独立线程
    threads = []
    for gpu_id in args.gpu:  # 遍历列表中的每个GPU ID（单个整数）
        thread = threading.Thread(
            target=occupy_and_utilize_gpu, 
            args=(gpu_id, args.mem)  # 传入单个整数gpu_id，而非列表
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # 延迟启动，避免同时初始化冲突
    
    for thread in threads:
        thread.join()
    