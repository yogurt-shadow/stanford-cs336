import pynvml

def get_gpu_memory():
    """
    返回一个列表，每个元素是字典:
    {
        'index': GPU 索引,
        'name': GPU 名称,
        'total': 总显存 MiB,
        'used': 已用显存 MiB,
        'free': 剩余显存 MiB
    }
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_list = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)  # 直接是 str
        gpu_list.append({
            'index': i,
            'name': name,
            'total': mem_info.total // 1024**2,
            'used': mem_info.used // 1024**2,
            'free': mem_info.free // 1024**2
        })
    pynvml.nvmlShutdown()
    return gpu_list

# 使用示例
if __name__ == "__main__":
    gpus = get_gpu_memory()
    for gpu in gpus:
        print(f"GPU {gpu['index']} ({gpu['name']}): {gpu['free']} MiB free / {gpu['total']} MiB total")
