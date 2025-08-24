import cs336_basics.model
import torch, einops, time, os
import numpy as np
import pandas as pd
import pynvml
from annotated_scaled_dot_product_attention import annotated_scaled_dot_product_attention

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

model_specifications = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },
    # "large": {
    #     "d_model": 1280,
    #     "d_ff": 5120,
    #     "num_layers": 36,
    #     "num_heads": 20
    # },
    # "xl": {
    #     "d_model": 1600,
    #     "d_ff": 6400,
    #     "num_layers": 48,
    #     "num_heads": 25
    # },
    # "2.7B": {
    #     "d_model": 2560,
    #     "d_ff": 10240,
    #     "num_layers": 32,
    #     "num_heads": 32
    # }
}

def get_free_gpus():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    mems = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.free / mem_info.total > 0.9:
            free_gpus.append(i)
            mems.append(mem_info.free // 1024**2)
    pynvml.nvmlShutdown()
    return free_gpus, mems

def generate_sample_data(batch_size: int, seq_length: int, vocab_size: int, device: str):
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def benchmark(model, input_data, target_data, use_warmup: bool = True, warmup_steps: int = 10, benchmark_steps: int = 50, test_forward_only: bool = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    # synchronize cuda
    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # forward only
    def forward_only():
        with torch.no_grad():
            outputs = model(input_data)
    
    # forward and backward
    def forward_backward():
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = loss_fn(einops.rearrange(outputs, "b s v -> (b s) v"), einops.rearrange(target_data, "b s -> (b s)"))
        loss.backward()
        optimizer.step()

    call_fn = forward_only if test_forward_only else forward_backward
    # warm up
    if use_warmup:
        for _ in range(warmup_steps):
            print("Warming up...")
            call_fn()

    # benchmark
    times = []
    for _ in range(benchmark_steps):
        print("Benchmarking...")
        synchronize()
        start = time.time()
        call_fn()
        synchronize()
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)


def test_specification():
    batch_size, vocab_size = 4, 10000
    context_length = 256
    rope_theta = 10000
    free_gpus, mems = get_free_gpus()
    if len(free_gpus) == 0:
        raise RuntimeError("No free GPU found!")
    device = "cpu"
    input_data, target_data = generate_sample_data(batch_size, context_length, vocab_size, device), generate_sample_data(batch_size, context_length, vocab_size, device)
    results = []

    for model_name, spec in model_specifications.items():
        free_gpus, mems = get_free_gpus()
        for i in range(len(free_gpus)):
            print(f"Found GPU {free_gpus[i]} with {mems[i]} MiB free memory.")

        device = f"cuda:{free_gpus[0]}" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} for model {model_name}")
        input_data, target_data = input_data.to(device), target_data.to(device)

        model = cs336_basics.model.BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=spec["d_model"],
            d_ff=spec["d_ff"],
            num_layers=spec["num_layers"],
            num_heads=spec["num_heads"],
            rope_theta=rope_theta
        ).to(device)

        for feed_only in [True, False]:
            print(f"Testing model {model_name} with forward_only={feed_only}")
            mean_time, std_time = benchmark(model, input_data, target_data, use_warmup=True, warmup_steps=5, benchmark_steps=10, test_forward_only=feed_only)
            results.append({
                "model": model_name,
                "d_model": spec["d_model"],
                "d_ff": spec["d_ff"],
                "num_layers": spec["num_layers"],
                "num_heads": spec["num_heads"],
                "function": "forward_only" if feed_only else "forward_backward",
                "mean_time": f"{mean_time:.4f} s",
                "std_time": f"{std_time:.4f} s"
            })
        del model
        torch.cuda.empty_cache()
    data = pd.DataFrame(results)
    with open("./cs336_systems/benchmark_results2.md", "w") as f:
        f.write(data.to_markdown(index=False))

if __name__ == "__main__":
    test_specification()