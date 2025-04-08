import os
import time
import torch
import multiprocessing

def matmul_work(_):
    a = torch.randn(1000, 1000)
    b = torch.randn(1000, 1000)
    for _ in range(50):
        _ = a @ b

def run_parallel_matmuls(n_proc):
    print(f"\nRunning {n_proc} parallel workers...")
    start = time.time()
    with multiprocessing.Pool(n_proc) as pool:
        pool.map(matmul_work, range(n_proc))
    end = time.time()
    print(f"{n_proc} workers finished in {end - start:.2f} seconds")

if __name__ == "__main__":
    print(f"System reports {os.cpu_count()} total CPUs")
    for n in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]:
        try:
            run_parallel_matmuls(n)
        except Exception as e:
            print(f"Failed with {n} processes: {e}")
