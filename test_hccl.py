import os
import time
import ray
import torch
from ray.util.collective import collective

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.device = torch.device('npu')
        torch.npu.set_device(self.device)
        print(f"[{os.getpid()}] Worker initialized on NPU: {self.device}")

    def ping(self):
        return True

    def test_hccl(self, rank, world_size, group_name):
        print(f"[{os.getpid()}] Rank {rank} starting HCCL test.")
        
        # Prepare tensor
        tensor = torch.ones(10, dtype=torch.float32, device=self.device)
        if rank == 0:
            tensor *= 5.0 # Rank 0 tensor is [5.0, ..., 5.0]
        else:
            tensor *= 2.0 # Rank 1 tensor is [2.0, ..., 2.0]
            
        print(f"[{os.getpid()}] Rank {rank} before broadcast: {tensor}")
        
        # Synchronize NPU stream before collective
        torch.npu.synchronize()
        
        try:
            # Broadcast from rank 0
            collective.broadcast(tensor, src_rank=0, group_name=group_name)
            
            # Synchronize NPU stream after collective
            torch.npu.synchronize()
            print(f"[{os.getpid()}] Rank {rank} after broadcast: {tensor}")
            
            # Verify result
            is_success = torch.all(tensor == 5.0).item()
            return is_success
        except Exception as e:
            print(f"[{os.getpid()}] Rank {rank} HCCL exception: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    
    num_workers = 2
    print(f"Creating {num_workers} actors...")
    workers = [Worker.remote() for i in range(num_workers)]
    ray.get([w.ping.remote() for w in workers])
    
    group_name = "test_hccl_group"
    
    print("Creating HCCL collective group...")
    try:
        collective.create_collective_group(
            workers,
            num_workers,
            list(range(num_workers)),
            backend="hccl",
            group_name=group_name
        )
        print("HCCL collective group created successfully!")
    except Exception as e:
        print(f"Failed to create HCCL collective group: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running HCCL Broadcast test...")
    results = ray.get([
        w.test_hccl.remote(i, num_workers, group_name) 
        for i, w in enumerate(workers)
    ])
    
    if all(results):
        print("\n✅ HCCL Test Passed! Broadcast works perfectly.")
    else:
        print("\n❌ HCCL Test Failed! See worker logs above.")

if __name__ == "__main__":
    main()
