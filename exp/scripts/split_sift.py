import numpy as np
import os
from vecs_util import *

base_path = "/root/datasets/sift/10k/sift_base.fvecs"
query_path = "/root/datasets/sift/10k/sift_query.fvecs"
output_dir = "/root/datasets/sift/10k/split"

os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    print("Reading base and query")
    base = fvecs_read(base_path)
    query = fvecs_read(query_path)

    n = base.shape[0]
    print("base total:", n)

    split_index = int(n * 0.9)
    base_99 = base[:split_index]    # First 90% of base
    base_tail = base[split_index:]  # Last 10% of base

    print("base 99% shape:", base_99.shape)
    print("base 1% shape:", base_tail.shape)

    # Calculate ground truth for initial 90% base
    indices, distances = calculate_gt(base_99, query, 100)
    write_ivecs(indices, os.path.join(output_dir, "gt_0.ivecs"))
    print("Saved gt_step0.ivecs")

    # Sliding window sizes
    tail_chunk_size = len(base_tail) // 10  # 1% of base
    head_chunk_size = n // 100   # 1% of base

    # Initial base setup
    current_base = base_99.copy()
    
    print("Current base shape:", current_base.shape)
    print("Current base first few entries:", current_base[:1])
    print("Ground truth first few entries:", indices[:1])

    # Sliding window loop: perform 10 steps
    for i in range(10):
        # Delete the first 1% of current_base and add the next chunk from base_tail
        start = i * tail_chunk_size
        end = (i + 1) * tail_chunk_size if i < 9 else len(base_tail)
        chunk = base_tail[start:end]

        # Update current_base: delete the first 1% of base_99 and add the next chunk from base_tail
        current_base = np.vstack([current_base[head_chunk_size:], chunk])

        print(f"Step {i+1}: base shape {current_base.shape}")

        # Calculate ground truth for the updated current_base
        indices, distances = calculate_gt(current_base, query, 100)
        offset = (i + 1) * head_chunk_size
        indices = indices + offset  # Add the offset to each index
    
        out_path = os.path.join(output_dir, f"gt_{i+1}.ivecs")
        write_ivecs(indices, out_path)
        print(f"Saved {out_path}")
        
        print(f"Step {i+1}: current_base first few entries:", current_base[:1])
        print(f"Step {i+1}: ground truth first few entries:", indices[:1])

