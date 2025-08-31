from vecs_util import *
import numpy as np

base_path = "/root/datasets/sift/100k/sift_base.fvecs"
query_path = "/root/datasets/sift/100k/sift_query.fvecs"

output_dir = "/root/datasets/sift/100k/split"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    print("Reading base and query")
    base = fvecs_read(base_path)
    query = fvecs_read(query_path)

    n = base.shape[0]
    print("base total:", n)

    split_index = int(n * 0.9)
    base_99 = base[:split_index]
    base_tail = base[split_index:]

    print("base 99% shape:", base_99.shape)
    print("base 1% shape:", base_tail.shape)

    indices, distances = calculate_gt(base_99, query, 100)
    write_ivecs(indices, os.path.join(output_dir, "gt_0.ivecs"))
    print("Saved gt_step0.ivecs")

    tail_chunk_size = len(base_tail) // 10
    head_chunk_size = n // 100

    current_base = base_99.copy()
    for i in range(10):
        start = i * tail_chunk_size
        end = (i + 1) * tail_chunk_size if i < 9 else len(base_tail)
        chunk = base_tail[start:end]
        current_base = np.vstack([current_base, chunk])[head_chunk_size:]

        print(f"Step {i+1}: base shape {current_base.shape}")

        indices, distances = calculate_gt(current_base, query, 100)
        out_path = os.path.join(output_dir, f"gt_{i+1}.ivecs")
        write_ivecs(indices, out_path)
        print(f"Saved {out_path}")