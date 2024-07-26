import os
import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm


def main(args):
    random.seed(2024)

    data = []
    with open(os.path.join(args.dataset_dir, f"{args.name}.jsonl"), 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    random.shuffle(data)
    size = len(data) // args.n + 1
    output_path = os.path.join(args.output_dir, args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
        
    print(f"Generating {args.n} batches each with size {size}.")
    for i in tqdm(range(args.n)):
        data_batch = data[i * size: (i + 1) * size]
        with open(os.path.join(output_path, f"batch_{i}.jsonl"), 'w') as f:
            for dp in data_batch:
                f.write(json.dumps(dp))
                f.write('\n')
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--name", type=str, default="data")
    parser.add_argument("--n", type=int, default=100)

    args = parser.parse_args()
    main(args)