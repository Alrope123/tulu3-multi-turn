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
    data = data[:args.n]

    with open(os.path.join(args.output_dir, f"{args.name}-{args.n}.jsonl"), 'w') as f:
        for dp in data:
            f.write(json.dumps(dp))
            f.write('\n')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--name", type=str, default="data")
    parser.add_argument("--n", type=int, default=50000)

    args = parser.parse_args()
    main(args)