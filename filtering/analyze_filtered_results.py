import os
import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def main(args):
    random.seed(2024)

    filtered_data = json.load(open(os.path.join(args.dataset_dir, f"{args.name}.json"), 'r'))['data']
    scores = []

    for dp in filtered_data:
        scores.append(dp['score'])

    plt.hist(scores)
    plt.xlim([-1, 5])
    plt.savefig(os.path.join(args.output_dir, f"{args.name}.png"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--name", type=str, default="data")

    args = parser.parse_args()
    main(args)