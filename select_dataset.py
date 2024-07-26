import os
import json
import argparse
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

def convert_dp_to_tulu_format(dataset_name, dataset):
    # set matching keywords
    prompt_keywords = ['prompt', 'instruction', 'query', "question"]
    answer_keywords = ['answer', "output", "response"]
    data_field_keywords = ["message", "conversation", "data"]
    
    label_map = {"human" : "user", "gpt": "assistant", "system": "system"}

    # matching function
    def match_key(targets, pools, dp):
        found = False
        found_key = ""
        for pool in pools:
            for target in targets:
                if target.lower() in pool.lower() and type(dp[pool]) in [list, dict]:
                    if found:
                        choice = input(f"Found multiple matched keys: <{found_key}>, <{pool}>")
                        if choice in [found_key, pool]:
                            found_key = choice
                        else:
                            continue
                    else:
                        found_key = pool
                        found = True
        return found_key if found else None
                
    # start matching
    keys = dataset[0].keys()
    prompt_key = match_key(prompt_keywords, keys, dataset[0])
    answer_key = match_key(answer_keywords, keys, dataset[0])
    data_key = match_key(data_field_keywords, keys, dataset[0])
    id_key = match_key(["id"], keys, dataset[0])
    print(prompt_key)
    print(answer_key)
    print(data_key)
    # fixed_keys = ["dataset", "id", "messages"]
    # checking matching
    output_dataset = []
    if prompt_key is not None:
        assert answer_key is not None, f"Found prompt but not answer in {keys}."
        assert data_key is None, f"Found both prompt and data in {keys}."
        print(f"Using fields <{prompt_key}> and <{answer_key}>")
        for i, dp in enumerate(tqdm(dataset)):
            output_dp = {
                "dataset": dp["dataset"] if "dataset" in dp else dataset_name,
                "id": dp["id"] if "id" in dp else (dp[id_key] if id_key is not None else f"{dataset_name}_{i}") 
            }
            output_dp["messages"]= [{"role": "user", "content": dp[prompt_key]}, {"role": "assistant", "content": dp[answer_key]}]
            # output_dp.update({k: v for k, v in dp.items() if k not in fixed_keys})
            output_dataset.append(output_dp)
    elif data_key is not None:
        assert answer_key is None, f"Found both answer and data in {keys}."
        print(f"Using field <{data_key}>")
        for i, dp in enumerate(tqdm(dataset)):
            conversations = dp[data_key]
            assert type(conversations) == list, f"Data field <{data_key}> is not a list"
            output_dp = {
                "dataset": dp["dataset"] if "dataset" in dp else dataset_name,
                "id": dp["id"] if "id" in dp else (dp[id_key] if id_key is not None else f"{dataset_name}_{i}") 
            }
            # output_dp.update({k: v for k, v in dp.items() if k not in fixed_keys})
            output_messages = []
            if type(conversations[0]) == str:
                for i in range(0, len(conversations), 2):
                    output_messages.extend([{"role": "user", "content": conversations[i]}, {"role": "assistant", "content": conversations[i+1]}])
            elif type(conversations[0]) == dict:
                for conversation in conversations:
                    if "role" in conversation and "content" in conversation:
                        output_messages.append({"role": conversation["role"], "content": conversation["content"]})
                    elif "from" in conversation and "value" in conversation:
                        output_messages.append({"role": label_map[conversation["from"]], "content": conversation["value"]})
                    elif "input" in conversation and "output" in conversation:
                        output_messages.extend([{"role": "user", "content": conversation["input"]}, {"role": "assistant", "content": conversation["output"]}])
                    else:
                        raise ValueError(f"Not supported fields in {conversation.keys()}")
                    
            output_dp["messages"] = output_messages
            output_dataset.append(output_dp)
    else:
        raise ValueError(f"Didn't find any matching fields in {keys}")
    return output_dataset

def get_multi_conversation_ratio(data):
    lengths = []
    for dp in data:
        cur_length = len([conversation for conversation in dp["messages"] if conversation["role"] != "system"])
        assert cur_length % 2 == 0, f"Conversation length is not even with {cur_length}:\n {dp["messages"]}"
        lengths.append(cur_length)
    return len([l for l in lengths if l >= 4]) / len(lengths), sum(lengths) / len(lengths)


def main(args):
    accepted_datasets = []
    rejected_datasets = []
    error_datasets = []
    for i, dataset in enumerate(args.dataset_list):
        # try:
        print(f"Processing No.{i}: {dataset}...")
        dataset_name = dataset.split('/')[-1]
        # Config path
        output_path = os.path.join(args.output_dir, f"{dataset_name}.jsonl")
        print(f"Output path: {output_path}")

        print("Loading the dataset")
        data = load_dataset(dataset, cache_dir= args.cache_dir, split='train')
        data = data.filter(lambda x: x["language"]=="English")
        # data = data.filter(lambda x: x["dataset"]=="sharegpt")
        print(len(data))
        print("Converting to Tulu Format")
        data = convert_dp_to_tulu_format(dataset_name, data) 
        print(data[0])
        print("Counting conversation length distribution")
        ratio, average = get_multi_conversation_ratio(data)

        command = input(f"Accept {dataset_name} with a multi-turn conversation ratio of {round(ratio, 2)} and average of {round(average, 2)}?\n")
        if "y" in command:
            print(f"Accepted {dataset_name}: saving to the disk.")
            accepted_datasets.append(dataset)
            with open(output_path, 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp))
                    f.write('\n')
        else:
            print(f"Rejected {dataset_name}: continuing.")
            rejected_datasets.append(dataset)
            continue            
        print("\n")
        # except Exception as e:
        #     print(f"Error occured during processing {dataset}:\n{e}\n")
        #     error_datasets.append(dataset)
        #     continue
    
    print(f"Finshed Processing {len(args.dataset_list)} datasets")
    print(f"Accepted datasets ({len(accepted_datasets)}): {accepted_datasets}")
    print(f"Rejected datasets ({len(accepted_datasets)}): {accepted_datasets}")
    print(f"Error datasets ({len(accepted_datasets)}): {accepted_datasets}")
   

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_list", type=str, nargs="+")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")

    args = parser.parse_args()
    main(args)