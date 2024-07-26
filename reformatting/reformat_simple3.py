import jsonlines
import argparse
import random
from tqdm import tqdm
import nltk
import json
import re
import vllm
import torch
import os
import pandas as pd
from collections import Counter
from datasets import load_dataset

break_prompt = "Partition the following text enclosed by <div> into an optional task instruction and a list of conversations, except the last conversation only contains a query. Enclose the task instruction with <span> if it exists, enclose each conversation with <section>. Do not add or delete anything from the original text."

DEMOS = [
    {
        "input": "Generate a context and a hypothesis.\n\nAnswer: Context: Increased liabilities will add a little to the cost of marine insurance but commercial vessels insured in mutual protection and indemnity associations will probably see no substantive increase in insurance rates because coverage already provided by mutual associations is unlimited.\n\nHypothesis: Increased liabilities will up the cost of marine insurance a little.\n\n\nGenerate a context and a hypothesis.\n\nAnswer: Context: The Bird's Nest<br>The couple planted a tree in their backyard. A bird immediately moved in. He made an intricate nest. Soon, there were a few eggs! The bird took care of the eggs until they hatched.\n\nHypothesis: the bird is female\n\n\nGenerate a context and a hypothesis.\n\nAnswer: Context: Safety<br>Anna and her boyfriend were fooling around. Then Anna saw that they were out of condoms. Her boyfriend told her it didn't matter, but Anna remained steadfast. Later, she thought the incident over. She was proud of herself for doing the right thing.\n\nHypothesis: : Anna like he boyfiend thought it didn't matter that they were out of condoms\n\n\nGenerate a context and a hypothesis.\n\nAnswer:",
        "outputs": {
            "instuction": None,
            "examples": [
                "Generate a context and a hypothesis.\n\nAnswer: Context: Increased liabilities will add a little to the cost of marine insurance but commercial vessels insured in mutual protection and indemnity associations will probably see no substantive increase in insurance rates because coverage already provided by mutual associations is unlimited.\n\nHypothesis: Increased liabilities will up the cost of marine insurance a little.\n\n\n",
                "Generate a context and a hypothesis.\n\nAnswer: Context: The Bird's Nest<br>The couple planted a tree in their backyard. A bird immediately moved in. He made an intricate nest. Soon, there were a few eggs! The bird took care of the eggs until they hatched.\n\nHypothesis: the bird is female\n\n\n",
                "Generate a context and a hypothesis.\n\nAnswer: Context: Safety<br>Anna and her boyfriend were fooling around. Then Anna saw that they were out of condoms. Her boyfriend told her it didn't matter, but Anna remained steadfast. Later, she thought the incident over. She was proud of herself for doing the right thing.\n\nHypothesis: : Anna like he boyfiend thought it didn't matter that they were out of condoms\n\n\n",
                "Generate a context and a hypothesis.\n\nAnswer:"
            ]
        }
    },
    {
        "input": "Choose your answer. Palestinians seek to visit Arafat\n\nTop Palestinian officials are to meet the French president in the hope of visiting the ailing Yasser Arafat.\nWhich topic is this article about?\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n\nWorld\n\n\nquestion: Artest suspended for rest of season\n\nThe NBA has suspended Indiana Pacers forward Ron Artest for the remainder of the season for his part in a brawl during the final minute of Friday #39;s win at Detroit.\n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\nanswer: Sports\n\n\nquestion: Online travel store Lastminute.com LMC.L says earnings for its crucial summer quarter will come in towards the lower end of expectations, sending its stock down 5 percent in early trade.\nQ: Which is the best summary of this article?\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\nI think the answer is\nanswer: Business\n\n\nIN: Five people were dead and four still missing on Sunday as Japan began a clean up after the most powerful typhoon in a decade hit the Tokyo region.\n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n\nOUT: World\n\n\nThe Open Source Development Labs (OSDL), a global consortium dedicated to accelerating the adoption of Linux in the enterprise, today announced the creation of a Open Source Software Licensing amp; Legal education track at its upcoming Enterprise Linux \n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n\nCorrect title: Science/Tech\n\n\nOverture - So Much More Than Pay Per Click\\\\Overture is known for being the Leader in \"Per Per Click\" (PPC).Owned by Yahoo, Overture advertisers can reach over 80 of Internet users. If you take \\a closer look and pop the hood, you'll find a whole suite of useful tools for ...\nWhat's this about?\n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n",
        "outputs": {
            "instuction": "Choose your answer. ",
            "examples": [
                "Palestinians seek to visit Arafat\n\nTop Palestinian officials are to meet the French president in the hope of visiting the ailing Yasser Arafat.\nWhich topic is this article about?\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n\nWorld\n\n\n",
                "question: Artest suspended for rest of season\n\nThe NBA has suspended Indiana Pacers forward Ron Artest for the remainder of the season for his part in a brawl during the final minute of Friday #39;s win at Detroit.\n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\nanswer: Sports\n\n\n",
                "question: Online travel store Lastminute.com LMC.L says earnings for its crucial summer quarter will come in towards the lower end of expectations, sending its stock down 5 percent in early trade.\nQ: Which is the best summary of this article?\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\nI think the answer is\nanswer: Business\n\n\n",
                "IN: Five people were dead and four still missing on Sunday as Japan began a clean up after the most powerful typhoon in a decade hit the Tokyo region.\n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n\nOUT: World\n\n\n",
                "The Open Source Development Labs (OSDL), a global consortium dedicated to accelerating the adoption of Linux in the enterprise, today announced the creation of a Open Source Software Licensing amp; Legal education track at its upcoming Enterprise Linux \n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n\nCorrect title: Science/Tech\n\n\n",
                "Overture - So Much More Than Pay Per Click\\\\Overture is known for being the Leader in \"Per Per Click\" (PPC).Owned by Yahoo, Overture advertisers can reach over 80 of Internet users. If you take \\a closer look and pop the hood, you'll find a whole suite of useful tools for ...\nWhat's this about?\n\nOPTIONS:\n- World\n- Sports\n- Business\n- Science/Tech\n"
            ]
        }
    },
    {
        "input": "input question: To find new music when you are bored of your usual bands,\n\nOPTIONS:\n- try looking at some playlists on Spotify.\n- try listening to AM radio instead of FM.\n\noutput answer: try looking at some playlists on Spotify.\n\n[Q]: Objective: To heal chapped lips when they are developing, OPTIONS:\n- leave a wet green tea bag on your lips over night.\n- leave a salt packet on your lips at night while asleep.\n[A]: leave a wet green tea bag on your lips over night.\n\nQ: Here is a goal: brush\n\nHow would you accomplish this goal?\n\nOPTIONS:\n- can be gripped by fingernails \n- can be gripped by mittens \n\nA: can be gripped by mittens \n\nGoal:\nStore baby pacifiers.\nOPTIONS:\n- Keep in full plastic sauce containers.\n- Keep in empty plastic sauce containers.\nAnswer:\nKeep in empty plastic sauce containers.\n\nQuestion:\nGoal: To get better rest at night,\n\nWhich of the following methods is more reasonable for accomplishing this goal?\n\nOPTIONS:\n- don't use any electronics, especially your phone for an hour before bed.\n- download a sleep app on your phone and sleep with it on your pillow.\nAnswer:\ndon't use any electronics, especially your phone for an hour before bed.\n\nGoal:\nTo easily organize headphones inside your purse and avoid tangles\nOPTIONS:\n- clip the headphone coil together with a hair tie.\n- Put the headphones inside of a small pocket.\nAnswer:\n",
        "outputs": {
            "instuction": None,
            "examples": [
                "input question: To find new music when you are bored of your usual bands,\n\nOPTIONS:\n- try looking at some playlists on Spotify.\n- try listening to AM radio instead of FM.\n\noutput answer: try looking at some playlists on Spotify.\n\n",
                "[Q]: Objective: To heal chapped lips when they are developing, OPTIONS:\n- leave a wet green tea bag on your lips over night.\n- leave a salt packet on your lips at night while asleep.\n[A]: leave a wet green tea bag on your lips over night.\n\n",
                "Q: Here is a goal: brush\n\nHow would you accomplish this goal?\n\nOPTIONS:\n- can be gripped by fingernails \n- can be gripped by mittens \n\nA: can be gripped by mittens \n\n",
                "Goal:\nStore baby pacifiers.\nOPTIONS:\n- Keep in full plastic sauce containers.\n- Keep in empty plastic sauce containers.\nAnswer:\nKeep in empty plastic sauce containers.\n\n",
                "Question:\nGoal: To get better rest at night,\n\nWhich of the following methods is more reasonable for accomplishing this goal?\n\nOPTIONS:\n- don't use any electronics, especially your phone for an hour before bed.\n- download a sleep app on your phone and sleep with it on your pillow.\nAnswer:\ndon't use any electronics, especially your phone for an hour before bed.\n\n",
                "Goal:\nTo easily organize headphones inside your purse and avoid tangles\nOPTIONS:\n- clip the headphone coil together with a hair tie.\n- Put the headphones inside of a small pocket.\nAnswer:\n"
            ]
        }
    },
]

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def create_prompt_with_llama3_format(prompt, demonstrations=None, system_message="You are a helpful assistant that breaks text into required partitions."):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(system_message)
    else:
        formatted_text = "<|begin_of_text|>"
    for demonstration in demonstrations:
        formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + demonstration["input"] + "<|eot_id|>"
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n" + demonstration["output"] + "<|eot_id|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

def format_input(input):
    inp = break_prompt
    inp += f"\n\n<div>{input}</div>"
    return inp

def format_outputs(outputs):
    out = "Partitions:\n"
    if "instruction" in outputs and outputs["instruction"] is not None:
        out += f"<span>{outputs['instruction']}</span>"
    for example in outputs["examples"]:
        out += f"<section>{example}"
        out += "</section>"
    return out.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--download_dir", type=str)
    parser.add_argument("--model_name", type=str, default="/model")
    parser.add_argument("--output_dir", type=str, default="reformatting")
    parser.add_argument("--use_demos", default=False, action="store_true")
    
    args = parser.parse_args()

    print(args)

    # load data
    data = load_dataset("allenai/tulu-v2-sft-mixture", cache_dir= args.cache_dir, split='train[:10]')
    data = data.filter(lambda x: x["dataset"]=="flan_v2")
                
    # formulate prompts 
    prompts = []
    if args.use_demos:
        demostrations = []
        for demo in DEMOS:
            demostrations.append({
                "input": format_input(demo["input"]),
                "output": format_outputs(demo["outputs"])
            }) 
    else:
        demostrations = None

    for item in data:
        start_index = 0 if item["messages"][0]["role"] == "user" else 1
        for i in range(start_index, len(item["messages"]), 2):
            assert item["messages"][i]["role"] == "user"
            assert item["messages"][i+1]["role"] == "assistant"
            instruction = item["messages"][i]["content"]
            prompts.append(create_prompt_with_llama3_format(format_input(instruction), demostrations))


    # run vllm inference
    sampling_params = vllm.SamplingParams(
        temperature=0.8,  # greedy decoding
        max_tokens=2000,
        stop_token_ids=[128009]
    )
    
    
    model = vllm.LLM(
        model="/model",
        tokenizer=args.model_name,
        tokenizer_mode="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=args.download_dir,
        enforce_eager=True,
        disable_custom_all_reduce=True
    )
    
    outputs = model.generate(prompts, sampling_params)
    outputs = [it.outputs[0].text for it in outputs]
    
    # postprocess model predictions 
    print(json.dumps(prompts))
    print(json.dumps(outputs[:10]))


    json.dump([dp["messages"][0]["content"] for dp in data], open(os.path.join(args.output_dir, "prompts_original.json"), 'w'))
    json.dump(outputs, open(os.path.join(args.output_dir, "prompts_partitioned.json"), 'w'))


if __name__ == '__main__':
    main()