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

system_prompt = "We would like to request your feedback on the performance of the AI assistant in response to the instruction and the given input displayed below."
instance_prompts = "\n\nInstruction: {instruction}\n\nInput:{input}\n\nRespone: {output}\n\n"
instance_prompts_oi = "Instruction: {instruction}\n\nRespone: {output}\n\n"
user_prompt = (
    "Please rate the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level." 
    "Please first output a single line containing the value indicating the scores, preceded by 'Score: '. Then, in the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias, which is marked as 'Explanations'. "
)
user_prompt_oi = (
    "Please rate the response to the instruction. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level." 
    " Please first output a single line containing the value indicating the scores, preceded by 'Score: '. Then, in the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias, which is marked as 'Explanations'. "
)
user_prompt_oi_factuality = (
    "Please rate the response to the instruction on the response's factuality. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates more factual." 
    " Please first output a single line containing the value indicating the scores, preceded by 'Score: '. Then, in the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias, which is marked as 'Explanations'. "
)

citation_instructions = " The AI assistant's response should be high-quality and fully supported by one or more of the provided passages in the 'References:'. Each statement that requires a citation must include a citation number (e.g., [0]), which must accurately reflect the information in the cited passage. If the response does not include any citation numbers, the response is low-quality. If a citation number is included but the passage does not support the statement, the citation is inaccurate, and the response will not be considered high-quality."

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def create_prompt_with_llama3_format(prompt, system_message="We would like to request your feedback on the performance of the AI assistant in response to the instruction and the given input displayed below."):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(system_message)
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--cache_file", type=str, default=None)
    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--task_instruction", type=str, default=None)
    parser.add_argument("--sample_n", type=int, default=-1)
    parser.add_argument("--bar", type=float, default=4)
    parser.add_argument("--mode", type=str, default="oi")
    parser.add_argument("--factuality_only", default=False, action="store_true")
    
    args = parser.parse_args()

    print(args)

    # load data
    if args.input_file is not None:
        if args.input_file.endswith("jsonl"):
            data = load_jsonlines(args.input_file)
        else:
            data = json.load(open(args.input_file))
            if "data" in data:
                data = data["data"]
         
    if args.sample_n > 0 and len(data) > args.sample_n:
        data = data[:args.sample_n]
    # process data
    if args.mode == "synthetic":
        for item in data:
            if "final_passages" not in item:
                ctxs = item["ctxs"][:args.top_k]
                ctx_text = ""
                for i, ctx in enumerate(ctxs):
                    if "retrieval text" in ctx and "text" not in ctx:
                        ctx["text"] = ctx["retrieval text"]
                    if len(ctx["text"]) == 0:
                        continue
                    if "title" in ctx:
                        ctx_text += "[{}] Title: {title} Text: {text}\n".format_map(ctx)
                    else:
                        ctx_text += "[{}] {text}".format_map(ctx)
                item["final_passages"] = ctx_text
                
    # formulate prompts 
    prompts = []
    for item in data:
        if args.mode == "synthetic":
            instance_input = "References: {final_passages}\nInput: {}"
            instruction = args.task_instruction if args.task_instruction is not None else item["instruction"]
            
            
            prompt = instance_prompts.format_map({"input": instance_input, "instruction": instruction, "output": item["output"]})
            prompt += user_prompt
            
            prompts.append(create_prompt_with_llama3_format(prompt))
        elif args.mode == "oi":
            prompt = ""
            start_index = 0 if item["messages"][0]["role"] == "user" else 1
            for i in range(start_index, len(item["messages"]), 2):
                assert item["messages"][i]["role"] == "user"
                assert item["messages"][i+1]["role"] == "assistant"
                instruction = item["messages"][i]["content"]
                response = item["messages"][i+1]["content"]
                prompt += instance_prompts_oi.format_map({"instruction": instruction, "output": response})

            prompt += user_prompt_oi if not args.factuality_only else user_prompt_oi_factuality
            if "References:" in instruction:
                prompt += citation_instructions
            prompts.append(create_prompt_with_llama3_format(prompt))
    
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
    final_data = []
    
    print(outputs[:10])
    for instance, pred in zip(data, outputs):
        print(pred)
        if "Score: " not in pred:
            instance["score"] = -1
            continue
        score = pred.split("Score: ")[1].split("\n")[0]
        try:
            
            instance["score"] = float(score)
            
        except:
            instance["score"] = -1
            continue
        instance["filter_preds"] =  pred
        print(instance["score"])
        if float(score) >= args.bar:
            print("data is added")
            final_data.append(instance)
            
    print("data distributions: {}".format(Counter([item["score"] for item in data])))
    
    with open(args.output_file, "w") as outfile:
        json.dump({"data": final_data}, outfile)
       
    with open(args.output_file + "_orig", "w") as outfile:
        json.dump({"data": data}, outfile)
if __name__ == '__main__':
    main()