from tqdm import tqdm 
import json
import time
from datetime import datetime


# from openai import OpenAI
import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
import random
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import HfArgumentParser
from typing import Optional
from dataclasses import dataclass, field
import google.generativeai as genai

# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    judge_name: Optional[str] = field(default="gemini-1.5-flash", metadata={"help": "model's HF directory or local path"})
    model_name: Optional[str] = field(default="gpt-4o-2024-08-06", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="out/comprehension/batch_api/gpt-4o-2024-08-06/illogical_least_similar/2025-02-10_15-47-55/comprehension_illogical_least_similar.jsonl", metadata={"help": "Input data file path."})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=40, metadata={"help": "Top k sampling."})
    n_sampling: Optional[int] = field(default=12, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
    n_shots: Optional[str] = field(default="5", metadata={"help": "Number of shots to use for each prompts."})
    random_shots: Optional[bool] = field(default=True, metadata={"help": "Whether to use random or fixed example shots."})



if __name__ == "__main__":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    MODEL_NAME =  args.model_name

    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/comprehension/evaluation/{MODEL_NAME}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/comprehension/evaluation/{MODEL_NAME}/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)
    

    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        args.judge_name,
        generation_config=genai.GenerationConfig(
        max_output_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    ))

    model_info = genai.get_model(f"models/{args.judge_name}")
    logger.info(model_info)

    if "illogical" not in args.input_data.lower() and "logical" in args.input_data.lower():
        MODE = "logical"
    elif "illogical" in args.input_data.lower() and "most_similar" in args.input_data.lower():
        MODE = "illogical_most_similar"
    elif "illogical" in args.input_data.lower() and "least_similar" in args.input_data.lower():
        MODE = "illogical_least_similar"
    else:
        raise ValueError("Invalid input: 'logical' or 'illogical' must be present in input_data path.")

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    for k, item in enumerate(tqdm(data)): 

        if not (item["pun"] and item["answer"]):
            continue
        
        if MODE == "logical":
            correct = item['answer'].lower().replace("-", "").strip().startswith("yes")
        
        if "illogical" in MODE:
            correct = item['answer'].lower().replace("-", "").strip().startswith("no")

        prefix = item["pun"].lower().split("that")[0].replace("what do you call a", '').replace("what do you call an", "")
        definition =  item["pun"].lower().split("?")[0].split("that")[1].strip()
        answer = item["pun"].lower().split("?")[1].strip().replace(".",'')
        judge_explanation = ""

        if MODE == "logical" and correct:
            gold_rationale = f"""There are two reasons:

- First, '{answer}' starts with '{prefix}' or '{answer}' sounds like {prefix} or '{answer}' is a play of words with '{prefix}'.
- Second, '{answer}' means '{definition}' or '{answer}' is highly associated to '{definition}'."""

            prompt = f"""Determine whether the given explanations are equivalent. This means that the predicted answer should match both of the two reasons given by the gold answer.

Gold explanation:
{gold_rationale.strip()}

Predicted explanation:
{item['answer'].strip()}

Question: 
Are the two explanations equivalent?

Explain briefly your decision and then answer with "yes" or "no" prefixed by "Answer:".
"""
            response = model.generate_content(prompt)
            judge_explanation = response.text.lower().split("answer:")[0].strip()

            if "no" in response.text.lower().split("answer:")[1]:
                correct = False
        
        with open(f'out/comprehension/evaluation/{MODEL_NAME}/{output_dir}/comprehension_{MODE}_{MODEL_NAME}.jsonl', 'a') as f:
            json.dump({**item, 'correct': correct, "judge": judge_explanation}, f, ensure_ascii=False)
            f.write("\n")
            
    print("Conversion to excel...")
    with open(f'out/comprehension/evaluation/{MODEL_NAME}/{output_dir}/comprehension_{MODE}_{MODEL_NAME}.jsonl') as f:
        data_to_convert = [json.loads(line) for line in f.readlines()]
    
    data_valid = [el for el in data_to_convert if el['correct']]
    acc = round(len(data_valid) / len(data_to_convert) * 100, 1)
    logger.info(f"Accuracy: {acc}")
    print("Accuracy:", round(len(data_valid) / len(data_to_convert) * 100, 1))
    pd.DataFrame(data_to_convert).to_excel(f'out/comprehension/evaluation/{MODEL_NAME}/{output_dir}/comprehension_{MODE}_{MODEL_NAME}.xlsx', index=False)
    
    with open("results/comprehension.jsonl", 'a') as f:
        json.dump({"model": MODEL_NAME, "mode": MODE, "accuracy": acc}, f, ensure_ascii=False)
        f.write("\n")
    
    print("Done!")