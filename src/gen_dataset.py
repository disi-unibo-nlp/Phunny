from openai import OpenAI
import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import  HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import re
# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-4o-2024-08-06", metadata={"help": "model's HF directory or local path"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default="gen_definitions", metadata={"help": "directory where to store results."})
    #candidates_input: Optional[str] = field(default="", metadata={"help": "directory where to store results."})
    candidates_input: Optional[str] = field(default="out/dataset/batch_api/gen_words_2025-01-15_17-30-40/candidates.txt", metadata={"help": "directory where to store results."})

def get_prefix(word):
    regex = r"\*\*(.*?)\*\*"
    return re.search(regex, word).group(1)


if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    now = datetime.now()
    # Format the date and time as a string
    output_dir = "out/dataset/batch_api/" + args.mode + "_" + now.strftime("%Y-%m-%d_%H-%M-%S") if not args.candidates_input else "/".join(args.candidates_input.split("/")[:-1])
    
    if not args.candidates_input:
        os.makedirs(output_dir, exist_ok=True)
        
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{output_dir}/batch_{args.mode}.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    
    MODEL_NAME =  args.model_name 
    
    if args.mode == "gen_words":

        PROMPT = """Generate a list of 10 words where each contains a valid subword of 3 or 4 characters as a prefix. The words should not be compound words like moonlight, goldfish, blueprint, snowball, treehouse, bookstore, etc. The subword must have a real meaning and appear at the beginning of the word as a prefix.

Good examples: **dog**ma, **card**inal, **bat**tery, **cow**ard. Note that the part of the word following the prefix (e.g., "ma", "inal", "tery", "ard") should not have any independent meaning. This is a key constraint.

Return as output only a Python list of 10 words, with the subwords highlighted in **. Do not provide any additional explanation.
"""

        total_prompts = 0
   
        batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": args.model_name, "messages": [{"role": "system", "content": "You are a helpful assistant."},], "temperature": args.temperature, "max_tokens": 512}}
        
        batch_request['body']["messages"].append({"role": "user", "content": PROMPT})

        for k in range(args.n_sampling):
            batch_request["custom_id"] = f"request-{k}"
            with open(f'{output_dir}/input_batch_{args.mode}.jsonl', 'a') as f:
                json.dump(batch_request, f, ensure_ascii=False)
                f.write("\n")
                total_prompts+=1

    if args.mode == "gen_definitions":
        
        PROMPT = """EXAMPLES:
What do you call a gene that [works everywhere]? Generalizable
What do you call a dog that [is incontrovertibly true]? Dogma
What do you call a pen that [is very sorry]? Penitence
What do you call a rat that [is obsessed with stats]? Ratio
What do you call a star that [leads the way]? Starter
What do you call a fan that [plays an instrument]? Fanfare

Now replace X with a concise definition of the answer word, ensuring the pun works effectively. Keep X clear and straightforward.

Input: What do you call a {subword} that X ? {word}

Return as output only the value of X, in this format: X = "definition"
Do not provide any additional explanation."""
   

        with open(args.candidates_input) as f:
            words = f.readlines()
            words = [line.strip() for line in words]

        total_prompts = 0
        for k, word in enumerate(tqdm(words)):
            
            prefix = get_prefix(word)
            answer_word = word.replace("**", "")
            
        
            batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": args.model_name, "messages": [{"role": "system", "content": "You are a helpful assistant."},], "temperature": args.temperature, "max_tokens": 512}}
            
            batch_request['body']["messages"].append({"role": "user", "content": PROMPT.replace("{subword}",prefix).replace("{word}", answer_word)})
            
            batch_request["custom_id"] = f"request-{k}"

            with open(f'{output_dir}/input_batch_{args.mode}.jsonl', 'a') as f:
                json.dump(batch_request, f, ensure_ascii=False)
                f.write("\n")
                total_prompts+=1

    logger.info(f"UNIQUE PROMPTS: {total_prompts / args.n_sampling}")
    logger.info(f"TOTAL PROMPTS: {total_prompts}")
    

    batch_input_file = client.files.create(
    file=open(f"{output_dir}/input_batch_{args.mode}.jsonl", "rb"),
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "Running batch inference for pun dataset generation."
        }
    )
    logger.info(batch_obj)

    batch_id = batch_obj.id
    logger.info(f"BATCH ID: {batch_id}")

    with open(f'{output_dir}/batch_id_{args.mode}.txt', 'w') as f:
        f.write(batch_id)
    