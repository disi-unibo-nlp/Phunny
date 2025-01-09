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

# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-4o-2024-08-06", metadata={"help": "model's HF directory or local path"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=100, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "Sampling temperature parameter"})
    n_shots: Optional[str] = field(default="0,1,3,5", metadata={"help": "Number of shots to use for each prompts."})

if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/batch_api/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/batch_api/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    MODEL_NAME =  args.model_name 
    
    SHOTS = [
"What do you call a gene that works everywhere? Generalizable.",
"What do you call a dog that is incontrovertibly true? Dogma.",
"What do you call a pen that is very sorry? Penitence.",
"What do you call a rat that is obsessed with stats? Ratio.",
"What do you call a star that leads the way? Starter.",
"What do you call a fan that plays an instrument? Fanfare."]
    
    
    #######################################
    #### 1. Preparing Your Batch File #####
    #######################################

    total_promtps = 0
    shots = args.n_shots.split(",")
    print(shots)
    for n_shot in shots: 
        
        n_shot = int(n_shot)
        batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": args.model_name, "messages": [{"role": "system", "content": "You are a helpful assistant."},], "temperature": args.temperature, "max_tokens": 512}}
        
        if n_shot == 0:
            prompt = """Create an English pun using the format "What do you call a X that Y? XZ".
 
Follow these guidelines:
- Select a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a new legitimate word XZ (the answer of the question).
- Use the actual definition of the word XZ to effectively replace Y, providing the punchline.
 
Return as output ONLY the pun, prefixed by "pun:"."""

        else:
            shots_selected = "\n".join(SHOTS[:n_shot])
            prompt = f"""Create an English pun using the format "What do you call a X that Y? XZ".
 
Follow these guidelines:
- Select a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a new legitimate word XZ (the answer of the question).
- Use the actual definition of the word XZ to effectively replace Y, providing the punchline.

Example pun(s):
{shots_selected}

Return as output ONLY the pun, prefixed by "pun:"."""
            
        batch_request['body']["messages"].append({"role": "user", "content": prompt})

        for k in range(args.n_sampling):
            batch_request["custom_id"] = f"request-{n_shot}shot-{k}"
            with open(f'out/batch_api/{output_dir}/input_batch.jsonl', 'a') as f:
                json.dump(batch_request, f, ensure_ascii=False)
                f.write("\n")
                total_promtps+=1
        
    logger.info(f"UNIQUE PROMPTS: {total_promtps / args.n_sampling}")
    logger.info(f"TOTAL PROMPTS: {total_promtps}")
    
    batch_input_file = client.files.create(
    file=open(f"out/batch_api/{output_dir}/input_batch.jsonl", "rb"),
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "Running batch inference for english puns generation."
        }
    )
    logger.info(batch_obj)

    batch_id = batch_obj.id
    logger.info(f"BATCH ID: {batch_id}")

    with open(f'out/batch_api/{output_dir}/batch_id.txt', 'w') as f:
        f.write(batch_id)
    