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
from datasets import load_dataset

# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 # o3-mini-2025-01-31

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-4o-mini-2024-07-18", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="data/Phunny_cohmprension.jsonl", metadata={"help": "Input data file path."})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    n_shots: Optional[str] = field(default="5", metadata={"help": "Number of shots to use for each prompts."})
    mode: Optional[str] = field(default="illogical", metadata={"help": "Number of shots to use for each prompts."})
    illogical_selection: Optional[str] = field(default="most_similar", metadata={"help": "Number of shots to use for each prompts."})


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
    MODEL_NAME =  args.model_name 
    MODE = args.mode if args.mode == "logical" else args.mode + f"_{args.illogical_selection}"
    
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/comprehension/batch_api/{MODEL_NAME}/{MODE}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/comprehension/batch_api/{MODEL_NAME}/{MODE}/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

   

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    if args.start_idx > 0:
        data = data[args.start_idx:]
    
    if args.max_samples > 0:
        data = data[:args.max_samples]
    
    #######################################
    #### 1. Preparing Your Batch File #####
    #######################################

    total_promtps = 0
    shots = args.n_shots.split(",")
    print(shots)

    for k, item in enumerate(data):
        actual_index = k + args.start_idx
        if "o3" in MODEL_NAME.lower():
            batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL_NAME, "messages": [{"role": "system", "content": "You are a helpful assistant."},], "reasoning_effort": "high"}}
        else:
            batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": MODEL_NAME, "messages": [{"role": "system", "content": "You are a helpful assistant."},], "temperature": args.temperature, "max_tokens": 4096}}

        if MODE == "logical":
            prompt = item['pun'] + "\n\nDid you get it? Possible answers:\n- Yes, because {your explanation}\n- No, because {your explanation}\n\nDon't add further information."
        
        else:
            new_subject = item['most_similar'] if args.illogical_selection=="most_similar" else item['least_similar']
            prompt = f"What do you call a {new_subject} that {item['definition']}? {item['answer'][0]}\n\nDid you get it? Possible answers:\n- Yes, because {{your explanation}}\n- No, because {{your explanation}}\n\nDon't add further information."
        
        logger.info(f"Selected prompt:\n{prompt}")
            
        batch_request['body']["messages"].append({"role": "user", "content": prompt})
        if k == 0:
            logger.info(f"Prompt: {prompt}")

        for n_sample in range(args.n_sampling):
            batch_request["custom_id"] = f"request-id{actual_index}"
            with open(f'out/comprehension/batch_api/{MODEL_NAME}/{MODE}/{output_dir}/input_batch.jsonl', 'a') as f:
                json.dump(batch_request, f, ensure_ascii=False)
                f.write("\n")
                total_promtps+=1
        
    logger.info(f"UNIQUE PROMPTS: {total_promtps / args.n_sampling}")
    logger.info(f"TOTAL PROMPTS: {total_promtps}")
    
    batch_input_file = client.files.create(
    file=open(f"out/comprehension/batch_api/{MODEL_NAME}/{MODE}/{output_dir}/input_batch.jsonl", "rb"),
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "Running batch inference for english puns resolution."
        }
    )
    logger.info(batch_obj)

    batch_id = batch_obj.id
    logger.info(f"BATCH ID: {batch_id}")

    with open(f'out/comprehension/batch_api/{MODEL_NAME}/{MODE}/{output_dir}/batch_id.txt', 'w') as f:
        f.write(batch_id)
    