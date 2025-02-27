from tqdm import tqdm 
import json
import time
from datetime import datetime


from openai import OpenAI
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
from huggingface_hub import login 


# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="deepseek-chat", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="data/Phunny_comprehension.jsonl", metadata={"help": "Input data file path."})
    max_samples: Optional[int] = field(default=10, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default="illogical", metadata={"help": "Number of shots to use for each prompts."})
    illogical_selection: Optional[str] = field(default="most_similar", metadata={"help": "Number of shots to use for each prompts."})

def make_completion(args, prompt):
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages = [
                {"role": "system","content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=args.temperature,
            max_tokens=1024,
            top_p=args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            seed=42
        )
        
        return response
    except Exception as e:
        print(e)
        return ""


if __name__ == "__main__":
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/comprehension/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/comprehension/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if "deepseek" in args.model_name.lower():
        client = OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com"
        )
    else:
        client = OpenAI(
            api_key=OPENAI_KEY
        )

    MODEL_NAME = "deepseek-chat" if "deepseek" in args.model_name.lower() else args.model_name 

    logger.info(args)

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    if args.max_samples > 0:
        data = data[:args.max_samples]
    
    for k, item in enumerate(data): 
       
        if args.mode == "logical":
            prompt = item['pun'] + "\n\nDid you get it?"
            
        else:
            new_subject = item['most_similar'] if args.illogical_selection=="most_similar" else item['least_similar']
            prompt = f"What do you call a {new_subject} that {item['definition']}? {item['answer']}\n\nDid you get it?"
        
        logger.info(f"Selected prompt:\n{prompt}")
        
        response = make_completion(args, prompt)

        completion = response.choices[0].message.content.strip()
        model = response.model

        output_file = f'out/comprehension/{output_dir}/{model}_{args.mode}.jsonl'
        if args.mode == "illogical":
            output_file = output_file.replace(".jsonl", f"_{args.illogical_selection}.jsonl")

        with open(output_file, 'a') as f:
            out_dict = {"pun": item['pun'], "prompt": prompt, "answer": completion}
            json.dump(out_dict, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Completed generation for {len(data)} samples.")
    
    print("Done!")