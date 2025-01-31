# pip install -q -U google-generativeai

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


# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gemini-1.5-flash", metadata={"help": "model's HF directory or local path"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=1180, metadata={"help": "Index of first data sample to process."})
    input_data: Optional[str] = field(default="data/parsing/2025-01-26_22-54-21/data_parsed.jsonl", metadata={"help": "Input data file path."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=40, metadata={"help": "Top p sampling."})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
   
if __name__ == "__main__":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/contamination/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/contamination/{output_dir}/contamination.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    logger.info(args)
    MODEL_NAME =  args.model_name 

    if MODEL_NAME == "gemini-2.0-flash-exp":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=genai.GenerationConfig(
            max_output_tokens=2048,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        ))

        model_info = genai.get_model(f"models/{MODEL_NAME}")
        logger.info(model_info)


    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    if args.start_idx > 0:
        data = data[args.start_idx:]

    if args.max_samples > 0:
        data = data[:args.max_samples]
    

    PROMPT_TEMPLATE = """This is an English pun: What do you call a {SUBJECT} that {PREDICATE}? {ANSWER}. 
In this case we have the SUBJECT "{SUBJECT}", the PREDICATE "{PREDICATE}", and the ANSWER "{ANSWER}".

Your task is to determine whether the provided English pun has been contaminated by or exists within the content of a given web page.

Contamination is confirmed only if both of the following conditions are met:

- The web page contains sufficient information to reconstruct the question part of the pun, including EXPLICIT information on both the SUBJECT and the PREDICATE.
- The web page also provides explicitly the ANSWER to the pun.

If all the conditions are satisfied, explain why, otherwise return only the string "No evidence found."

Web Page Content: {content}
"""

    for k, item in enumerate(data): 
       
        subject = item['pun'].split('that')[0].lower().replace('what do you call a', '').strip()
        predicate = item['pun'].split('that')[1].split("?")[0].strip()
        answer = item['pun'].split("?")[1].replace(".","").strip()
        
        #prompt = PROMPT_TEMPLATE.replace("{pun}", item['pun']).replace("{content}", item['content'])
        prompt = PROMPT_TEMPLATE.replace("{SUBJECT}", subject).replace("{PREDICATE}", predicate).replace("{ANSWER}", answer).replace("{content}", item['content'])

        if k == 0:
            logger.info(f"Selected prompt:\n{prompt}")
        
        if MODEL_NAME == "gemini-2.0-flash-exp":
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=prompt, 
                config=types.GenerateContentConfig(
                system_instruction='You are a helpful assistant.',
                temperature=args.temperature,
                top_p=args.top_p,
                )
            )

        else: 
            response = model.generate_content(prompt)

        if response.candidates:
            completion = response.text.strip() 
        else:
            continue
       

        output_file = f'out/contamination/{output_dir}/{args.model_name}.jsonl'

        with open(output_file, 'a') as f:
            is_safe = "no evidence found" in completion.lower()
            out_dict = {"safe": is_safe, "pun": item['pun'], "href": item['href'], "contamination": completion}
            json.dump(out_dict, f, ensure_ascii=False)
            f.write("\n")

        
        time.sleep(6)
    
    logger.info(f"Completed check for {len(data)} shot(s).")
    
    print("Done!")