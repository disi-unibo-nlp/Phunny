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
    model_name: Optional[str] = field(default="gemini-2.0-flash-thinking-exp", metadata={"help": "model's HF directory or local path"})
    max_samples: Optional[int] = field(default=15, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    input_data: Optional[str] = field(default="data/Phunny_comprehension.jsonl", metadata={"help": "Input data file path."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=200, metadata={"help": "Top p sampling."})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default="logical", metadata={"help": "Number of shots to use for each prompts."})
    illogical_selection: Optional[str] = field(default="most_similar", metadata={"help": "Number of shots to use for each prompts."})

if __name__ == "__main__":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
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

    logger.info(args)
    MODEL_NAME =  args.model_name 

    if MODEL_NAME == "gemini-2.0-flash-exp":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
    
    elif MODEL_NAME == "gemini-2.0-flash-thinking-exp":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version':'v1alpha'})

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

    for k, item in enumerate(data): 
       
        if args.mode == "logical":
            #prompt = item['pun'] + "\n\nDid you get it? Possible answers:\n- Yes, because {your explanation}\n- No, because {your explanation}\n\nDon't add further informations."
            prompt = item['pun'] + "\n\nDid you get it? Possible answers:\n- Yes, for exactly two reasons. First, ... Second, ...\n- No, because ...\n\nDon't add further information."
            
        else:
            new_subject = item['most_similar'] if args.illogical_selection=="most_similar" else item['least_similar']
            new_pun = f"What do you call a {new_subject} that {item['definition']}? {item['answer'][0]}"
            prompt = f"{new_pun}\n\nDid you get it? Possible answers:\n- Yes, because {{your explanation}}\n- No, because {{your explanation}}\n\nDon't add further information."
        
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
        
        elif MODEL_NAME == "gemini-2.0-flash-thinking-exp":
            config = {'thinking_config': {'include_thoughts': True}}
            response = client.models.generate_content(
                model='gemini-2.0-flash-thinking-exp',
                contents=prompt,
                config=config
            )

            model_reasoning = ""
            for part in response.candidates[0].content.parts:
                if part.thought:
                    model_reasoning = part.text
                else:
                    response = part

        else: 
            response = model.generate_content(prompt)

        completion = response.text.strip()

        if args.mode == "logical":
            correct = completion.lower().replace("-", "").strip().startswith("yes")
        
        if args.mode == "illogical":
            correct = completion.lower().replace("-", "").strip().startswith("no")

        output_file = f'out/comprehension/{output_dir}/{args.model_name}_{args.mode}.jsonl'
        if args.mode == "illogical":
            output_file = output_file.replace(".jsonl", f"_{args.illogical_selection}.jsonl")
        
        with open(output_file, 'a') as f:
            out_dict = {"pun": new_pun if args.mode == "illogical" else item['pun'], "answer": completion, "correct": correct}
            if MODEL_NAME == "gemini-2.0-flash-thinking-exp" and model_reasoning:
                out_dict['cot'] = model_reasoning
            json.dump(out_dict, f, ensure_ascii=False)
            f.write("\n")

        
        time.sleep(6)
    
    logger.info(f"Completed generation for {len(data)} shot(s).")

    logger.info(f"Exporting to excel file...")
    output_file_xlsx = output_file.replace(".jsonl", ".xlsx")
    with open(output_file) as f:
        output_data = [json.loads(line) for line in f.readlines()]
    
    pd.DataFrame(output_data).to_excel(output_file_xlsx, index=False)
    
    print("Done!")