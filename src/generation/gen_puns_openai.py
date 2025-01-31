# import google.generativeai as genai

# genai.configure(api_key="AIzaSyCgyl1NrBOGn_zNfBxWbpoEHCxCc5zGjTg")
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("Explain how AI works")
# print(response.text)

# from google import genai
# from google.genai import types

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
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=10, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "Sampling temperature parameter"})
    n_shots: Optional[str] = field(default="5", metadata={"help": "Number of shots to use for each prompts."})
    random_shots: Optional[bool] = field(default=True, metadata={"help": "Whether to use random or fixed example shots."})


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
            max_tokens=512,
            top_p=args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            seed=None
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

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/generation/{args.model_name}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/generation/{args.model_name}/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    
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

    # Generation Config

    SHOTS = [
    "What do you call a gene that works everywhere? Generalizable.",
    "What do you call a dog that is incontrovertibly true? Dogma.",
    "What do you call a man that does nails? Manicure.",
    "What do you call a rat that is obsessed with stats? Ratio.",
    "What do you call a star that leads the way? Starter.",
    "What do you call a fan that plays an instrument? Fanfare.",
    "What do you call a cat that is clear and obvious? Categorical",
    "What do you call a port that is part of a whole? Portion.",
    "What do you call a bowl that throws balls? Bowler.",
    "What do you call a trip that multiplies by three? Triple."
    ]

    shots = args.n_shots.split(",")
    print(shots)
    
    for n_shot in shots: 
        n_shot = int(n_shot)

        for k in tqdm(range(args.n_sampling)):
            if n_shot == 0:
                prompt = """Create an English pun using the format "What do you call a X that Y? XZ".
    
Follow these guidelines:
- Select a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a new legitimate word XZ (the answer of the question).
- Use the actual definition of the word XZ to effectively replace Y, providing the punchline.

Return as output ONLY the pun, prefixed by "pun:"."""

            else:

                if args.random_shots:
                    shots_selected = random.sample(SHOTS, n_shot)
                    shots_selected = "\n".join(shots_selected)

                else:
                    shots_selected = "\n".join(SHOTS[:n_shot])
                    #shots_selected = SHOTS[:n_shot]

                prompt = f"""Create an English pun using the format "What do you call a X that Y? XZ".

Follow these guidelines:
- Select a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a new legitimate word XZ (the answer of the question).
- Use the actual definition of the word XZ to effectively replace Y, providing the punchline.

Example pun(s):
{shots_selected}

Return as output ONLY your new pun, prefixed by "pun:"."""

            logger.info(f"Selected prompt:\n{prompt}")
        
            id = f"request-{n_shot}shot-{k}"

            
            response = make_completion(args, prompt)

            completion = response.choices[0].message.content.strip()
            model = response.model

            prefix_question = completion.split("that")[0].lower()
            prefix_question = prefix_question.replace("what do you call a", "").replace("pun:", "").strip()
            answer_question = completion.split("?")[1].strip()
            is_valid = answer_question.lower().startswith(prefix_question)

            with open(f'out/generation/{args.model_name}/{output_dir}/puns-{model}.jsonl', 'a') as f:
                json.dump({"pun": completion.replace("pun:", "").strip(), "prefix": prefix_question, "answer": answer_question, "valid": "" if is_valid else is_valid, "id": id}, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Completed generation for {n_shot} shot(s).")
    
    print("Done!")