# pip install google-genai
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
from datasets import load_dataset

# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gemini-2.0-flash-thinking-exp", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="data/Phunny.jsonl", metadata={"help": "HF folder or local folder data path."})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=200, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=50, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "Sampling temperature parameter"})
    n_shots: Optional[str] = field(default="5", metadata={"help": "Number of shots to use for each prompts."})
    random_shots: Optional[bool] = field(default=True, metadata={"help": "Whether to use random or fixed example shots."})
    mode: Optional[str] = field(default="cot", metadata={"help": "Modality of prompting: chain-of-thoughts or direct inference.", "choices": ["cot", "direct"]})
    gen_type: Optional[str] = field(default="driven", metadata={"help": "Modality of prompting: chain-of-thoughts or direct inference.", "choices": ["free", "driven"]})


def load_subjects(input_path):
    try:
        # Try to load from Hugging Face Hub
        dataset = load_dataset(input_path)
        dataset = dataset[args.split]
        subjects = list(set(dataset['prefix']))
        return subjects
    except Exception:
        # If loading from HF fails, check if it's a local path
        if os.path.exists(input_path):
            dataset = load_dataset("json", data_files=args.input_data)['train']
            subjects = list(set(dataset['prefix']))
            return subjects
        else:
            raise FileNotFoundError(f"Dataset not found in Hugging Face Hub or locally: {input_path}")



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
    os.makedirs(f'out/generation/{MODEL_NAME}/{args.mode}/{args.gen_type}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/generation/{MODEL_NAME}/{args.mode}/{args.gen_type}/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)
    
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

    if args.gen_type == "free":
        SHOTS = [
        "What do you call a gene that works everywhere? Generalizable.",
        "What do you call a dog that is incontrovertibly true? Dogma.",
        "What do you call a pen that is very sorry? Penitence.",
        "What do you call a rat that is obsessed with stats? Ratio.",
        "What do you call a star that leads the way? Starter.",
        "What do you call a fan that plays an instrument? Fanfare.",
        "What do you call a cat that is clear and obvious? Categorical",
        "What do you call a port that is part of a whole? Portion.",
        "What do you call a bowl that throws balls? Bowler.",
        "What do you call a trip that multiplies by three? Triple."
        ]
    else:
        SHOTS = [
    "What do you call a X='gene' that Y? XZ.\nAnswer: Y='works everywhere', XZ='generalizable'",
    "What do you call a X='dog' that Y? XZ\nAnswer: Y='is incontrovertibly true', XZ='dogma'.",
    "What do you call a X='pen' that Y? XZ.\nAnswer: Y='is very sorry', XZ='penitence'.",
    "What do you call a X='rat' that Y? XZ.\nAnswer: Y='is obsessed with stats', XZ='ratio'.",
    "What do you call a X='star' that Y? XZ.\nAnswer: Y='is served by a waiter', XZ='starter'."]    
        

    shots = args.n_shots.split(",")
    print(shots)

    if args.gen_type == "driven":
        data = load_subjects(args.input_data)
        print("NUMBER OF UNIQUE data:", len(data))
        if args.start_idx > 0:
            data = data[args.start_idx:]
    elif args.gen_type == "free":
        data = range(args.n_sampling)
    
    for n_shot in shots: 
        n_shot = int(n_shot)

        for k, subject in enumerate(tqdm(data)):
            if n_shot == 0:
                prompt = """Create an English pun using the format "What do you call a X that Y? XZ".

Follow these guidelines:
- Select a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a new legitimate word XZ (the answer of the question).
- Use the actual definition of the word XZ to effectively replace Y, providing the punchline.

Return as output ONLY the pun, prefixed by "pun:"."""

            elif args.gen_type == "free":
                if args.random_shots:
                    shots_selected = random.sample(SHOTS, n_shot)
                    shots_selected = "\n\n".join(shots_selected)

                else:
                    shots_selected = "\n\n".join(SHOTS[:n_shot])

                prompt = f"""Given a subject X, create an English pun using the format "What do you call a X that Y? XZ".
    
Follow these guidelines:

- Choose a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a real word XZ (the punchline).
- Ensure XZ’s actual definition naturally replaces Y, making the joke logical.
- Do not use compound words (e.g., dog → dogsitter, star → starlight) or derivatives of X (e.g., dog → doggy, rat → rats, pay → payment) as value of XZ.

Example pun(s):
{shots_selected}"""

            else:
                shots_selected = "\n\n".join(SHOTS[:n_shot])
                prompt = f"""Given a subject X, create an English pun using the format "What do you call a X that Y? XZ".
    
Follow these guidelines:

- Attach a suffix Z to X, forming a real word XZ (the punchline).
- Ensure XZ’s actual definition naturally replaces Y, making the joke logical.
- Do not use compound words (e.g., dog → dogsitter, star → starlight) or derivatives of X (e.g., dog → doggy, rat → rats, pay → payment) as value of XZ.

Example pun(s):
{shots_selected}

New input:
What do you call a X='{subject}' that Y? XZ."""
                
            if args.mode == "cot" and args.gen_type == "driven":
                if args.model_name != 'gemini-2.0-flash-thinking-exp':
                    prompt += """\n\nFirst, think step by step and eventually return the final values of Y and XZ.\n\nThis must be your output format:\n### rationale: {step by step reasoning}\n### answer: Y='...', XZ='...'\n\nDon't add further infos."""
                else:
                    prompt += """\n\nAnswer by returning the final values of Y and XZ, prefixed by "### answer:"."""

            elif args.mode == "direct" and args.gen_type == "driven":
                prompt += """\n\nReturn as output ONLY the final values of Y and XZ, prefixed by "Answer:"."""
            
            elif args.mode == "cot" and args.gen_type == "free":
                if args.model_name != 'gemini-2.0-flash-thinking-exp':
                    prompt += """\n\nFirst, think step by step and eventually return the new pun.\n\nThis must be your output format:\n### rationale: {step by step reasoning}\n### pun: {your new pun}.\n\nDon't add further infos."""
                else:
                    prompt += """\n\nAnswer by returning the new pun, prefixed by "### pun:"."""

            elif args.mode == "direct" and args.gen_type == "free":
                prompt += """\n\nReturn as output ONLY the new pun, prefixed by "### pun:" allowing for its easy extraction."""

            logger.info(f"Selected prompt:\n{prompt}")
        
            id = f"request-{n_shot}shot-{k}" if args.gen_type == "free" else f"request-{n_shot}shot-{k}-{subject}"

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
            else: 
                response = model.generate_content(prompt)
        
            
            if response and args.gen_type == "driven":
                y = ""
                xz= ""
                pun = ""
                valid = False
                start_index =  response.text.lower().rfind("### answer:")
                if start_index >= 0:
                    answer = response.text.lower()[start_index+len('### answer:'):].strip()
                    y_xz = answer.split(",")
                    y = y_xz[0].lower().replace('y=', '').replace("'", '') if 'y=' in y_xz[0].lower() or 'y =' in y_xz[0].lower().replace('xz=', '').replace("'", '').replace(".", "").strip() else y_xz[1]
                    xz = y_xz[1].lower().replace('xz=', '').replace("'", '').replace(".", "").strip() if 'xz=' in y_xz[1].lower() or 'xz =' in y_xz[1].lower().replace('y=', '').replace("'", '') else y_xz[0]
                    pun = f"What do you call a {subject} that {y}? {xz.strip()}"
                    is_valid = xz.startswith(subject.lower())

                with open(f'out/generation/{MODEL_NAME}/{args.mode}/{args.gen_type}/{output_dir}/puns-{MODEL_NAME}.jsonl', "a") as f:
                    json.dump({"pun": pun.strip(), "definition": y.strip(), "answer": xz.replace(".","").strip(), "valid" : "" if is_valid else False, "id": id}, f, ensure_ascii=False)
                    f.write("\n")
                
            elif response and args.gen_type == "free":
                pun = response.text.lower().split("### pun:")[1].strip()
                cot = response.text.lower().split("### pun:")[0].strip()
                prefix_question = pun.split("that")[0]
                prefix_question = prefix_question.replace("what do you call an", "what do you call a")
                prefix_question = prefix_question.replace("what do you call a", "").replace("### pun:", "").strip()
                answer_question = pun.split("?")[1].replace(".", "").strip()
                is_valid = answer_question.lower().startswith(prefix_question)

                with open(f'out/generation/{MODEL_NAME}/{args.mode}/{args.gen_type}/{output_dir}/puns-{MODEL_NAME}.jsonl', 'a') as f:
                    json.dump({"pun": pun, "prefix": prefix_question, "answer": answer_question, "valid": "" if is_valid else is_valid, "cot": cot, "id": id}, f, ensure_ascii=False)
                    f.write("\n")

            if MODEL_NAME in ["gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp"]:
                time.sleep(6)

        logger.info(f"Completed generation for {n_shot} shot(s).")
    
    print("Done!")