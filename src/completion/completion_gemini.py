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
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gemini-2.0-flash-exp", metadata={"help": "model's HF directory or local path"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    input_data: Optional[str] = field(default="data/data_phunny.jsonl", metadata={"help": "Input data file path."})
    start_idx: Optional[int] = field(default=24, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=40, metadata={"help": "Top p sampling."})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default="cot", metadata={"help": "Input data file path.", "choices": ['direct', 'cot']})

def is_derivative(answer, gold):
    """
    Check if `answer` is a derivative of `gold` using WordNet.
    
    Args:
        answer (str): The answer word to check.
        gold (str): The gold standard word.
    
    Returns:
        bool: True if `answer` is a derivative of `gold`, False otherwise.
    """
    # Get synsets for both words
    answer_synsets = wordnet.synsets(answer)
    gold_synsets = wordnet.synsets(gold)

    # Check if either word is a derivative of the other
    for gold_syn in gold_synsets:
        for lemma in gold_syn.lemmas():
            # Check for derivationally related forms
            related_forms = lemma.derivationally_related_forms()
            if any(rel_form.name() == answer for rel_form in related_forms):
                return True

    return False

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
    os.makedirs(f'out/completion/{MODEL_NAME}/{args.mode}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/completion/{MODEL_NAME}/{args.mode}/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)
    

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

    hits = 0
    for k, item in enumerate(tqdm(data)): 
        index_end_question = item['pun'].rfind("?")
        pun = item['pun'][:index_end_question+1]
        prompt = f"""Examples:
What do you call a gene that works everywhere? 
Answer: Generalizable.

What do you call a dog that is incontrovertibly true? 
Answer: Dogma.

What do you call a man that does nails? 
Answer: Manicure.

What do you call a rat that is obsessed with stats? 
Answer: Ratio.

What do you call a star that is served by a waiter? 
Answer: Starter.

New input:
{pun}"""
        if args.mode == "direct":
            prompt += """\n\nAnswer by returning only one word as output, prefixed by "Answer:"."""
        
        if args.mode == "cot":
            prompt += """\n\nAnswer by reasoning step by step, and eventually return your final answer (only one word) prefixed by "### Answer:"."""
        
        if k == 0:
            logger.info(f"Selected prompt:\n{prompt}")

        GOLD_ANSWER = item['pun'][index_end_question+1:].strip()
        
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

        completion = response.text.strip()
       

        output_file = f'out/completion/{MODEL_NAME}/{args.mode}/{output_dir}/{args.model_name}_{args.mode}.jsonl'
        if args.mode == "direct":
            final_answer = completion.lower().replace("answer:","").replace(".","").strip()
        elif args.mode == "cot":
            final_answer = completion.lower().split("### answer:")[1].replace(".","").strip() if "### answer:" in completion.lower() else completion
        
        final_answer = final_answer.replace("-","")
        correct = final_answer.lower() == GOLD_ANSWER.lower() or is_derivative(final_answer.lower(), GOLD_ANSWER.lower()) or GOLD_ANSWER.lower() in final_answer.lower() or final_answer.lower() in GOLD_ANSWER.lower()
        out_dict = {"pun": item['pun'], "answer": final_answer, "gold": GOLD_ANSWER, "correct": correct, "completion": completion, "prompt": prompt}
            
        with open(output_file, 'a') as f:
            json.dump(out_dict, f, ensure_ascii=False)
            f.write("\n")

        if MODEL_NAME == "gemini-2.0-flash-exp":
            time.sleep(6)
        
        if correct:
            hits += 1

    logger.info(f"Completed generation for {len(data)} samples.")
    logger.info(f"Pass@1: {hits/len(data)*100}")
    
    print("Done!")