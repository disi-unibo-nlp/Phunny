from tqdm import tqdm 
import json
import time
from datetime import datetime
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from datasets import load_dataset
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
    model_name: Optional[str] = field(default="o3-mini-2025-01-31", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="data/data_phunny.jsonl", metadata={"help": "Input data file path."})
    max_samples: Optional[int] = field(default=3, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default="cot", metadata={"help": "Input data file path.", "choices": ['direct', 'cot']})
    n_shots: Optional[int] = field(default=5, metadata={"help": "Number of shots to use for inference.", "choices": [3, 5, 10]})


# def is_derivative(answer, gold):
#     """
#     Check if `answer` is a derivative of `gold` using WordNet.
    
#     Args:
#         answer (str): The answer word to check.
#         gold (str): The gold standard word.
    
#     Returns:
#         bool: True if `answer` is a derivative of `gold`, False otherwise.
#     """
#     # Get synsets for both words
#     answer_synsets = wordnet.synsets(answer)
#     gold_synsets = wordnet.synsets(gold)

#     # Check if either word is a derivative of the other
#     for gold_syn in gold_synsets:
#         for lemma in gold_syn.lemmas():
#             # Check for derivationally related forms
#             related_forms = lemma.derivationally_related_forms()
#             if any(rel_form.name() == answer for rel_form in related_forms):
#                 return True

#     return False

def load_data(input_path):
    try:
        # Try to load from Hugging Face Hub
        dataset = load_dataset(input_path)
        return dataset
    except Exception:
        # If loading from HF fails, check if it's a local path
        if os.path.exists(input_path):
            dataset = load_dataset("json", data_files=args.input_data)['train']
            print(dataset)
            return dataset
        else:
            raise FileNotFoundError(f"Dataset not found in Hugging Face Hub or locally: {input_path}")


def is_derivative(answer, gold):
    """
    Check if `answer` is a derivative of `gold` using WordNet and manual heuristics.

    Args:
        answer (str): The answer word to check.
        gold (str): The gold standard word.

    Returns:
        bool: True if `answer` is a derivative of `gold`, False otherwise.
    """
    lemmatizer = WordNetLemmatizer()
    
    # Get base forms
    answer_lemma = lemmatizer.lemmatize(answer)
    gold_lemma = lemmatizer.lemmatize(gold)

    # Get synsets for both words
    gold_synsets = wordnet.synsets(gold_lemma)
    
    # Check derivational relationships
    for gold_syn in gold_synsets:
        for lemma in gold_syn.lemmas():
            related_forms = {rel_form.name() for rel_form in lemma.derivationally_related_forms()}
            if answer_lemma in related_forms or answer in related_forms:
                return True
    
    # Heuristic: Check if gold is a participle form of a verb and answer is its agent noun
    base_gold = lemmatizer.lemmatize(gold, pos='v')  # Convert to base verb if possible
    if base_gold != gold and answer_lemma == base_gold + "er":
        return True  # Handle cases like "startler" from "startle"
    
    # Additional rule: Common suffix patterns
    derivational_patterns = [
        (gold_lemma.endswith("ing") and answer_lemma == gold_lemma[:-3] + "er"),  # startling → startler
        (gold_lemma.endswith("ed") and answer_lemma == gold_lemma[:-2] + "er"),  # startled → startler
        (gold_lemma.endswith("y") and answer_lemma == gold_lemma[:-1] + "ed"),
        (gold_lemma.endswith("ful") and answer_lemma == gold_lemma[:-3] + "able"),
        (gold_lemma.endswith("metry") and answer_lemma == gold_lemma[:-5] + "meter"),
        (gold_lemma.endswith("er") and answer_lemma == gold_lemma[:-2] + "y"),  # sticker → sticky # piercing → pierce
    ]
    
    if any(derivational_patterns):
        return True

    return False


def make_completion(args, prompt):
    try:
        # Create a chat completion using the question and context
        if "o3" in args.model_name:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages = [
                    {"role": "system","content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort="high",
                top_p=args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                seed=42
            )
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages = [
                    {"role": "system","content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=args.temperature,
                max_tokens=2048,
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

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    MODEL_NAME = "deepseek-chat" if "deepseek" in args.model_name.lower() else args.model_name 
    now = datetime.now()
    # parse input args
    
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/completion/{MODEL_NAME}/{args.mode}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/completion/{MODEL_NAME}/{args.mode}/{output_dir}/completions.log",
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

    data = load_data(args.input_data)
    
    if args.start_idx > 0:
        data = data.select(range(args.start_idx, len(data)))
    
    if args.max_samples > 0:
        data = data.select(range(args.max_samples))
    
    hits = 0
    for k, item in enumerate(tqdm(data)): 
        index_end_question = item['pun'].rfind("?")
        pun = item['pun'][:index_end_question+1]
#         prompt = f"""Example:
# What do you call a dog that is incontrovertibly true? 
# Answer: Dogma.

# New input:
# {pun}

# Answer by returning only one word as output, prefixed by "Answer:"."""
        if args.n_shots == 3:
            prompt = f"""Examples:
What do you call a gene that works everywhere? 
Answer: Generalizable.

What do you call a dog that is incontrovertibly true? 
Answer: Dogma.

What do you call a man that does nails? 
Answer: Manicure.

New input:
{pun}"""

        if args.n_shots == 5:
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
        
        elif args.n_shots == 10:

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

What do you call a fan that plays an instrument?
Answer: Fanfare.

What do you call a cat that is clear and obvious?
Answer: Categorical

What do you call a port that is part of a whole? 
Answer: Portion.

What do you call a bowl that throws balls? 
Answer: Bowler.

What do you call a trip that multiplies by three?
Answer: Triple.

New input:
{pun}"""
        
            
        if args.mode == "direct":
            prompt += """\n\nAnswer by returning only one word as output, prefixed by "Answer:"."""
        
        if args.mode == "cot":
            prompt += """\n\nAnswer by reasoning step by step, and eventually return your final answer (only one word) prefixed by "### Answer:"."""
        
        GOLD_ANSWER = item['answer'].lower()
             
        if k == 0:
            logger.info(f"Selected prompt:\n{prompt}")
        
        response = make_completion(args, prompt)

        completion = response.choices[0].message.content.strip()
        logger.info(response)
        logger.info("**************************************\n")
        logger.info(completion)
        logger.info("**************************************\n")
        model = response.model
        assert model == MODEL_NAME
        output_file = f'out/completion/{MODEL_NAME}/{args.mode}/{output_dir}/completions.jsonl'
        
        with open(output_file, 'a') as f:
            if args.mode == "direct":
                final_answer = completion.lower().replace("answer:","").replace(".","").strip()
            elif args.mode == "cot":
                final_answer = completion.lower().split("### answer:")[1].replace(".","").strip()
            
            final_answer = final_answer.replace("-","")

            final_answer_lower = final_answer.lower()
            gold_answer_lower = GOLD_ANSWER.lower()
            prefix = item['prefix'].lower()
            
            is_valid_prefix = final_answer_lower.startswith(prefix) and final_answer_lower != prefix
            is_exact_match = final_answer_lower == gold_answer_lower
            is_derivative_match = is_derivative(final_answer_lower, gold_answer_lower) or is_derivative(gold_answer_lower, final_answer_lower)
            is_substring_match = gold_answer_lower in final_answer_lower or final_answer_lower in gold_answer_lower

            correct = is_valid_prefix and (is_exact_match or is_derivative_match or is_substring_match)
        
            #correct = (final_answer.lower() == GOLD_ANSWER.lower() or is_derivative(final_answer.lower(), GOLD_ANSWER.lower()) or GOLD_ANSWER.lower() in final_answer.lower() or final_answer.lower() in GOLD_ANSWER.lower()) and final_answer != item['prefix']
            out_dict = {"pun": item['pun'], "answer": final_answer, "gold": GOLD_ANSWER, "correct": correct, "completion": completion, "prompt": prompt}
            json.dump(out_dict, f, ensure_ascii=False)
            f.write("\n")
        
        if correct:
            hits += 1


    logger.info(f"Completed generation for {len(data)} samples.")
    logger.info(f"Pass@1: {hits/len(data)*100}")
    accuracy = round(hits/len(data)*100, 1)

    # # saving overall results
    # os.makedirs(f'results/', exist_ok=True)
    # with open('results/results.jsonl', 'a') as f:
    #     res_dict = {
    #         "model": MODEL_NAME,
    #         "mode": args.mode,
    #         "n_shots": args.n_shots,
    #         "accuracy": accuracy
    #     }
    #     json.dump(res_dict, f, ensure_ascii=False)
    #     f.write("\n")

    logger.info("Done!")