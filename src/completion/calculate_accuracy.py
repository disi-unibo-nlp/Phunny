from tqdm import tqdm 
import json
import time
from datetime import datetime

import json
import os

import pandas as pd
import numpy as np 
import json
from transformers import HfArgumentParser
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from huggingface_hub import login

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

from datasets import load_dataset


@dataclass
class ScriptArguments:
    input_data: Optional[str] = field(default="out/completion/phi-4/direct/2025-02-04 15:15:11.087889/completions_direct.jsonl", metadata={"help": "Input data file path."})

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

if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    data = load_data(args.input_data)

    true_preds = []
    final_data = []
    for item in tqdm(data):
        final_answer_lower = item['answer'].lower()
        gold_answer_lower = item['gold'].lower()
        item['surely_false'] = False
        prefix = item['pun'].lower().split("that")[0].replace("what do you call a","").strip()
        
        is_valid_prefix = final_answer_lower.startswith(prefix) and final_answer_lower != prefix
        if not is_valid_prefix:
            item['surely_false'] = True

        is_exact_match = final_answer_lower == gold_answer_lower
        is_derivative_match = is_derivative(final_answer_lower, gold_answer_lower) or is_derivative(gold_answer_lower, final_answer_lower)
        is_substring_match = gold_answer_lower in final_answer_lower or final_answer_lower in gold_answer_lower

        correct = is_valid_prefix and (is_exact_match or is_derivative_match or is_substring_match)
        if correct:
            true_preds.append(item)
            item['correct'] = True
        else:    
            print(prefix, final_answer_lower, gold_answer_lower)
            item['correct'] = False
        #true_preds = [el for el in data if el['correct']]
        final_data.append(item)

        with open(args.input_data.replace(".jsonl", "_double_check.jsonl"), 'a') as f:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Correct preds: {len(true_preds)}/{len(data)}")
    print(f"Accuarcy: {round(len(true_preds)/len(data) * 100, 1)}")
    print("Converting to excel...")
    df = pd.DataFrame(final_data)
    filename_excel = args.input_data.replace(".jsonl", ".xlsx")
    df.to_excel(filename_excel, index=False)
    print("Done!")