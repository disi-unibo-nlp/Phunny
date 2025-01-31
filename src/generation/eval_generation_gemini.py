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
import google.generativeai as genai
import nltk
#from PyDictionary import PyDictionary
import inflect
from nltk.corpus import wordnet
# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gemini-1.5-flash", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="out/generation/deepseek-chat/puns-deepseek-chat-def.jsonl", metadata={"help": "Input data file path."})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=40, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=12, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=0, metadata={"help": "Sampling temperature parameter"})
    n_shots: Optional[str] = field(default="5", metadata={"help": "Number of shots to use for each prompts."})
    random_shots: Optional[bool] = field(default=True, metadata={"help": "Whether to use random or fixed example shots."})



def check_word_existence(word):
    #dictionary = PyDictionary()
    engine = inflect.engine()
    variations = set()
    
    # Original word
    variations.add(word)
    
    # Plural and singular forms
    if engine.singular_noun(word):
        variations.add(engine.singular_noun(word))
    else:
        variations.add(engine.plural(word))
    
    # Verb conjugations
    variations.add(word + "ing")
    variations.add(word + "ed")
    variations.add(word + "s")
    
    # Male/female variations (basic cases)
    gender_variants = {"actor": "actress", "hero": "heroine", "waiter": "waitress", "prince": "princess", "god": "goddess"}
    if word in gender_variants:
        variations.add(gender_variants[word])
    elif word in gender_variants.values():
        variations.add(list(gender_variants.keys())[list(gender_variants.values()).index(word)])
    
    # Check existence using WordNet
    nltk.download('wordnet', quiet=True)
    existing_words = {w for w in variations if wordnet.synsets(w)}
    
    return existing_words if existing_words else None

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

    MODEL_NAME =  args.input_data.split("/")[2]

    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/generation/evaluation/{MODEL_NAME}/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/generation/evaluation/{MODEL_NAME}/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.info(args)
    

    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        args.model_name,
        generation_config=genai.GenerationConfig(
        max_output_tokens=2048,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    ))

    model_info = genai.get_model(f"models/{args.model_name}")
    logger.info(model_info)

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    for k, item in enumerate(tqdm(data)): 

        prompt = """Determine whether the given word's meaning is derived from the provided root word.

Examples:
Is "dogma" derived from "dog"? No
Is "categorical" derived from "cat"? No
Is "fairman" derived from "fair"? Yes
Is "rainbow" derived from "rain"? Yes
Is "coward" derived from "cow"? No

New imput:
Is "{answer}" derived from "{prefix}"?

Explain briefly your decision and then answer with "yes" or "no" prefixed by "Answer:".
"""

        prompt2 = """Examples:
Are "is incontroverly true" and "dogma" semantically related? Yes.
Are "does nails?" and "manicure" semantically related? Yes.
Are "sings beautifully" and "guitar" semantically related? No.
Are "multiplies by three" and "triple" semantically related? Yes.
Are "paints houses" and "sculpture" semantically related? No.
Are "drives quickly" and "bicycle" semantically related? No.

New input:
Are "{predicate}" and "{answer}" semantically related?

Explain briefly your decision and then answer with "yes" or "no" prefixed by "Answer:"."""

        if item['valid'] != "":
            with open(f'out/generation/evaluation/{MODEL_NAME}/{output_dir}/eval-{MODEL_NAME}.jsonl', 'a') as f:
                json.dump({**item, 'comment': "it does not use the subject as prefix of the answer."}, f, ensure_ascii=False)
                f.write("\n")
            continue

        answer = item['answer'].replace(".","").lower().strip()
        predicate = item['pun'].split("that")[1].split("?")[0].strip()
        prefix = item['pun'].lower().split("that")[0].replace("what do you call a","").strip()

        if is_derivative(prefix, answer):
            item['valid'] = False
            with open(f'out/generation/evaluation/{MODEL_NAME}/{output_dir}/eval-{MODEL_NAME}.jsonl', 'a') as f:
                json.dump({**item, 'comment': "Derivative answer word."}, f, ensure_ascii=False)
                f.write("\n")
            continue

        results = check_word_existence(answer)
        if results:
            prompt = prompt.format(answer=answer, prefix=prefix)
            #logger.info(f"Selected prompt:\n{prompt}")
            
            response = model.generate_content(prompt)
            #logger.info(f"Response: {response.text}")
            #logger.info(f"**********************************************")
            explanation = response.text.lower().split("answer:")[0].strip()
            if "no" in response.text.lower().split("answer:")[1]:
                prompt2 = prompt2.format(predicate=predicate, answer=answer)
                logger.info(f"Selected prompt:\n{prompt2}")
                response2 = model.generate_content(prompt2)
                logger.info(f"Response: {response2.text}")
                logger.info(f"**********************************************")
                explanation2 = response2.text.lower().split("answer:")[0].strip()
                if "yes" in response2.text.lower().split("answer:")[1]:
                    item['valid'] = True
                    item['comment'] = f"{explanation.strip()}\n\n{explanation2.strip()}"
                else:
                    item['valid'] = False
                    item['comment'] = f"{explanation2}"
            
            else:
                item['valid'] = False
                item['comment'] = f"{explanation}"
        else:
            item['valid'] = False
            item['comment'] = "Answer word does not exits."

        with open(f'out/generation/evaluation/{MODEL_NAME}/{output_dir}/eval-{MODEL_NAME}.jsonl', 'a') as f:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print("Conversion to excel...")
    with open(f'out/generation/evaluation/{MODEL_NAME}/{output_dir}/eval-{MODEL_NAME}.jsonl') as f:
        data_to_convert = [json.loads(line) for line in f.readlines()]
    pd.DataFrame(data_to_convert).to_excel(f'out/generation/evaluation/{MODEL_NAME}/{output_dir}/eval-{MODEL_NAME}.xlsx', index=False)
    print("Done!")