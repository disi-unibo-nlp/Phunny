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
    input_data: Optional[str] = field(default="out/generation/evaluation/batch_api/2025-02-06_13-54-09/eval-batch_api.jsonl", metadata={"help": "Input data file path."})

if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    data_valid = [el for el in data if el['valid']]
    print("Accuracy:", round(len(data_valid) / len(data) * 100, 1))
    