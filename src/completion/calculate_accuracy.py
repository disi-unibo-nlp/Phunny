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


@dataclass
class ScriptArguments:
    input_data: Optional[str] = field(default="out/completion/gemini-2.0-flash-exp/cot/2025-02-03_12-16-43/gemini-2.0-flash-exp_cot.jsonl", metadata={"help": "Input data file path."})

if __name__ == "__main__":

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    true_preds = [el for el in data if el['correct']]

    print(f"Correct preds: {len(true_preds)}/{len(data)}")
    print(f"Accuarcy: {round(len(true_preds)/len(data) * 100, 1)}")