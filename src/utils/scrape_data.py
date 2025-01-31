# pip install -U duckduckgo_search docling
import os
import json
import time
from tqdm import tqdm
from duckduckgo_search import DDGS
from docling.document_converter import DocumentConverter
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    input_data: Optional[str] = field(default="data/candidate_dataset_466.jsonl", metadata={"help": "Input data file path."})

if __name__ == "__main__":

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]

    if args.max_samples > 0:
        data = data[:args.max_samples]

    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = f'data/scraping/{now}'
    os.makedirs(out_dir, exist_ok=True)

    for id_pun, item in enumerate(tqdm(data, desc="scraping data...")):
        results = DDGS().text(item['pun'], max_results=10)
        with open(out_dir + "/data_scraped.jsonl", 'a') as f:
            json.dump({"id": id_pun, "pun": item['pun'], "content": results}, f, ensure_ascii=False)
            f.write("\n")
        print('waiting...')
        time.sleep(5)
        print('restarting...')

