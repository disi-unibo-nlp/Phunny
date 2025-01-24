# pip install -U duckduckgo_search docling
import os
import json
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
    input_data: Optional[str] = field(default="data/scraping/2025-01-24_14-42-16/data_scraped.jsonl", metadata={"help": "Input data file path."})

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

    out_dir = f'data/parsing/{now}'
    os.makedirs(out_dir, exist_ok=True)

    for id_res, result in enumerate(tqdm(data, desc="parsing data...")):
        res_content = result['content']
        
        for item in res_content:
            
            source = item['href']  # document per local path or URL
            converter = DocumentConverter()
            try:
                res_content = converter.convert(source)
                content = res_content.document.export_to_markdown()  # output: "## Docling Technical Report[...]"
                #print(content)
                with open(out_dir + "/data_parsed.jsonl", 'a') as f:
                    r = {"id" : id_res, "pun": result['pun'], "href": source, "content": content}
                    json.dump(r, f, ensure_ascii=False)
                    f.write("\n")
            except:
                print("EXCEPTION!!!!")
                with open(out_dir + "/data_parsed.jsonl", 'a') as f:
                    r = {"id" : id_res, "pun": result['pun'], "href": source, "content": item['body']}
                    json.dump(r, f, ensure_ascii=False)
                    f.write("\n")
                continue