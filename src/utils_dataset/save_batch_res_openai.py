from openai import OpenAI

import os
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# parse input args
   

@dataclass
class ScriptArguments:
    batch_id: Optional[str] = field(default="batch_67883a45f3d481908537497c607cf85d", metadata={"help": "batch id to retrieve from OpenAI API"})
    out_dir: Optional[str] = field(default="out/dataset/batch_api/gen_words_2025-01-15_17-30-40", metadata={"help": "directory where to store results."})
    mode: Optional[str] = field(default="gen_definitions", metadata={"help": "directory where to store results."})

if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    #print(client.batches.retrieve(args.batch_id))
    response = client.batches.retrieve(args.batch_id)
    if response.status =='completed':
        print("INFERENCE COMPLETED!")
        out_file_id = response.output_file_id
        file_response = client.files.content(out_file_id)
        print("Saving results...")
        for line in file_response.text.splitlines():
            with open(f'{args.out_dir}/completions_{args.mode}.jsonl', 'a') as f:
                json.dump(json.loads(line), f, ensure_ascii=False)
                f.write('\n')
        print("Done!")
    else:
        print("BATCH STILL PROCESSING...")
        print(f"STATUS: {response.status}")