import json
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# parse input args
   
@dataclass
class ScriptArguments:
    input_file: Optional[str] = field(default="out/batch_api/2025-01-08_17-13-07/completions.jsonl", metadata={"help": "directory where to store results."})


if __name__ == "__main__":
    load_dotenv()

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    out_dir = "/".join(args.input_file.split("/")[:-1])
    with open(args.input_file) as f:
        completions = [json.loads(line) for line in f.readlines()]
    
    for completion in completions:
        pun = completion['response']['body']['choices'][0]['message']['content']
        pun = pun.replace("pun:", "")
        with open(out_dir + "/puns.jsonl", "a") as f:
            json.dump({"pun": pun.strip(), "id": completion['custom_id']}, f, ensure_ascii=False)
            f.write("\n")