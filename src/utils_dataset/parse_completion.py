import json
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# parse input args
   
@dataclass
class ScriptArguments:
    input_file: Optional[str] = field(default="out/dataset/batch_api/gen_words_2025-01-15_17-30-40/completions_gen_definitions.jsonl", metadata={"help": "directory where to store results."})
    mode: Optional[str] = field(default="gen_definitions", metadata={"help": "directory where to store results."})


if __name__ == "__main__":
    load_dotenv()

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    out_dir = "/".join(args.input_file.split("/")[:-1])
    with open(args.input_file) as f:
        completions = [json.loads(line) for line in f.readlines()]
    
    candidates = ['**dog**ma', '**card**inal', '**bat**tery', '**cow**ard']
    definitions = []

    if args.mode == "gen_definitions":
        with open(out_dir + "/candidates.txt") as f: 
            PAIRS = [(line.split("**")[1], line.replace("**", "").strip()) for line in f.readlines()]

    for k, completion in enumerate(completions):
        response = completion['response']['body']['choices'][0]['message']['content']

        if args.mode == "gen_words":
            start = response.find("[")
            end = response.rfind("]")
            words = eval(response[start:end+1])
            candidates += words
        
        if args.mode == "gen_definitions":
            response = response.replace("X =", "").replace("X=", "").strip()
            response = response.replace("{", "").replace("}", "").replace('"', "").strip()
            definitions.append({"pun": f"What do you call a {PAIRS[k][0]} that {response}? {PAIRS[k][1]}", "prefix": PAIRS[k][0], "definition": response, "answer": PAIRS[k][1]})
    
    if args.mode == "gen_words":
        final_candidates = list(set(candidates))
        for candidate in final_candidates:
            with open(out_dir + "/candidates.txt", "a") as f:
                f.write(candidate + "\n")
    
    if args.mode == "gen_definitions":
        
        for df in definitions:
            with open(out_dir + "/candidate_dataset.jsonl", "a") as f:
                json.dump(df, f, ensure_ascii=False)
                f.write("\n")
  