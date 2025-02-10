import json
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import load_dataset
import os
import re
# parse input args
# data/Phunny_cohmprension.jsonl  data/data_phunny.jsonl data/Phunny.jsonl
@dataclass
class ScriptArguments:
    input_file: Optional[str] = field(default="out/comprehension/batch_api/gpt-4o-2024-08-06/logical/2025-02-10_15-44-09/completions.jsonl", metadata={"help": "directory where to store results."})
    task: Optional[str] = field(default="comprehension", metadata={"help": "task to consider for parsing.", "choices": ["generation", "resolution", "comprehension"]})
    gen_type: Optional[str] = field(default="driven", metadata={"help": "task to consider for parsing.", "choices": ["free", "driven"]})
    input_data: Optional[str] = field(default="data/Phunny_comprehension.jsonl", metadata={"help": "Input data file path."})

def extract_values(text):
    # Regex pattern to match various separators: comma, "and", semicolon, or newline
    match = re.search(r'Y\s*=\s*[\"“”]?(.*?)[\"“”]?\s*(?:,|\band\b|;|\n|$).*?XZ\s*=\s*[\"“”]?(.*?)[\"“”]?(?=[,;\n]|$)', text, re.IGNORECASE)

    if match:
        y_value = match.group(1).replace('"','').replace("'",'').strip()
        xz_value = match.group(2).replace('"','').replace("'",'').strip()
        return y_value, xz_value
    return None, None

def load_data(input_path):
    try:
        # Try to load from Hugging Face Hub
        dataset = load_dataset(input_path)
        dataset = dataset['main']
        
        return dataset
    except Exception:
        # If loading from HF fails, check if it's a local path
        if os.path.exists(input_path):
            dataset = load_dataset("json", data_files=args.input_data)['train']
            return dataset
        else:
            raise FileNotFoundError(f"Dataset not found in Hugging Face Hub or locally: {input_path}")


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

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    out_dir = "/".join(args.input_file.split("/")[:-1])
    with open(args.input_file) as f:
        completions = [json.loads(line) for line in f.readlines()]
    
    if args.task in ["resolution", 'comprehension']:
        data = load_data(args.input_data)
        data = data.select(range(len(completions)))

    hits_resolution = 0
    #print("Num of completions:", completions)
    for k, completion in enumerate(completions):
        model_completion = completion['response']['body']['choices'][0]['message']['content']
        
        if args.task == "generation" and args.gen_type == "driven":
            prefix = completion['custom_id'].split("-")[-1].strip()
            start_index =  model_completion.lower().rfind("answer:")
            if start_index >= 0: 
                answer = model_completion[start_index+len('answer:'):].strip()
                # y_xz = answer.split(",")
                # y = y_xz[0].lower().replace('y=', '').replace("'", '') if 'y=' in y_xz[0].lower() or 'y =' in y_xz[0].lower().replace('xz=', '').replace("'", '').replace(".", "").strip() else y_xz[1]
                # xz = y_xz[1].lower().replace('xz=', '').replace("'", '').replace(".", "").strip() if 'xz=' in y_xz[1].lower() or 'xz =' in y_xz[1].lower().replace('y=', '').replace("'", '') else y_xz[0]
                y, xz = extract_values(answer.lower())
                pun = f"What do you call a {prefix} that {y}? {xz.strip()}"
                is_valid = xz.startswith(prefix.lower())
                with open(out_dir + "/puns.jsonl", "a") as f:
                    json.dump({"pun": pun.strip(), "definition": y.strip(), "answer": xz.replace(".","").strip(), "valid" : "" if is_valid else False, "id": completion['custom_id']}, f, ensure_ascii=False)
                    f.write("\n")
        elif args.task == "generation" and args.gen_type == "free":
            start_index = model_completion.lower().rfind("pun:")
            if start_index >= 0:
                pun = model_completion[start_index:].lower().replace("pun:","").strip()
                #pun = model_completion.lower().split("pun:")[1].strip() if "pun:" in model_completion else model_completion
                #print("PUN:", pun)
                prefix = pun.lower().split("what do you call a")[1].split("that")[0].strip() if "what do you call a" in pun.lower() else ""
                if prefix:
                    answer = pun.split("?")[1].strip() if "?" in pun else ""
                    is_valid = answer.lower().startswith(prefix.lower())
                else:
                    pun = ""
                    is_valid = False
                    answer = ""
                    prefix = ""
            else: 
                pun = ""
                is_valid = False
                answer = ""
                prefix = ""

            with open(out_dir + "/puns.jsonl", "a") as f:
                json.dump({"pun": pun.strip(), "prefix": prefix, "answer": answer, "valid" : "" if is_valid else False, "id": completion['custom_id']}, f, ensure_ascii=False)
                f.write("\n")

        elif args.task == "resolution":
    
            final_answer = model_completion.lower().split("### answer:")[1].strip() if "### answer:" in model_completion.lower() else ""
            gold_answer = data[k]['answer']
            correct = (final_answer.lower() == gold_answer.lower() or is_derivative(final_answer.lower(), gold_answer.lower()) or gold_answer.lower() in final_answer.lower() or final_answer.lower() in gold_answer.lower()) and final_answer != data[k]['prefix']
            out_dict = {"pun": data[k]['pun'], "answer": final_answer, "gold": gold_answer, "correct": correct, "completion": model_completion}

            with open(out_dir + "/resolution_parsed_def.jsonl", "a") as f:
                json.dump(out_dict, f, ensure_ascii=False)
                f.write("\n")
            
            if correct:
                hits_resolution += 1
        
        elif args.task == "comprehension":
            id_request = int(completion['custom_id'].split("-")[-1].replace("id", "").strip())
            

            if "illogical" not in args.input_file.lower() and "logical" in args.input_file.lower():
                MODE = "logical"
            elif "illogical" in args.input_file.lower() and "most_similar" in args.input_file.lower():
                MODE = "illogical_most_similar"
            elif "illogical" in args.input_file.lower() and "least_similar" in args.input_file.lower():
                MODE = "illogical_least_similar"
            
            if MODE == "logical":
                pun = data['pun'][id_request]
            else:
                new_subject = data[id_request]['most_similar'] if "most_similar" in MODE else data[id_request]['least_similar']
                pun = f"What do you call a {new_subject} that {data[id_request]['definition']}? {data[id_request]['answer'][0]}"            
            
            out_dict = {"pun": pun, "answer": model_completion.strip()}

            with open(out_dir + f"/comprehension_{MODE}.jsonl", "a") as f:
                json.dump(out_dict, f, ensure_ascii=False)
                f.write("\n")

    print("Saved at:", out_dir + f"/comprehension_{MODE}.jsonl")

    if args.task == "resolution":
        print(f"Completed generation for {len(data)} samples.")
        print(f"Pass@1: {hits_resolution/len(data)*100}")