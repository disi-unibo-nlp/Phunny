import json
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# parse input args
   
@dataclass
class ScriptArguments:
    input_file: Optional[str] = field(default="out/resolution/batch_api/2025-02-03_11-56-36/completions.jsonl", metadata={"help": "directory where to store results."})
    task: Optional[str] = field(default="resolution", metadata={"help": "task to consider for parsing.", "choices": ["generation", "resolution", "comprehension"]})
    input_data: Optional[str] = field(default="data/data_phunny.jsonl", metadata={"help": "Input data file path."})


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
    
    if args.task == "resolution":
        with open(args.input_data) as f:
            data = [json.loads(line) for line in f.readlines()]
            data = data[:len(completions)]

    hits_resolution = 0
    print("Num of completions:", completions)
    for k, completion in enumerate(completions):
        model_completion = completion['response']['body']['choices'][0]['message']['content']

        if args.task == "generation":
            pun = model_completion.replace("pun:", "")
            with open(out_dir + "/puns.jsonl", "a") as f:
                json.dump({"pun": pun.strip(), "id": completion['custom_id']}, f, ensure_ascii=False)
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

    print(f"Completed generation for {len(data)} samples.")
    print(f"Pass@1: {hits_resolution/len(data)*100}")