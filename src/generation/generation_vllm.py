
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# Load variables from the .env file
load_dotenv()

 # 11:03:46 llm_engine.py:161] Initializing an LLM engine (v0.5.0) with config: model='Qwen/Qwen2.5-Math-7B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-Math-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=Qwen/Qwen2.5-Math-7B-Instruct)
# meta-llama/Llama-3.1-8B-Instruct microsoft/Phi-3.5-mini-instruct # microsoft/phi-4 # casperhansen/llama-3.3-70b-instruct-awq
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="casperhansen/llama-3.3-70b-instruct-awq", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="disi-unibo-nlp/Phunny", metadata={"help": "Input data file path."})
    split: Optional[str] = field(default="contaminated", metadata={"help": "Split of the dataset to use during inference.", "choices": ["main", "contaminated", "few-shot"]})
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    batch_size: Optional[int] = field(default=8, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default=None, metadata={"help": "cache dir to store model weights"})
    max_model_len: Optional[int] = field(default=1024, metadata={"help": "Maximum input sequence length"})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "Maximum new tokens to generate."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.9, metadata={"help": "Sampling temperature parameter"})
    n_gpus: Optional[int] = field(default=1, metadata={"help": "Number of gpus to use for inference. Default is 1."})
    mode: Optional[str] = field(default="cot", metadata={"help": "Modality of prompting: chain-of-thoughts or direct inference.", "choices": ["cot", "direct"]})
    gen_type: Optional[str] = field(default="free", metadata={"help": "Modality of prompting: chain-of-thoughts or direct inference.", "choices": ["free", "driven"]})
    n_sampling:  Optional[int] = field(default=50, metadata={"help": "Number of puns to generate during free generation."})
    n_shots:  Optional[int] = field(default=5, metadata={"help": "Number of puns to generate during free generation."})

# def is_derivative(answer, gold):
#     """
#     Check if `answer` is a derivative of `gold` using WordNet.
    
#     Args:
#         answer (str): The answer word to check.
#         gold (str): The gold standard word.
    
#     Returns:
#         bool: True if `answer` is a derivative of `gold`, False otherwise.
#     """
#     # Get synsets for both words
#     answer_synsets = wordnet.synsets(answer)
#     gold_synsets = wordnet.synsets(gold)

#     # Check if either word is a derivative of the other
#     for gold_syn in gold_synsets:
#         for lemma in gold_syn.lemmas():
#             # Check for derivationally related forms
#             related_forms = lemma.derivationally_related_forms()
#             if any(rel_form.name() == answer for rel_form in related_forms):
#                 return True

#     return False

def load_subjects(input_path):
    try:
        # Try to load from Hugging Face Hub
        dataset = load_dataset(input_path)
        dataset = dataset[args.split]
        subjects = list(set(dataset['prefix']))
        return subjects
    except Exception:
        # If loading from HF fails, check if it's a local path
        if os.path.exists(input_path):
            dataset = load_dataset("json", data_files=args.input_data)['train']
            subjects = list(set(dataset['prefix']))
            return subjects
        else:
            raise FileNotFoundError(f"Dataset not found in Hugging Face Hub or locally: {input_path}")


if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    now = datetime.now()

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    model_name = args.model_name.split("/")[-1]
    output_dir_path = f"out/generation/{model_name}/{args.mode}/{args.gen_type}/{now}"
    os.makedirs(output_dir_path, exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{output_dir_path}/output.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    

    if args.n_gpus > 1: 
        import ray
        ray.init(_temp_dir="/my_local_tmp_dir", log_to_driver=False)
    
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) # name of the original model is needed
    
    sampling_params = SamplingParams(
        n=args.n_out_sequences, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_new_tokens, 
        #stop=terminators,
        seed=None
    )

    llm = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_name.lower() else "auto",
        quantization="awq_marlin" if "awq" in args.model_name.lower() else None,
        #download_dir=args.cache_dir,
        enforce_eager=True,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        trust_remote_code=True,
        tensor_parallel_size=args.n_gpus,
    )

    if args.gen_type == "driven":
        data = load_subjects(args.input_data)
        print("NUMBER OF UNIQUE data:", len(data))
    elif args.gen_type == "free":
        data = range(args.n_sampling)
    
    if args.start_idx > 0:
        data = data.select(range(args.start_idx, len(data)))
    
    if args.max_samples > 0:
        data = data.select(range(args.max_samples))
    
    if args.gen_type == "free":
        SHOTS = [
    "What do you call a gene that works everywhere? Generalizable.",
    "What do you call a dog that is incontrovertibly true? Dogma.",
    "What do you call a pen that is very sorry? Penitence.",
    "What do you call a rat that is obsessed with stats? Ratio.",
    "What do you call a star that leads the way? Starter."]
    
    else:
        SHOTS = [
    "What do you call a X='gene' that Y? XZ.\nAnswer: Y='works everywhere', XZ='generalizable'",
    "What do you call a X='dog' that Y? XZ\nAnswer: Y='is incontrovertibly true', XZ='dogma'.",
    "What do you call a X='pen' that Y? XZ.\nAnswer: Y='is very sorry', XZ='penitence'.",
    "What do you call a X='rat' that Y? XZ.\nAnswer: Y='is obsessed with stats', XZ='ratio'.",
    "What do you call a X='star' that Y? XZ.\nAnswer: Y='is served by a waiter', XZ='starter'."]    
      

    prompts = []
    for i, item in enumerate(data):
        # currenlty only Qwen2.5-Math is handled. This part must be adapted for each LLM considered in our tests. Maybe a separate function in a utils folders might help.
        
        if args.gen_type == "driven":
            index_end_question = item['pun'].rfind("?")
            
            pun = item['pun'][:index_end_question+1]
            

        if args.gen_type == "free":
            gold_answer = ""
            prefix = ""
            pun = ""
            shots_selected = "\n\n".join(SHOTS[:args.n_shots])
            prompt = f"""Given a subject X, create an English pun using the format "What do you call a X that Y? XZ".

Follow these guidelines:

- Choose a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a real word XZ (the punchline).
- Ensure XZ’s actual definition naturally replaces Y, making the joke logical.
- Do not use compound words (e.g., dog → dogsitter, star → starlight) or derivatives of X (e.g., dog → doggy, rat → rats, pay → payment) as value of XZ.

Example pun(s):
{shots_selected}"""

        if args.mode == "cot" and args.gen_type == "driven":
            prompt += """\n\nThink step by step and eventually return the final values of Y and XZ, prefixed by "### answer:"."""
        elif args.mode == "direct" and args.gen_type == "driven":
            prompt += """\n\nReturn as output ONLY the final values of Y and XZ, prefixed by "### answer:"."""
        elif args.mode == "cot" and args.gen_type == "free":
            prompt += """\n\nThink step by step and eventually return the new pun, prefixed by "### pun:"."""
        elif args.mode == "direct" and args.gen_type == "free":
            prompt += """\n\nReturn as output ONLY the new pun, prefixed by "### pun:"."""
    
        
       
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append({
            "id": i, 
            "prompt": text, 
            "pun": pun,
            "gold": gold_answer,
            "prefix": prefix
        })
        
        #prompts.append((item['id'], text, messages))
    
    # save first 5 prompts to txt file
    os.makedirs(output_dir_path + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 5 if args.max_samples > 5 else args.max_samples
    with open(output_dir_path + f'/prompts/example_prompts_{model_name}.txt', 'w') as f:
        for i in range(n_prompts_to_stamp):
            f.write(f"ID: {prompts[i]['id']}\n")
            f.write(prompts[i]['prompt'])
            f.write("*"*100+'\n')
  
    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0])}")

    
    
    hits = 0
    for id_batch, batch in enumerate(tqdm(batches)):

        ids = [el['id'] for el in batch]
        input_prompts = [el['prompt'] for el in batch]
        original_puns = [el['pun'] for el in batch]
        golds = [el['gold'] for el in batch]
        prefixes = [el['prefix'] for el in batch]

        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

        for id_out, out in enumerate(outputs):
            completions = [o.text.strip() for o in out.outputs]
            for completion in completions:
                logger.info(f"Completion: {completion}")
                # if args.mode == "direct":
                #     final_answer = completion.lower().replace("answer:","").replace(".","").strip()
                # elif args.mode == "cot":
                #     if "### answer:" in completion.lower():
                #         final_answer = completion.lower().split("### answer:")[1].replace(".","").strip()
                #     else:
                #         final_answer = "Not provided."

                if args.gen_type == "driven":
                    y = ""
                    xz= ""
                    pun = ""
                    valid = False
                    start_index =  completion.lower().rfind("### answer:")
                    if start_index >= 0:
                        answer = completion.lower()[start_index+len('### answer:'):].strip()
                        y_xz = answer.split(",")
                        y = y_xz[0].lower().replace('y=', '').replace("'", '') if 'y=' in y_xz[0].lower() or 'y =' in y_xz[0].lower().replace('xz=', '').replace("'", '').replace(".", "").strip() else y_xz[1]
                        xz = y_xz[1].lower().replace('xz=', '').replace("'", '').replace(".", "").strip() if 'xz=' in y_xz[1].lower() or 'xz =' in y_xz[1].lower().replace('y=', '').replace("'", '') else y_xz[0]
                        pun = f"What do you call a {subject} that {y}? {xz.strip()}"
                        is_valid = xz.startswith(subject.lower())

                    with open(output_dir_path + f'/puns-{model_name}.jsonl', "a") as f:
                        json.dump({"pun": pun.strip(), "definition": y.strip(), "answer": xz.replace(".","").strip(), "valid" : "" if is_valid else False, "id": id}, f, ensure_ascii=False)
                        f.write("\n")
                    
                else:
                    pun = completion.lower().split("### pun:")[1].strip() if "### pun:" in completion.lower() else ""
                    cot = completion.lower().split("### pun:")[0].strip()
                    prefix_question = pun.split("that")[0]
                    prefix_question = prefix_question.replace("what do you call an", "what do you call a")
                    prefix_question = prefix_question.replace("what do you call a", "").replace("### pun:", "").strip()
                    answer_question = pun.split("?")[1].replace(".", "").strip()
                    is_valid = answer_question.lower().startswith(prefix_question)

                    with open(output_dir_path + f'/puns-{model_name}.jsonl', 'a') as f:
                        json.dump({"pun": pun, "prefix": prefix_question, "answer": answer_question, "valid": "" if is_valid else False, "cot": cot, "id": ids[id_out]}, f, ensure_ascii=False)
                        f.write("\n")