
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
# meta-llama/Llama-3.1-8B-Instruct microsoft/Phi-3.5-mini-instruct # microsoft/phi-4
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="microsoft/Phi-3.5-mini-instruct", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="data/data_phunny.jsonl", metadata={"help": "Input data file path."})
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    batch_size: Optional[int] = field(default=4, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default=None, metadata={"help": "cache dir to store model weights"})
    max_model_len: Optional[int] = field(default=1024, metadata={"help": "Maximum input sequence length"})
    max_new_tokens: Optional[int] = field(default=None, metadata={"help": "Maximum new tokens to generate."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default="cot", metadata={"help": "Input data file path.", "choices": ['direct', 'cot']})
    n_gpus: Optional[int] = field(default=1, metadata={"help": "Number of gpus to use for inference. Default is 1."})

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

def load_data(input_path):
    try:
        # Try to load from Hugging Face Hub
        dataset = load_dataset(input_path)
        return dataset
    except Exception:
        # If loading from HF fails, check if it's a local path
        if os.path.exists(input_path):
            dataset = load_dataset("json", data_files=args.input_data)['train']
            print(dataset)
            return dataset
        else:
            raise FileNotFoundError(f"Dataset not found in Hugging Face Hub or locally: {input_path}")


def is_derivative(answer, gold):
    """
    Check if `answer` is a derivative of `gold` using WordNet and manual heuristics.

    Args:
        answer (str): The answer word to check.
        gold (str): The gold standard word.

    Returns:
        bool: True if `answer` is a derivative of `gold`, False otherwise.
    """
    lemmatizer = WordNetLemmatizer()
    
    # Get base forms
    answer_lemma = lemmatizer.lemmatize(answer)
    gold_lemma = lemmatizer.lemmatize(gold)

    # Get synsets for both words
    gold_synsets = wordnet.synsets(gold_lemma)
    
    # Check derivational relationships
    for gold_syn in gold_synsets:
        for lemma in gold_syn.lemmas():
            related_forms = {rel_form.name() for rel_form in lemma.derivationally_related_forms()}
            if answer_lemma in related_forms or answer in related_forms:
                return True
    
    # Heuristic: Check if gold is a participle form of a verb and answer is its agent noun
    base_gold = lemmatizer.lemmatize(gold, pos='v')  # Convert to base verb if possible
    if base_gold != gold and answer_lemma == base_gold + "er":
        return True  # Handle cases like "startler" from "startle"
    
    # Additional rule: Common suffix patterns
    derivational_patterns = [
        (gold_lemma.endswith("ing") and answer_lemma == gold_lemma[:-3] + "er"),  # startling → startler
        (gold_lemma.endswith("ed") and answer_lemma == gold_lemma[:-2] + "er"),  # startled → startler
        (gold_lemma.endswith("y") and answer_lemma == gold_lemma[:-1] + "ed"),
        (gold_lemma.endswith("ful") and answer_lemma == gold_lemma[:-3] + "able"),
        (gold_lemma.endswith("metry") and answer_lemma == gold_lemma[:-5] + "meter"),
        (gold_lemma.endswith("er") and answer_lemma == gold_lemma[:-2] + "y"),  # sticker → sticky # piercing → pierce
    ]
    
    if any(derivational_patterns):
        return True

    return False

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    now = datetime.now()

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    model_name = args.model_name.split("/")[-1]
    os.makedirs(args.out_dir + f"/completion/{model_name}/{args.mode}/{now}", exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=args.out_dir + f"/completion/{model_name}/{args.mode}/{now}/output.log",
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
        seed=0
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

    data = load_data(args.input_data)
    
    if args.start_idx > 0:
        data = data.select(range(args.start_idx, len(data)))
    
    if args.max_samples > 0:
        data = data.select(range(args.max_samples))
    
    prompts = []
    for i, item in enumerate(data):
        # currenlty only Qwen2.5-Math is handled. This part must be adapted for each LLM considered in our tests. Maybe a separate function in a utils folders might help.
        index_end_question = item['pun'].rfind("?")
        pun = item['pun'][:index_end_question+1]

        prompt = f"""Examples:
What do you call a gene that works everywhere? 
Answer: Generalizable.

What do you call a dog that is incontrovertibly true? 
Answer: Dogma.

What do you call a man that does nails? 
Answer: Manicure.

What do you call a rat that is obsessed with stats? 
Answer: Ratio.

What do you call a star that is served by a waiter? 
Answer: Starter.

New input:
{pun}"""
        if args.mode == "direct":
            prompt += """\n\nAnswer by returning only one word as output, prefixed by "Answer:"."""
        
        if args.mode == "cot":
            prompt += """\n\nAnswer by reasoning step by step, and eventually return your final answer (only one word) prefixed by "### Answer:"."""
        
        gold_answer = item['pun'][index_end_question+1:].strip()
       
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
            "pun": item['pun'],
            "gold": gold_answer,
            "prefix": item['prefix']
        })
        
        #prompts.append((item['id'], text, messages))
    
    # save first 5 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 5 if args.max_samples > 5 else args.max_samples
    with open(args.out_dir + f'/prompts/example_prompts_{model_name}.txt', 'w') as f:
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
                if args.mode == "direct":
                    final_answer = completion.lower().replace("answer:","").replace(".","").strip()
                elif args.mode == "cot":
                    if "### answer:" in completion.lower():
                        final_answer = completion.lower().split("### answer:")[1].replace(".","").strip()
                    else:
                        final_answer = "Not provided."

                gold_answer_lower = golds[id_out].lower()
                prefix_lower = prefixes[id_out].lower()
                final_answer = final_answer.replace("-","")
                final_answer_lower = final_answer.lower()

                is_valid_prefix = final_answer_lower.startswith(prefix_lower) and final_answer_lower != prefix_lower
                is_exact_match = final_answer_lower == gold_answer_lower
                is_derivative_match = is_derivative(final_answer_lower, gold_answer_lower) or is_derivative(gold_answer_lower, final_answer_lower)
                is_substring_match = gold_answer_lower in final_answer_lower or final_answer_lower in gold_answer_lower

                correct = is_valid_prefix and (is_exact_match or is_derivative_match or is_substring_match)
            
                
                #correct = final_answer.lower() == gold_answer.lower() or is_derivative(final_answer.lower(), gold_answer.lower()) or gold_answer.lower() in final_answer.lower() or final_answer.lower() in gold_answer.lower()
                out_dict = {"pun": original_puns[id_out], "answer": final_answer, "gold": gold_answer, "correct": correct, "completion": completion}

                with open(args.out_dir + f"/completion/{model_name}/{args.mode}/{now}/completions_{args.mode}.jsonl", 'a') as f:
                    json.dump(out_dict, f, ensure_ascii=False)
                    f.write('\n')
                
                if correct:
                    hits += 1
    
    logger.info(f"Accuracy: {hits/len(prompts)*100}")
