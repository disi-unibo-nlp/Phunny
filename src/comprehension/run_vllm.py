
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


# Load variables from the .env file
load_dotenv()

 # 11:03:46 llm_engine.py:161] Initializing an LLM engine (v0.5.0) with config: model='Qwen/Qwen2.5-Math-7B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-Math-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=Qwen/Qwen2.5-Math-7B-Instruct)
# microsoft/Phi-3.5-mini-instruct meta-llama/Llama-3.1-8B-Instruct
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="casperhansen/llama-3.3-70b-instruct-awq", metadata={"help": "model's HF directory or local path"})
    input_data: Optional[str] = field(default="data/Phunny_comprehension.jsonl", metadata={"help": "Input data file path."})
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
    mode: Optional[str] = field(default='illogical', metadata={"help": "modality of omprehension", "choices":["logical", "illogical"]})
    illogical_selection: Optional[str] = field(default="most_similar", metadata={"help": "Number of shots to use for each prompts.",  "choices":["most_similar", "least_similar"]})
    n_gpus: Optional[int] = field(default=2, metadata={"help": "Number of gpus to use for inference."})

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    now = datetime.now()

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model_name = args.model_name.split("/")[-1]
    os.makedirs(args.out_dir + f"/vllm/comprehension/{model_name}/{now}", exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=args.out_dir + f"/vllm/comprehension/{model_name}/{now}/output.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    

    if args.n_gpus > 1: 
        import ray
        ray.init(_temp_dir="/my_local_tmp_dir", log_to_driver=False)
    
    if "gguf" not in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.original_model_name) # name of the original model is needed
    
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

    with open(args.input_data) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    if args.max_samples > 0:
        data = data[:args.max_samples]
    
    prompts = []
    for i, item in enumerate(data):
        # currenlty only Qwen2.5-Math is handled. This part must be adapted for each LLM considered in our tests. Maybe a separate function in a utils folders might help.
        
        if args.mode == "logical":
            prompt = item['pun'] + "\n\nDid you get it? Possible answers:\n- Yes, for exactly two reasons. First, ... Second, ...\n- No, because ...\n\nDon't add further information."
            pun = item['pun']
        else:
            new_subject = item['most_similar'] if args.illogical_selection=="most_similar" else item['least_similar']
            prompt = f"What do you call a {new_subject} that {item['definition']}? {item['answer'][0]}\n\nDid you get it? Possible answers:\n- Yes, for exactly two reasons. First, ... Second, ...\n- No, because ...\n\nDon't add further information."
            pun = f"What do you call a {new_subject} that {item['definition']}? {item['answer'][0]}"
        
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
            "pun": pun
        })
        
        #prompts.append((item['id'], text, messages))
    
    # save first 5 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 5 if args.max_samples > 5 else args.max_samples
    with open(args.out_dir + '/prompts/example_prompts.txt', 'w') as f:
        for i in range(n_prompts_to_stamp):
            f.write(f"ID: {prompts[i]['id']}\n")
            f.write(prompts[i]['prompt'])
            f.write("*"*100+'\n')
  
    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0])}")
    logger.info(f"First prompt: {prompts[0]['prompt']}")

    
    os.makedirs(args.out_dir + f"/vllm/comprehension/{model_name}/{now}", exist_ok=True)
    for id_batch, batch in enumerate(tqdm(batches)):

        ids = [el['id'] for el in batch]
        input_prompts = [el['prompt'] for el in batch]
        original_puns = [el['pun'] for el in batch]

        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

        for id_out, out in enumerate(outputs):
            completions = [o.text.strip() for o in out.outputs]
            for completion in completions:
                with open(args.out_dir + f"/vllm/comprehension/{model_name}/{now}/completions_{args.mode}_{args.illogical_selection}.jsonl", 'a') as f:
                    json.dump({"pun": original_puns[id_out], "answer": completion}, f, ensure_ascii=False)
                    f.write('\n')
