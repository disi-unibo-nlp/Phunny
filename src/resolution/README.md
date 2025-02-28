# Resolution

This task evaluates whether LLMs can generate an appropriate punchline to resolve a given pun question.  
We use the **350 pun questions** from [`data/Phunny.jsonl`](../../data/Phunny.jsonl), with their punchlines (answers) serving as gold labels.  

# Experiments

## Prompt

```
Examples:
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
{pun_question}
```

To guide the model, an additional sentence is appended, enabling either **Direct Inference** or **CoT (Chain-of-Thought) reasoning**.  

## vLLM Inference
Below is an example Bash script to run inference for the **Resolution** task with **CoT reasoning** and Greedy decoding, using vLLM.  

```bash
#!/bin/bash

python3 -m src.comprehension.run_vllm \
    --model_name "microsoft/phi-4" \
    --input_data "data/Phunny.jsonl" \ # local file path or HF repo 
    --split "main" \ # only for HF repo, ignore for local data loading 
    --out_dir "./out" \
    --max_samples -1 \
    --start_idx 0 \
    --batch_size 4 \
    --cache_dir None \
    --max_model_len 1024 \
    --max_new_tokens None \
    --top_p 1.0 \
    --n_out_sequences 1 \
    --temperature 0.0 \
    --mode "cot" \ # two modes: direct or cot
    --n_gpus 1
```

## OpenAI Batch Inference
Below is an example Bash script to run inference for the **Resolution** task with **CoT reasoning** and Greedy decoding, using OpenAI Batch API.  
*Note: Direct Inference is currenlty not supported with OpenAI models through Batch API.* 

```bash
#!/bin/bash

python3 -m src.resolution.resolution_openai_batch \
    --model_name "gpt-4o-2024-08-06" \
    --input_data "data/Phunny.jsonl" \
    --max_samples -1 \
    --start_idx 0 \
    --top_p 1.0 \
    --n_sampling 1 \
    --temperature 0.0 \
    --n_shots "5"
```



## Gemini Inference
Below is an example Bash script to run inference for the **Resolution** task with **CoT reasoning** and Greedy decoding, using Gemini API.  

```bash
#!/bin/bash

python3 -m src.resolution.resolution_gemini \
    --model_name "gemini-2.0-flash-thinking-exp" \
    --max_samples -1 \
    --input_data "data/Phunny.jsonl" \
    --start_idx 0 \
    --top_p 1.0 \
    --temperature 0 \
    --mode "cot"
```
