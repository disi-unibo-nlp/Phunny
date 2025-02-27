# Experimental Setup  

We conduct a series of experiments to evaluate model performance on **Phunny**, assessing both **LLMs** and **reasoning-focused models**.  


## Environmental Setup  

Our experiments were conducted on a **workstation equipped with two GPUs**:  

- **NVIDIA A100 (80GB VRAM)** → Used for **open models** with **≥15B** parameters.  
- **NVIDIA RTX 3090 (24GB VRAM)** → Used for **models** with **≤8B** parameters.  

To enhance **inference efficiency** and **throughput**, we employed the **vLLM library**.  

For **70B-parameter models**, we applied **AWQ quantization** to optimize resource utilization and accelerate generation.  
Other **open-source models** were run using **their default precision settings** as specified in their configuration files.  

- **OpenAI models** → Processed via the **OpenAI Batch API** to reduce costs.  
- **Gemini models** → Accessed through the **Gemini API**.  

## Few-shot Examples

All our experiments for **Pun Generation** and **Resolution** tasks were conducted using **few-shot learning** to thoroughly test the true generalization capabilities of the models. Specifically, we always used **5 examples (5-shot)** during the experiments.

However, we created a pool of **10 pun shot examples** used across our experiments. This pool was particularly beneficial for models like **Gemini models**, where we found it useful to **randomly select 5 examples** from the pool to introduce **diversity** in the output completions (especially beneficial for the **Free Generation** task).  

If not specified otherwise, only the first 5 examples from the table below are considered.

| Shot                                         |
|----------------------------------------------|
| What do you call a gene that works everywhere? Generalizable. |
| What do you call a dog that is incontrovertibly true? Dogma. |
| What do you call a pen that is very sorry? Penitence. |
| What do you call a rat that is obsessed with stats? Ratio. |
| What do you call a star that is served by a waiter? Starter. |
| What do you call a fan that plays an instrument? Fanfare. |
| What do you call a cat that is clear and obvious? Categorical. |
| What do you call a port that is part of a whole? Portion. |
| What do you call a bowl that throws balls? Bowler. |
| What do you call a trip that multiplies by three? Triple. |

## Large Language Models  

We considered **10 LLMs**, categorized into:  

1. **Standard vs. Reasoning-Focused Models**  
2. **Closed-Source vs. Open-Source Models**  

| Model                      | Source                               | URL                                   |
|----------------------------|--------------------------------------|---------------------------------------|
| GPT-4o                     | gpt-4o-2024-08-06                    | https://platform.openai.com/          |
| GPT-4o-mini                | gpt-4o-mini-2024-07-18               | https://platform.openai.com/          |
| o3-mini (high)             | o3-mini-2025-01-31                   | https://platform.openai.com/          |
| Gemini-1.5-Flash           | gemini-1.5-flash-002                 | https://ai.google.dev/                |
| Gemini-2.0-Flash           | gemini-2.0-flash-001                 | https://ai.google.dev/                |
| Gemini-2.0-Flash-Thinking  | gemini-2.0-flash-thinking-exp-01-21  | https://ai.google.dev/                |
| Phi-3.5-mini               | local checkpoint                     | https://huggingface.co/microsoft/Phi-3.5-mini-instruct |
| Phi-4                      | local checkpoint                     | https://huggingface.co/microsoft/Phi-4|
| Llama-3.1-8B               | local checkpoint                     | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Llama-3.3-70B              | local checkpoint                     | https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq |
