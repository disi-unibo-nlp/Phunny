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
