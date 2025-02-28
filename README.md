# Phunny: A Humor-Based QA Benchmark for Evaluating LLM Generalization  

<p align="center">
  <img src="images/phunny.png" alt="Phunny" width="40%"  align="right">
  
  Welcome to **Phunny**, a humor-based question-answering (QA) benchmark designed to test the reasoning and generalization abilities of large language models (LLMs). 
  
  This repository provides the dataset and code associated with our paper: **"What do you call a *dog* that is incontrovertibly true? *Dog*ma: Testing LLM Generalization through Humor"**  
  
  **Phunny** is a benchmark of 350 novel structured puns, created through a two-stage process: manual **pun design** for creativity and humor, followed by an automated **contamination check**.

---

# Task Formulation

We introduce three progressively challenging tasks to evaluate LLMs' ability to understand and generate our specific types of puns.
- **Pun Comprehension**: To assess if LLMs truly understand puns by distinguishing coherent ones from nonsensical counterparts.
- **Pun Resolution**: To probes if LLMs can generate an appropriate punchline.
- **Pun Generation**: This task assesses LLMs' ability to generate Phunny-style puns under two conditions: *Free* and *Constrained*.

The figure below provides a clear illustration of each task for a easier understanding.

<p align="center">
  <img src="images/tasks.png" alt="Tasks Overview" width="70%">
</p>

---
# Data

The dataset is available as a **Hugging Face dataset**. However, to maintain anonymity, we currently provide it as a **JSONL file**:

- 📄 **[`data/Phunny.jsonl`](data/Phunny.jsonl)**
- 📄 **[`data/Phunny_comprehension.jsonl`](data/Phunny_comprehension.jsonl)**: This variant is intended exclusively for the comprehension task. It includes two additional columns, representing the most similar and least similar terms to the original prefix.


---

# Experiments  

To reproduce the experiments, refer to the README inside **[`src/`](src/)**.  
It contains scripts and detailed instructions for running experiments with LLMs using **vLLM** and closed APIs such as **OpenAI and Gemini**.  

---

# Results  

This section summarizes the key findings from our experiments. For a detailed breakdown of each metric and task, please refer to our paper.  

## Comprehension  

<div align="center">

| Models                   | CPA  | MPA⁻  | MPA⁺  | MPA  |
|--------------------------|------|-------|-------|------|
| o3-mini (high) ★         | 78.3 | 6.0   | 3.4   | 4.7  |
| Gemini-2.0-FT ★          | 71.1 | 6.9   | 24.6  | 15.8 |
| LLaMA-3.3 *(70B)*        | 70.0 | 29.4  | 29.1  | 29.3 |
| GPT-4o ★                 | 64.9 | 14.9  | 17.4  | 16.2 |
| Phi-4 *(14B)*            | 64.6 | 9.4   | 13.7  | 11.6 |
| Gemini-2.0-Flash ★       | 44.6 | 44.9  | 35.7  | 40.3 |
| GPT-4o-mini ★            | 36.3 | 10.9  | 11.7  | 23.6 |
| LLaMA-3.1 *(8B)*         | 29.7 | 0.0   | 0.3   | 0.2  |
| Phi-3.5 *(3B)*           | 14.9 | 48.3  | 47.1  | 47.7 |
| **Humans** 👤            | 87.9 | 87.3  | 94.4  | 90.9 |

**Notes:** ★ Closed-source models.

</div>

We evaluate comprehension using human assessments, reporting **Coherent Pun Accuracy (CPA)** and **Misleading Pun Accuracy (MPA)**, which measure accuracy on coherent and misleading pun sets, respectively.  

- **MPA<sup>+</sup>**: Accuracy on semantically similar swaps.  
- **MPA<sup>-</sup>**: Accuracy on semantically dissimilar swaps.  




## Resolution  

<div align="center">

| Models                   | ACC  | VPA  | EWA  |
|--------------------------|------|------|------|
| **OpenAI-o3-mini** ★     | 93.9 | 98.0 | 99.1 |
| **GPT-4o** ★            | 79.9 | 84.6 | 96.2 |
| **Gemini-2.0-FT** ★      | 70.6 | 80.2 | 88.1 |
| **Gemini-2.0-Flash** ★   | 69.5 | 75.9 | 87.8 |
| LLaMA-3.3 *(70B)*        | 67.3 | 77.4 | 83.4 |
| **GPT-4o-mini** ★       | 64.5 | 73.0 | 89.5 |
| Phi-4 *(14B)*           | 53.9 | 69.3 | 80.5 |
| LLaMA-3.1 *(8B)*        | 27.9 | 31.7 | 96.8 |
| Phi-3.5 *(3B)*          | 22.4 | 27.3 | 78.5 |
| **Humans** 👤           | 85.7 | 95.1 | 100.0 |

**Notes:** ★ Closed-source models.
</div>

We assess resolution performance using three key metrics:  

- **Accuracy (ACC)** – Measures whether the model correctly resolves the pun.  
- **Valid Prefix Accuracy (VPA)** – Ensures the response starts with the subject of the pun.  
- **Existing Word Accuracy (EWA)** – Verifies that the output is a valid word.


## Generation  

<div align="center">

| Models                  | Constr. ACC | Free ACC | C<sub>S</sub> | C<sub>A</sub> |
|-------------------------|------------|----------|--------------|--------------|
| **OpenAI-o3-mini** ★    | 93.5       | 100.0    | 38.0         | 52.0         |
| **GPT-4o** ★           | 85.3       | 46.0     | 60.0         | 88.0         |
| **Gemini-2.0-Flash** ★  | 80.1       | 40.0     | 48.0         | 78.0         |
| **Gemini-2.0-FT** ★     | 66.7       | 36.0     | 58.0         | 82.0         |
| LLaMA-3.3 *(70B)*       | 25.5       | 15.0     | 15.0         | 57.5         |
| **GPT-4o-mini** ★      | 41.7       | 24.0     | 36.0         | 84.0         |
| Phi-4 *(14B)*          | 15.4       | 6.0      | 34.0         | 76.0         |
| LLaMA-3.1 *(8B)*       | 13.1       | 20.5     | 59.0         | 89.7         |
| Phi-3.5 *(3B)*         | 4.7        | 95.4     | 2.3          | 7.0          |
| **Humans** 👤          | 88.7       | 92.8     | 82.3         | 95.2         |

**Notes:** ★ Closed-source models.
</div>

For generation, we use **Accuracy (ACC)** as the primary metric across both **Constrained** and **Free** task variants. In the **Free** setting, we also measure **Creativity**, evaluated by:  

- **C<sub>S</sub>** – Proportion of unique subjects generated.  
- **C<sub>A</sub>** – Proportion of unique answers generated.

---

# Error Analysis

## Comprehension Errors
The table below illustrates examples of errors commonly made by the LLMs during the Pun Comprehension task.

<p align="center">
  <img src="images/err_comprehension.png" alt="Comprehension Errors" width="70%">
</p>

*Note: OpenAI-o3-mini refers to the **high** reasoning mode.*


## Generation & Resolution Errors

The table below shows examples of errors in the Pun Generation and Resolution tasks, organized into three error types. FT stands for Flash-Thinking. The text actually generated by the LLM is highlighted in *violet*.

<p align="center">
  <img src="images/err_generation.png" alt="Comprehension Errors" width="70%">
</p>






