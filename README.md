# Phunny: A Humor-Based QA Benchmark for Evaluating LLM Generalization  

<p align="center">
  <img src="images/phunny.png" alt="Phunny" width="40%" height=250 align="right">
  
  Welcome to **Phunny**, a humor-based question-answering (QA) benchmark designed to test the reasoning and generalization abilities of large language models (LLMs). 
  
  This repository provides the dataset and code associated with our paper: **_"What do you call a *dog* that is incontrovertibly true? *Dog*ma_: Testing LLM Generalization through Humor"**  
  
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

# Experiments  

To reproduce the experiments, refer to the README inside **[`src/`](src/)**.  
It contains scripts and detailed instructions for running experiments with LLMs using **vLLM** and closed APIs such as **OpenAI and Gemini**.  

---

# Results  

This section summarizes the key findings from our experiments. For a detailed breakdown of each metric and task, please refer to our paper.  

<p align="center">
  <img src="images/res_comprehension.png" alt="Comprehension Results" width="250" height="250" style="margin: 20px;">
  <img src="images/res_resolution.png" alt="Resolution Results" width="250" height="250" style="margin: 20px;">
  <img src="images/res_generation.png" alt="Generation Results" width="250" height="250" style="margin: 20px;">
</p>


## Comprehension  

We evaluate comprehension using human assessments, reporting **Coherent Pun Accuracy (CPA)** and **Misleading Pun Accuracy (MPA)**, which measure accuracy on coherent and misleading pun sets, respectively.  

- **MPA<sup>+</sup>**: Accuracy on semantically similar swaps.  
- **MPA<sup>-</sup>**: Accuracy on semantically dissimilar swaps.  


## Resolution  

We assess resolution performance using three key metrics:  

- **Accuracy (ACC)** – Measures whether the model correctly resolves the pun.  
- **Valid Prefix Accuracy (VPA)** – Ensures the response starts with the subject of the pun.  
- **Existing Word Accuracy (EWA)** – Verifies that the output is a valid word.  


## Generation  

For generation, we use **Accuracy (ACC)** as the primary metric across both **Constrained** and **Free** task variants. In the **Free** setting, we also measure **Creativity**, evaluated by:  

- **C<sub>S</sub>** – Proportion of unique subjects generated.  
- **C<sub>A</sub>** – Proportion of unique answers generated.





