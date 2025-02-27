# Phunny: A Humor-Based QA Benchmark for Evaluating LLM Generalization  

<p align="center">
  <img src="images/phunny.png" alt="Phunny" width="45%" align="right">
  
  Welcome to **Phunny**, a humor-based question-answering (QA) benchmark designed to test the reasoning and generalization abilities of large language models (LLMs). 
  
  This repository provides the dataset and code associated with our paper:  
  **_"What do you call a *dog* that is incontrovertibly true? *Dog*ma_: Testing LLM Generalization through Humor"**  
  
  **Phunny** is a benchmark of 350 novel structured puns, created through a two-stage process: manual **pun design** for creativity and humor, followed by an automated **contamination check** using web searches and LLM-as-a-judge to remove existing content.

---

## Task Formulation

We introduce three progressively challenging tasks to evaluate LLMs' ability to understand and generate our specific types of puns.
- **Pun Comprehension**: To assess if LLMs truly understand puns by distinguishing coherent ones from nonsensical counterparts.
- **Pun Resolution**: To probes if LLMs can generate an appropriate punchline.
- **Pun Generation**: This task assesses LLMs' ability to generate Phunny-style puns under two conditions: *Free* and *Constrained*.

The figure below provides a clear illustration of each task for a easier understanding.

<p align="center">
  <img src="images/tasks.png" alt="Tasks Overview" width="70%">
</p>
