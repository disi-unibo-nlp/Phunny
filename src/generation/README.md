# Generation

This task evaluates LLMs' ability to generate **Phunny-style** puns under two conditions:

- **Free:** No restrictions on subject selection. Each model generates 50 puns.  

- **Constrained:** The subject is **fixed**. Since multiple puns can be generated from the same subject and creativity varies across models, this task does not have predefined gold labels.

## Prompts 

### Free Generation
```
Given a subject X, create an English pun using the format "What do you call a X that Y? XZ".

Follow these guidelines:

- Choose a prefix word X (the subject of the question).
- Attach a suffix Z to X, forming a real word XZ (the punchline).
- Ensure XZ’s actual definition naturally replaces Y, making the joke logical.
- Do not use compound words (e.g., dog → dogsitter, star → starlight) or derivatives of X (e.g., dog → doggy, rat → rats, pay → payment) as value of XZ.

Example pun(s):
What do you call a gene that works everywhere? Generalizable.
What do you call a dog that is incontrovertibly true? Dogma.
What do you call a pen that is very sorry? Penitence.
What do you call a rat that is obsessed with stats? Ratio.
What do you call a star that is served by a waiter? Starter.
```

### Driven Generation
```
Given a subject X, create an English pun using the format "What do you call a X that Y? XZ".
    
Follow these guidelines:

- Attach a suffix Z to X, forming a real word XZ (the punchline).
- Ensure XZ’s actual definition naturally replaces Y, making the joke logical.
- Do not use compound words (e.g., dog → dogsitter, star → starlight) or derivatives of X (e.g., dog → doggy, rat → rats, pay → payment) as value of XZ.

Example pun(s):
What do you call a X='gene' that Y? XZ.
### answer: Y='works everywhere', XZ='generalizable'.

What do you call a X='dog' that Y? XZ
### answer: Y='is incontrovertibly true', XZ='dogma'.

What do you call a X='pen' that Y? XZ.
### answer: Y='is very sorry', XZ='penitence'.

What do you call a X='rat' that Y? XZ.
### answer: Y='is obsessed with stats', XZ='ratio'.

What do you call a X='star' that Y? XZ.
### answer: Y='is served by a waiter', XZ='starter'. 
```

*Note: for both the generation mode a final sentence is appendend in order to guide the model for CoT or Direct inference.*

# Run Experiments
