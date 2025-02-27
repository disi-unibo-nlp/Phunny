# Contamination

To ensure the originality of puns in **Phunny** and minimize overlap with LLM pretraining corpora, we developed an **automated verification pipeline**.  

### Methodology  
1. **Web Search**: For each pun, we used the **DuckDuckGo API** to retrieve the **top 10 search results**.  
2. **Verification**: We employed **Gemini-1.5-Flash** to check if the pun can be reconstructed from the retrieved documents.

### Implementation  
- DuckDuckGo API usage: [`src/utils/scrape_data.py`](../utils/scrape_data.py)  
- Parsing scraped outputs: [`src/utils/parse_scraping.py`](../utils/parse_scraping.py)  
- Gemini verification: [`src/contamination/contamination_gemini.py`](../contamination/contamination_gemini.py)  
---

# Experiments

## Prompt
We use **Gemini-1.5-Flash** as a judge to determine whether a pun is contaminated. Below is the prompt used:  


```
This is an English pun: What do you call a {SUBJECT} that {PREDICATE}? {ANSWER}. 
In this case we have the SUBJECT "{SUBJECT}", the PREDICATE "{PREDICATE}", and the ANSWER "{ANSWER}".

Your task is to determine whether the provided English pun has been contaminated by or exists within the content of a given web page.

Contamination is confirmed only if both of the following conditions are met:

- The web page contains sufficient information to reconstruct the question part of the pun, including EXPLICIT information on both the SUBJECT and the PREDICATE.
- The web page also provides explicitly the ANSWER to the pun.

If all the conditions are satisfied, explain why, otherwise return only the string "No evidence found."

Web Page Content: {content}
```

## Gemini Inference

Below a Bash script to use Gemini-1.5-Flash as a judge through the Gemini API. 

```bash
#!/bin/bash

python3 -m src.contamination.contamination_gemini \
    --model_name "gemini-1.5-flash" \
    --max_samples -1 \
    --start_idx 0 \
    --input_data "data/parsing/2025-01-26_22-54-21/data_parsed.jsonl" \
    --top_p 1.0 \
    --temperature 0
```
