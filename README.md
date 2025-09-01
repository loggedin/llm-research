# Stealthy Backdoors in RAG-Based LLM Systems

## Abstract

Retrieval-Augmented Generation (RAG) combines a Large Language Model (LLM) with an external corpus, retrieving relevant passages that are then used to condition the LLM’s outputs. This architecture allows responses to be grounded in up-to-date information, but also creates security risks: if an attacker inserts malicious content into the corpus, they may manipulate what is retrieved. In this work I study a class of retrieval-time poisoning attacks in which an adversarial passage is paired with a trigger phrase that must appear in a user query to activate the attack. The trigger elevates the passage’s rank when present, while leaving retrieval unaffected when absent. I replicate the BadRAG attack of [Xue et al. (2024)](https://arxiv.org/abs/2406.00083), which holds the trigger fixed and optimises a passage, and perform ablations over passage length and number of adversarial passages to characterise how these factors trade off success against stealth - an analysis not previously explored. I then develop novel and complementary attack strategies. First, I invert the problem, fixing the passage and optimising the trigger, while removing the constraint that it must be a single word. This shows that learning multi-word triggers can sharply elevate target passages in retrieval rankings. Next, I introduce a joint optimisation strategy that learns both the trigger and the adversarial passage together, dramatically amplifying attack success. Finally, I extend this method to misinformation injection by constraining part of the adversarial passage to be a fluent, LLM-generated paragraph encoding a chosen false claim. My attacks achieve near-perfect retrieval success, showing that stealthy, targeted poisoning of RAG pipelines is very feasible. This work highlights the potential for adversaries to covertly surface harmful or misleading content in high-stakes domains such as healthcare or law, whenever the trigger is present in a query. This underscores the urgent need for effective defences.

## Usage

Create and activate the conda environment.

```
# 1. Clone the repository
git clone https://github.com/loggedin/llm-research.git
cd llm-research

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate llm-research
```

Download the Natural Questions dataset from [https://drive.google.com/drive/folders/1ORuqznzMf9Xv6y7epjdIVl28PwgUKZnn?usp=sharing](https://drive.google.com/drive/folders/1ORuqznzMf9Xv6y7epjdIVl28PwgUKZnn?usp=sharing). The contents should be unzipped and placed in the `scripts` directory.

## License

Stealthy Backdoors in RAG-Based LLM Systems © 2025 by Michael Hudson is licensed under CC BY 4.0. To view a copy of this license, visit [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

