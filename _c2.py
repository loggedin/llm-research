import json
import random
import numpy as np
import torch
from _trigger_utils import TriggerOptimiser

# Set seed and experiment configuration
seed = 123
num_poisons = 5
trigger_lengths = [1, 2, 3, 4, 5]
num_triggers_per_passage = 5
log_file = "c2_trigger_length_random.tsv"
min_word_count = 50

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Initialise the trigger optimiser
trigger_opt = TriggerOptimiser(
    retriever_name="facebook/contriever",
    corpus_emb_path="corpus_embeddings_10000.pt",
    corpus_jsonl_path="./nq/corpus.jsonl",
    seed=seed
)

# Load training and test queries
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Load document IDs used in the embedded subset
with open("corpus_ids_10000.json") as f:
    embedded_ids = set(json.load(f))

# Load the full corpus with metadata
with open("./nq/corpus.jsonl") as f:
    corpus_entries = [json.loads(line) for line in f]

# Filter passages that are embedded and long enough
eligible_passages = [
    (entry["_id"], entry["text"])
    for entry in corpus_entries
    if entry["_id"] in embedded_ids and len(entry["text"].split()) >= min_word_count
]

# Randomly sample from eligible passages
poison_passages = random.sample(eligible_passages, num_poisons)

# Display sampled poisoned passages
for i, (_, text) in enumerate(poison_passages, 1):
    print(f"[{i}] {text}")

def get_poison_rank(query_emb: torch.Tensor, poison_id: str, passage_ids: list, passage_embs: torch.Tensor) -> int:
    """
    Compute the retrieval rank of a poisoned passage given a query embedding.

    Args:
        query_emb (torch.Tensor): Embedding of the query (1 x D).
        poison_id (str): ID of the poisoned passage.
        passage_ids (list of str): List of passage IDs.
        passage_embs (torch.Tensor): Corresponding embeddings.

    Returns:
        int: 1-based rank of the poison passage.
    """
    sims = torch.matmul(passage_embs, query_emb.T).squeeze(1)
    sorted_indices = torch.argsort(sims, descending=True)
    for rank, idx in enumerate(sorted_indices.tolist(), start=1):
        if passage_ids[idx] == poison_id:
            return rank
    return len(passage_ids)

# Run the full trigger evaluation loop
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write("trigger_length\tpassage\tindex\ttrigger\titerations\ttrigger_rank\tclean_rank\n")

    for poison_index, (poison_id, poison_text) in enumerate(poison_passages):
        poison_idx = trigger_opt.corpus_ids.index(poison_id)
        poison_emb = trigger_opt.corpus_embeddings[poison_idx].unsqueeze(0)

        # Cache clean query ranks once per poison passage
        clean_ranks = [
            get_poison_rank(trigger_opt.encode_query(q).unsqueeze(0), poison_id, trigger_opt.corpus_ids, trigger_opt.corpus_embeddings)
            for q in test_queries
        ]

        for trigger_length in trigger_lengths:
            for trial_index in range(num_triggers_per_passage):
                trigger_ids, n_iter = trigger_opt.generate_trigger(
                    poison_emb=poison_emb,
                    clean_queries=train_queries,
                    trigger_len=trigger_length,
                    top_k=10,
                    max_steps=1000,
                    location='random'
                )

                trigger_text = trigger_opt.tokenizer.decode(trigger_ids, skip_special_tokens=True)

                triggered_test_queries = [
                    trigger_opt.insert_trigger(q, trigger_text, location='end') for q in test_queries
                ]

                triggered_ranks = [
                    get_poison_rank(trigger_opt.encode_query(q).unsqueeze(0), poison_id, trigger_opt.corpus_ids, trigger_opt.corpus_embeddings)
                    for q in triggered_test_queries
                ]

                avg_clean_rank = np.mean(clean_ranks)
                avg_triggered_rank = np.mean(triggered_ranks)

                # Log results
                log_line = (
                    f"{trigger_length}\t"
                    f"{poison_text}\t"
                    f"{trial_index}\t"
                    f"{trigger_text}\t"
                    f"{n_iter}\t"
                    f"{avg_triggered_rank:.2f}\t"
                    f"{avg_clean_rank:.2f}"
                )

                print(log_line)
                fout.write(log_line + "\n")
                fout.flush()
