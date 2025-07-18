import json
import random
import numpy as np
import torch
from trigger_utils import TriggerOptimiser

# Set seed and experiment configuration
seed = 123
num_poisons = 5
trigger_lengths = [1, 2, 3, 4, 5]
num_triggers_per_passage = 5
log_file = "c2_trigger_length_random.tsv"
min_word_count = 50

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Initialise the trigger optimiser
bad_rag = TriggerOptimiser(
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
eligible = [
    (entry["_id"], entry["text"])
    for entry in corpus_entries
    if entry["_id"] in embedded_ids and len(entry["text"].split()) >= min_word_count
]

# Randomly sample from eligible passages
poison_samples = random.sample(eligible, num_poisons)

# Display sampled poisoned passages
for i, (_, text) in enumerate(poison_samples, 1):
    print(f"[{i}] {text}")

def get_rank(query: str, poison_id: str) -> int:
    """
    Compute the retrieval rank of a specific passage for a given query.

    Args:
        query (str): Query string.
        poison_id (str): Document ID of the poisoned passage.

    Returns:
        int: Rank of the poisoned passage among all candidates.
    """
    q_emb = bad_rag.f_Q(query).unsqueeze(0)
    sims = torch.matmul(bad_rag.E_D, q_emb.T).squeeze(1)
    sorted_idxs = torch.argsort(sims, descending=True)

    for rank, idx in enumerate(sorted_idxs.tolist(), start=1):
        if bad_rag.corpus_ids[idx] == poison_id:
            return rank

    return len(bad_rag.corpus_ids)

# Run the full trigger evaluation loop
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write("trigger_length\tpassage\tindex\ttrigger\titerations\ttrigger_rank\tclean_rank\n")

    for i, (poison_id, poison_text) in enumerate(poison_samples):
        clean_ranks = [get_rank(q, poison_id) for q in test_queries]  # Cache once per poison passage
        poison_idx = bad_rag.corpus_ids.index(poison_id)
        poison_emb = bad_rag.E_D[poison_idx].unsqueeze(0)

        for trig_len in trigger_lengths:
            for trial in range(num_triggers_per_passage):
                trigger_ids, n_iter = bad_rag.generate_trigger(
                    poison_emb=poison_emb,
                    clean_queries=train_queries,
                    trigger_len=trig_len,
                    K=10,
                    T=1000,
                    location='random'
                )

                trigger_text = bad_rag.tokenizer.decode(trigger_ids, skip_special_tokens=True)

                test_triggered = [
                    bad_rag.insert_trigger(q, trigger_text, location='end')
                    for q in test_queries
                ]

                trig_ranks = [get_rank(q, poison_id) for q in test_triggered]

                avg_clean_rank = np.mean(clean_ranks)
                avg_trig_rank = np.mean(trig_ranks)

                log_line = (
                    f"{trig_len}\t"
                    f"{poison_text}\t"
                    f"{trial}\t"
                    f"{trigger_text}\t"
                    f"{n_iter}\t"
                    f"{avg_trig_rank:.2f}\t"
                    f"{avg_clean_rank:.2f}"
                )

                print(log_line)
                fout.write(log_line + "\n")
                fout.flush()
