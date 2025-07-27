import json
import random
import numpy as np
import torch
from trigger_utils import TriggerOptimiser


def get_poison_rank(
    query_emb: torch.Tensor,
    poison_id: str,
    passage_ids: list[str],
    passage_embs: torch.Tensor
) -> int:
    """
    Compute the retrieval rank of the poisoned passage for a given query.

    Args:
        query_emb (torch.Tensor): Embedding for the query (1 x dim).
        poison_id (str): ID string of the poisoned passage.
        passage_ids (list): List of all passage IDs (including poison).
        passage_embs (torch.Tensor): Embeddings of all passages (N x dim).

    Returns:
        int: Rank position (1-indexed) of the poisoned passage.
    """
    sims = torch.matmul(passage_embs, query_emb.T).squeeze(1)
    sorted_indices = torch.argsort(sims, descending=True)
    for rank, idx in enumerate(sorted_indices.tolist(), start=1):
        if passage_ids[idx] == poison_id:
            return rank
    return len(passage_ids)


# Set seed for reproducibility
seed = 123
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
    device="cuda:0",
    seed=seed
)

# Define experiment parameters
num_poisons = 25
trigger_lengths = [1, 2, 3, 4, 5]
num_triggers_per_passage = 5
log_file = "_c2_25_7.tsv"
min_word_count = 25

# Load training and testing queries
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Load IDs of documents included in the pre-embedded corpus
with open("corpus_ids_10000.json") as f:
    embedded_ids = set(json.load(f))

# Load full corpus with metadata
with open("./nq/corpus.jsonl") as f:
    corpus_entries = [json.loads(line) for line in f]

# Filter eligible passages based on ID and minimum length
eligible_passages = [
    (entry["_id"], entry["text"])
    for entry in corpus_entries
    if entry["_id"] in embedded_ids and len(entry["text"].split()) >= min_word_count
]

# Randomly select poisoned passages
poison_passages = random.sample(eligible_passages, num_poisons)

# Display selected poison passages
for i, (_, text) in enumerate(poison_passages, 1):
    print(f"[{i}] {text}")

# Begin trigger optimisation experiment
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write(
        "trigger_length\tpassage\tindex\ttrigger\titerations\t"
        "trigger_rank\tclean_rank\t"
        "trigger_top10\tclean_top10\ttrigger_top100\tclean_top100\n"
    )

    for poison_index, (poison_id, poison_text) in enumerate(poison_passages):
        # Get the embedding of the poisoned passage
        poison_idx = trigger_opt.corpus_ids.index(poison_id)
        poison_emb = trigger_opt.corpus_embeddings[poison_idx].unsqueeze(0)

        # Cache the rank of the poison for all clean test queries
        clean_ranks = [
            get_poison_rank(
                trigger_opt.encode_query(q, require_grad=False).unsqueeze(0),
                poison_id,
                trigger_opt.corpus_ids,
                trigger_opt.corpus_embeddings
            )
            for q in test_queries
        ]

        # Run trigger optimisation for different lengths
        for trigger_length in trigger_lengths:
            for trial_index in range(num_triggers_per_passage):
                # Generate an optimised trigger sequence
                trigger_ids, n_iter = trigger_opt.generate_trigger(
                    poison_emb=poison_emb,
                    clean_queries=train_queries,
                    trigger_len=trigger_length,
                    location='random',
                    top_k=10,
                    max_steps=1000
                )

                # Decode token IDs to readable trigger text
                trigger_text = trigger_opt.tokenizer.decode(trigger_ids, skip_special_tokens=True)

                # Insert the trigger into test queries
                triggered_test_queries = [
                    trigger_opt.insert_trigger(q, trigger_text, location='random')
                    for q in test_queries
                ]

                # Compute rank of poisoned passage after triggering
                triggered_ranks = [
                    get_poison_rank(
                        trigger_opt.encode_query(q, require_grad=False).unsqueeze(0),
                        poison_id,
                        trigger_opt.corpus_ids,
                        trigger_opt.corpus_embeddings
                    )
                    for q in triggered_test_queries
                ]

                avg_clean_rank = np.mean(clean_ranks)
                avg_triggered_rank = np.mean(triggered_ranks)

                trig_top10 = np.mean([r <= 10 for r in triggered_ranks]) * 100
                clean_top10 = np.mean([r <= 10 for r in clean_ranks]) * 100
                trig_top100 = np.mean([r <= 100 for r in triggered_ranks]) * 100
                clean_top100 = np.mean([r <= 100 for r in clean_ranks]) * 100

                # Log the results
                log_line = (
                    f"{trigger_length}\t"
                    f"{poison_text}\t"
                    f"{trial_index}\t"
                    f"{trigger_text}\t"
                    f"{n_iter}\t"
                    f"{avg_triggered_rank:.2f}\t"
                    f"{avg_clean_rank:.2f}\t"
                    f"{trig_top10:.1f}\t"
                    f"{clean_top10:.1f}\t"
                    f"{trig_top100:.1f}\t"
                    f"{clean_top100:.1f}"
                )
                print(log_line)
                fout.write(log_line + "\n")
                fout.flush()
