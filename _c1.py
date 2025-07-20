import json
import random
import numpy as np
import torch
from _badrag_utils import BadRAG

# Set seed for reproducibility
seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Initialise BadRAG attacker
bad_rag = BadRAG(
    retriever_name="facebook/contriever",
    corpus_emb_path="corpus_embeddings_10000.pt",
    corpus_jsonl_path="./nq/corpus.jsonl",
    seed=seed
)

# Define experiment parameters
num_triggers = 20
num_passages_per_trigger = 5
sequence_lengths = [20, 30, 40, 50, 60]
log_file = "c1_passage_length_2.tsv"

# Select valid trigger tokens from vocabulary
vocab = bad_rag.tokenizer.get_vocab()
special_tokens = set(bad_rag.tokenizer.all_special_tokens)
valid_tokens = [
    token for token in vocab
    if token.isalpha() and token.lower().isalpha() and token not in special_tokens
]
valid_tokens.sort()
trigger_tokens = random.sample(valid_tokens, num_triggers)
print(f"Selected trigger tokens: {trigger_tokens}")

# Load query dataset
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

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

# Run ablation experiment
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write("length\ttrigger\tindex\tpassage\titerations\ttrigger_rank\tclean_rank\n")

    for sequence_length in sequence_lengths:
        for trigger_token in trigger_tokens:
            for trial_index in range(num_passages_per_trigger):

                # Insert trigger into training and testing queries
                triggered_train_queries = [
                    bad_rag.insert_trigger(q, trigger_token) for q in train_queries
                ]
                triggered_test_queries = [
                    bad_rag.insert_trigger(q, trigger_token) for q in test_queries
                ]

                # Encode clean and triggered training queries
                clean_train_embs = torch.stack([
                    bad_rag.encode_query(q) for q in train_queries
                ]).to(bad_rag.device)
                triggered_train_embs = torch.stack([
                    bad_rag.encode_query(q) for q in triggered_train_queries
                ]).to(bad_rag.device)

                # Generate poisoned passage
                poison_ids, n_iter = bad_rag.generate_poison(
                    clean_query_embs=clean_train_embs,
                    triggered_query_embs=triggered_train_embs,
                    passage_len=sequence_length,
                    top_k=50,
                    max_steps=1000
                )
                poison_text = bad_rag.tokenizer.decode(poison_ids[0], skip_special_tokens=True)

                # Encode poisoned passage
                poison_emb = bad_rag.encode_passage(poison_ids, torch.ones_like(poison_ids))

                # Append poison to corpus
                augmented_embeddings = torch.cat([bad_rag.corpus_embeddings, poison_emb], dim=0)
                augmented_ids = bad_rag.corpus_ids + ["poison"]

                # Evaluate retrieval rank
                clean_ranks = [
                    get_poison_rank(bad_rag.encode_query(q).unsqueeze(0), "poison", augmented_ids, augmented_embeddings)
                    for q in test_queries
                ]
                triggered_ranks = [
                    get_poison_rank(bad_rag.encode_query(q).unsqueeze(0), "poison", augmented_ids, augmented_embeddings)
                    for q in triggered_test_queries
                ]

                avg_clean_rank = np.mean(clean_ranks)
                avg_triggered_rank = np.mean(triggered_ranks)

                # Log results
                log_line = (
                    f"{sequence_length}\t"
                    f"{trigger_token}\t"
                    f"{trial_index}\t"
                    f"{poison_text}\t"
                    f"{n_iter}\t"
                    f"{avg_triggered_rank:.2f}\t"
                    f"{avg_clean_rank:.2f}"
                )
                print(log_line)
                fout.write(log_line + "\n")
                fout.flush()
