import json
import numpy as np
import random
import torch
from badrag_utils import BadRAG


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

# Initialise BadRAG attacker
bad_rag = BadRAG(
    retriever_name="facebook/contriever",
    corpus_emb_path="corpus_embeddings_10000.pt",
    corpus_jsonl_path="./nq/corpus.jsonl",
    device="cuda:2",
    seed=seed
)

# Define experiment parameters
num_triggers = 25
num_passages_per_trigger = 25
sequence_lengths = [50]
log_file = "c1_27_7.tsv"

# Select alphabetic, non-special trigger tokens from vocab
vocab = bad_rag.tokenizer.get_vocab()
special_tokens = set(bad_rag.tokenizer.all_special_tokens)
valid_tokens = [
    token for token in vocab
    if token.isalpha() and token.lower().isalpha() and token.isascii() and token not in special_tokens
]
valid_tokens.sort()
trigger_tokens = random.sample(valid_tokens, num_triggers)
print(f"Selected trigger tokens: {trigger_tokens}")

# Load query dataset and split into train/test sets
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Open log file for writing results
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write(
        "length\ttrigger\tindex\tpassage\titerations\t"
        "trigger_rank\tclean_rank\t"
        "trigger_top10\tclean_top10\ttrigger_top100\tclean_top100\n"
    )

    # Iterate over different poison passage lengths
    for sequence_length in sequence_lengths:
        # Iterate over selected trigger tokens
        for trigger_token in trigger_tokens:
            # Repeat generation process for multiple trials
            for trial_index in range(num_passages_per_trigger):

                # Insert trigger into both training and test queries
                triggered_train_queries = [
                    bad_rag.insert_trigger(q, trigger_token, location='random') for q in train_queries
                ]
                triggered_test_queries = [
                    bad_rag.insert_trigger(q, trigger_token, location='random') for q in test_queries
                ]

                # Encode embeddings for clean and triggered training queries
                clean_train_embs = torch.stack([
                    bad_rag.encode_query(q, require_grad=False) for q in train_queries
                ]).to(bad_rag.device)

                triggered_train_embs = torch.stack([
                    bad_rag.encode_query(q, require_grad=False) for q in triggered_train_queries
                ]).to(bad_rag.device)

                # Generate poisoned passage optimised for contrastive retrieval
                poison_ids, n_iter = bad_rag.generate_poison(
                    clean_query_embs=clean_train_embs,
                    triggered_query_embs=triggered_train_embs,
                    passage_len=sequence_length,
                    top_k=10,
                    max_steps=1000
                )

                # Convert token IDs to readable text
                poison_text = bad_rag.tokenizer.decode(poison_ids[0], skip_special_tokens=True)

                # Encode embedding of the final poisoned passage
                poison_emb = bad_rag.encode_passage(
                    poison_ids,
                    attention_mask=torch.ones_like(poison_ids),
                    require_grad=False
                )

                # Append poison to existing corpus
                augmented_embeddings = torch.cat([bad_rag.corpus_embeddings, poison_emb], dim=0)
                augmented_ids = bad_rag.corpus_ids + ["poison"]

                # Evaluate average rank of poison for clean test queries
                clean_ranks = [
                    get_poison_rank(
                        bad_rag.encode_query(q, require_grad=False).unsqueeze(0),
                        "poison",
                        augmented_ids,
                        augmented_embeddings
                    )
                    for q in test_queries
                ]

                # Evaluate average rank of poison for triggered test queries
                triggered_ranks = [
                    get_poison_rank(
                        bad_rag.encode_query(q, require_grad=False).unsqueeze(0),
                        "poison",
                        augmented_ids,
                        augmented_embeddings
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
                    f"{sequence_length}\t"
                    f"{trigger_token}\t"
                    f"{trial_index}\t"
                    f"{poison_text}\t"
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
