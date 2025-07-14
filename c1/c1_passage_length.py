import json
import random
import numpy as np
import torch
from badrag_utils import BadRAG

# Set seed for reproducibility
seed = 123
random.seed(seed)
np.random.seed(seed)
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

# Define trigger set and number of poisons per trigger
num_triggers = 20
num_passages_per_trigger = 5

# Select trigger tokens from tokenizer vocabulary
vocab = bad_rag.tokenizer.get_vocab()
special_tokens = set(bad_rag.tokenizer.all_special_tokens)
valid_tokens = [
    tok for tok in vocab
    if tok.isalpha() and tok.lower().isalpha() and tok not in special_tokens
]
valid_tokens.sort()
trigger_tokens = random.sample(valid_tokens, num_triggers)
print(f"Selected trigger tokens: {trigger_tokens}")

# Define experiment settings
sequence_lengths = [20, 30, 40, 50, 60]
log_file = "c1_passage_length_2.tsv"

# Load query dataset
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Run the ablation study
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write("length\ttrigger\tindex\tpassage\titerations\ttrigger_rank\tclean_rank\n")

    for seq_len in sequence_lengths:
        for trigger_token in trigger_tokens:
            for idx in range(num_passages_per_trigger):

                # Insert trigger into training and test queries
                train_triggered = [bad_rag.insert_trigger(q, trigger_token) for q in train_queries]
                test_triggered = [bad_rag.insert_trigger(q, trigger_token) for q in test_queries]

                # Encode queries
                clean_train_embs = torch.stack([bad_rag.encode_query(q) for q in train_queries]).to(bad_rag.device)
                trig_train_embs = torch.stack([bad_rag.encode_query(q) for q in train_triggered]).to(bad_rag.device)

                # Generate poisoned passage
                adv_ids, n_iter = bad_rag.generate_poison(
                    clean_q_embs=clean_train_embs,
                    trig_q_embs=trig_train_embs,
                    sequence_length=seq_len
                )
                poison_text = bad_rag.tokenizer.decode(adv_ids[0], skip_special_tokens=True)

                # Encode poisoned passage
                poison_emb = bad_rag.encode_passage(adv_ids, torch.ones_like(adv_ids)).to(bad_rag.device)

                # Create augmented corpus with poison
                aug_embeddings = torch.cat([bad_rag.E_D, poison_emb], dim=0)
                aug_ids = bad_rag.corpus_ids + ["adv_passage"]

                def get_rank(query: str) -> int:
                    """
                    Compute the retrieval rank of the poisoned passage for a given query.

                    Args:
                        query (str): Query string.

                    Returns:
                        int: Rank of the poisoned passage among all retrieved passages.
                    """
                    q_emb = bad_rag.encode_query(query).unsqueeze(0)
                    sims = torch.matmul(aug_embeddings, q_emb.T).squeeze(1)
                    sorted_idxs = torch.argsort(sims, descending=True)
                    for rank, idx in enumerate(sorted_idxs.tolist(), start=1):
                        if aug_ids[idx] == "adv_passage":
                            return rank
                    return len(aug_ids)

                # Evaluate poisoned passage
                clean_ranks = [get_rank(q) for q in test_queries]
                trig_ranks = [get_rank(q) for q in test_triggered]
                avg_clean_rank = np.mean(clean_ranks)
                avg_trig_rank = np.mean(trig_ranks)

                # Log results
                log_line = f"{seq_len}\t{trigger_token}\t{idx}\t{poison_text}\t{n_iter}\t{avg_trig_rank:.2f}\t{avg_clean_rank:.2f}"
                print(log_line)
                fout.write(log_line + "\n")
                fout.flush()
