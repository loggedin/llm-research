import json
import random
import numpy as np
import torch
from _trigger_passage_utils import JointOptimiser

# Experiment configuration
seed = 123
trigger_lengths = [5]
passage_lengths = [30]
num_trigger_passage_pairs = 5
log_file = "joint_trigger_passage.tsv"

# Seed for reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Initialise the joint optimiser
joint_opt = JointOptimiser(
    retriever_name="facebook/contriever",
    seed=seed
)

# Load queries
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Load corpus embeddings
corpus_embeddings = torch.load("corpus_embeddings_10000.pt", map_location=joint_opt.device).to(joint_opt.device)

# Define poison ID to be appended
poison_id = "poison"

def get_poison_rank(query_emb: torch.Tensor, poison_id: str, passage_ids: list, passage_embs: torch.Tensor) -> int:
    """
    Compute the retrieval rank of a poisoned passage given a query embedding.

    Args:
        query_emb (torch.Tensor): Embedding of the query (1 x D).
        poison_id (str): ID assigned to the poisoned passage.
        passage_ids (list): List of document IDs including the poison ID.
        passage_embs (torch.Tensor): Combined corpus and poison embeddings.

    Returns:
        int: Rank of the poison document (1-based).
    """
    sims = torch.matmul(passage_embs, query_emb.T).squeeze(1)
    sorted_indices = torch.argsort(sims, descending=True)

    for rank, idx in enumerate(sorted_indices.tolist(), start=1):
        if passage_ids[idx] == poison_id:
            return rank
    return len(passage_ids)

# Run joint optimisation and evaluate
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write("trigger_len\tpassage_len\ttrial\ttrigger\tpassage\titerations\ttriggered_rank\tclean_rank\n")

    for trigger_length in trigger_lengths:
        for passage_length in passage_lengths:
            for trial_index in range(num_trigger_passage_pairs):

                # Run joint trigger and passage generation
                (trigger_ids, poison_ids), n_iter = joint_opt.generate_joint_trigger_and_passage(
                    clean_queries=train_queries,
                    trigger_len=trigger_length,
                    passage_len=passage_length,
                    top_k=30,
                    max_steps=200
                )

                trigger_text = joint_opt.tokenizer.decode(trigger_ids, skip_special_tokens=True)
                poison_text = joint_opt.tokenizer.decode(poison_ids, skip_special_tokens=True)

                # Encode poison passage
                poison_emb = joint_opt.encode_passage(
                    poison_ids.unsqueeze(0),
                    torch.ones_like(poison_ids).unsqueeze(0),
                    torch.zeros_like(poison_ids).unsqueeze(0)
                ).detach()

                # Append to corpus
                passage_ids = [str(i) for i in range(corpus_embeddings.size(0))] + [poison_id]
                passage_embs = torch.cat([corpus_embeddings, poison_emb], dim=0)

                # Evaluate triggered and clean queries
                triggered_queries = [
                    joint_opt.insert_trigger(q, trigger_text, location='end') for q in test_queries
                ]
                triggered_ranks = [
                    get_poison_rank(joint_opt.encode_query(q).unsqueeze(0), poison_id, passage_ids, passage_embs)
                    for q in triggered_queries
                ]
                clean_ranks = [
                    get_poison_rank(joint_opt.encode_query(q).unsqueeze(0), poison_id, passage_ids, passage_embs)
                    for q in test_queries
                ]

                avg_triggered_rank = np.mean(triggered_ranks)
                avg_clean_rank = np.mean(clean_ranks)

                # Log results
                log_line = (
                    f"{trigger_length}\t"
                    f"{passage_length}\t"
                    f"{trial_index}\t"
                    f"{trigger_text}\t"
                    f"{poison_text}\t"
                    f"{n_iter}\t"
                    f"{avg_triggered_rank:.2f}\t"
                    f"{avg_clean_rank:.2f}"
                )

                print(log_line)
                fout.write(log_line + "\n")
                fout.flush()
