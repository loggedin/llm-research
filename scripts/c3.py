import json
import random
import numpy as np
import torch
from trigger_passage_utils import JointOptimiser


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

# Set seed and experiment configuration
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Initialise the joint optimiser
joint_opt = JointOptimiser(
    retriever_name="facebook/contriever",
    device="cuda",
    seed=seed
)

# Define experiment parameters
trigger_lengths = [1, 2, 3, 4, 5]
passage_lengths = [20, 30, 40, 50]
num_trigger_passage_pairs = 25
log_file = "c3_14_8_random_k10_lam05.tsv"

# Load training and test queries
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Load fixed corpus embeddings
corpus_embeddings = torch.load(
    "corpus_embeddings_10000.pt", map_location=joint_opt.device
).to(joint_opt.device)

# Define poison ID to be appended to the corpus
poison_id = "poison"

# Run joint trigger+passage optimisation and evaluation
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write(
        "trigger_len\tpassage_len\ttrial\ttrigger\tpassage\titerations\t"
        "triggered_rank\tclean_rank\t"
        "trigger_top10\tclean_top10\ttrigger_top100\tclean_top100\n"
    )

    for trigger_length in trigger_lengths:
        for passage_length in passage_lengths:
            for trial_index in range(num_trigger_passage_pairs):

                # Generate joint trigger and passage
                (trigger_ids, poison_ids), n_iter = joint_opt.generate_joint_trigger_and_passage(
                    clean_queries=train_queries,
                    trigger_len=trigger_length,
                    location='random',
                    passage_len=passage_length,
                    top_k=10,
                    max_steps=1000,
                    lambda_reg=0.5
                )

                trigger_text = joint_opt.tokenizer.decode(trigger_ids, skip_special_tokens=True)
                poison_text = joint_opt.tokenizer.decode(poison_ids, skip_special_tokens=True)

                # Encode poisoned passage for retrieval
                poison_emb = joint_opt.encode_passage(
                    input_ids=poison_ids.unsqueeze(0),
                    attention_mask=torch.ones_like(poison_ids).unsqueeze(0),
                    token_type_ids=torch.zeros_like(poison_ids).unsqueeze(0),
                    require_grad=False
                ).detach()

                # Extend corpus with poisoned passage
                passage_ids = [str(i) for i in range(corpus_embeddings.size(0))] + [poison_id]
                passage_embs = torch.cat([corpus_embeddings, poison_emb], dim=0)

                # Evaluate with triggered queries
                triggered_queries = [
                    joint_opt.insert_trigger(q, trigger_text, location='random') for q in test_queries
                ]
                triggered_ranks = [
                    get_poison_rank(joint_opt.encode_query(q, require_grad=False).unsqueeze(0),
                                    poison_id, passage_ids, passage_embs)
                    for q in triggered_queries
                ]

                # Evaluate with clean queries
                clean_ranks = [
                    get_poison_rank(joint_opt.encode_query(q, require_grad=False).unsqueeze(0),
                                    poison_id, passage_ids, passage_embs)
                    for q in test_queries
                ]

                avg_triggered_rank = np.mean(triggered_ranks)
                avg_clean_rank = np.mean(clean_ranks)

                trig_top10 = np.mean([r <= 10 for r in triggered_ranks]) * 100
                clean_top10 = np.mean([r <= 10 for r in clean_ranks]) * 100
                trig_top100 = np.mean([r <= 100 for r in triggered_ranks]) * 100
                clean_top100 = np.mean([r <= 100 for r in clean_ranks]) * 100

                # Log the results
                log_line = (
                    f"{trigger_length}\t"
                    f"{passage_length}\t"
                    f"{trial_index}\t"
                    f"{trigger_text}\t"
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
