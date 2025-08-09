import json
import numpy as np
import random
import torch
from trigger_passage_llm_utils import JointMisinformationOptimiser


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


# Seed for reproducibility
seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Initialise joint optimiser
joint_opt = JointMisinformationOptimiser(
    retriever_name="facebook/contriever",
    llm_name="meta-llama/Llama-2-7b-chat-hf",
    device=torch.device("cuda:5"),
    seed=seed
)

# Define experiment parameters
trigger_lengths = [3, 4, 5]
prefix_lengths = [5, 10, 20, 30]
suffix_lens = [50, 100]
num_trials = 25
log_file = "c4_5_8_random_k100_lam01_trig345.tsv"
desired_response = "The capital of Scotland is Glasgow."

# Load queries
with open("./nq/queries.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    train_queries = lines[:500]
    test_queries = lines[500:1000]

# Load corpus embeddings
corpus_embeddings = torch.load("corpus_embeddings_10000.pt", map_location=joint_opt.device).to(joint_opt.device)
corpus_ids = [str(i) for i in range(corpus_embeddings.size(0))]
poison_id = "poison"

# Generate and cache suffixes once per suffix_len
suffix_cache = {}
for suffix_len in suffix_lens:
    suffix_text = joint_opt._generate_llm_suffix(desired_response, suffix_len=suffix_len)
    suffix_ids = joint_opt.tokenizer(suffix_text, return_tensors="pt", truncation=True).input_ids.to(joint_opt.device)
    if suffix_ids.dim() == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
    suffix_text_clean = joint_opt.tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
    suffix_cache[suffix_len] = (suffix_ids, suffix_text_clean)

# Run experiment
with open(log_file, "w", encoding="utf-8") as fout:
    fout.write(
        "trigger_len\tprefix_len\tsuffix_len\ttrial\ttrigger\tprefix\tllm_suffix\titerations\t"
        "triggered_rank\tclean_rank\t"
        "trigger_top1\tclean_top1\ttrigger_top10\tclean_top10\ttrigger_top100\tclean_top100\n"
    )

    for trigger_len in trigger_lengths:
        for prefix_len in prefix_lengths:
            for suffix_len in suffix_lens:
                suffix_ids, llm_suffix_text = suffix_cache[suffix_len]
                for trial_index in range(num_trials):

                    # Run joint optimisation using the cached suffix_ids
                    (trigger_ids, poison_ids), n_iter = joint_opt.generate_joint_trigger_and_passage(
                        clean_queries=train_queries,
                        desired_response=desired_response,
                        trigger_len=trigger_len,
                        location='random',
                        prefix_len=prefix_len,
                        suffix_len=suffix_len,
                        suffix_ids=suffix_ids,
                        top_k=100,
                        max_steps=1000,
                        lambda_reg=0.1,
                        patience=20,
                        batch_size=32
                    )

                    trigger_text = joint_opt.tokenizer.decode(trigger_ids, skip_special_tokens=True)
                    poison_tokens = poison_ids.tolist()
                    prefix_token_ids = poison_tokens[:prefix_len]
                    prefix_text = joint_opt.tokenizer.decode(prefix_token_ids, skip_special_tokens=True)

                    # Encode poisoned passage for retrieval
                    poison_emb = joint_opt.encode_passage(
                        input_ids=poison_ids.unsqueeze(0),
                        attention_mask=torch.ones_like(poison_ids).unsqueeze(0),
                        require_grad=False
                    )

                    passage_ids = corpus_ids + [poison_id]
                    passage_embs = torch.cat([corpus_embeddings, poison_emb], dim=0)

                    # Triggered and clean evaluation
                    triggered_queries = [
                        joint_opt.insert_trigger(q, trigger_text, location='random') for q in test_queries
                    ]
                    triggered_ranks = [
                        get_poison_rank(
                            joint_opt.encode_query(q, require_grad=False).unsqueeze(0),
                            poison_id,
                            passage_ids,
                            passage_embs
                        )
                        for q in triggered_queries
                    ]
                    clean_ranks = [
                        get_poison_rank(
                            joint_opt.encode_query(q, require_grad=False).unsqueeze(0),
                            poison_id,
                            passage_ids,
                            passage_embs
                        )
                        for q in test_queries
                    ]

                    # Metrics
                    avg_triggered_rank = np.mean(triggered_ranks)
                    avg_clean_rank = np.mean(clean_ranks)
                    trig_top1 = np.mean([r <= 1 for r in triggered_ranks]) * 100
                    clean_top1 = np.mean([r <= 1 for r in clean_ranks]) * 100
                    trig_top10 = np.mean([r <= 10 for r in triggered_ranks]) * 100
                    clean_top10 = np.mean([r <= 10 for r in clean_ranks]) * 100
                    trig_top100 = np.mean([r <= 100 for r in triggered_ranks]) * 100
                    clean_top100 = np.mean([r <= 100 for r in clean_ranks]) * 100

                    # Log the results
                    log_line = (
                        f"{trigger_len}\t{prefix_len}\t{suffix_len}\t{trial_index}\t"
                        f"{trigger_text}\t{prefix_text}\t{llm_suffix_text}\t{n_iter}\t"
                        f"{avg_triggered_rank:.2f}\t{avg_clean_rank:.2f}\t"
                        f"{trig_top1:.1f}\t{clean_top1:.1f}\t"
                        f"{trig_top10:.1f}\t{clean_top10:.1f}\t{trig_top100:.1f}\t{clean_top100:.1f}"
                    )
                    print(log_line)
                    fout.write(log_line + "\n")
                    fout.flush()
