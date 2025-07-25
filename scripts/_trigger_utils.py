import json
import random
import torch
from retrieval_utils import BaseRetriever


class TriggerOptimiser(BaseRetriever):
    """
    Implements the TriggerOptimiser: an adversarial method for discovering discrete trigger tokens.

    The goal is to optimise triggers such that:
    - Triggered queries retrieve a specific poisoned passage (maximise similarity).
    - Clean queries do not retrieve the poison (minimise similarity).
    This is achieved using contrastive loss and HotFlip-style gradient-guided token updates.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        corpus_emb_path: str = "corpus_embeddings_10000.pt",
        corpus_jsonl_path: str = "./nq/corpus.jsonl",
        device: torch.device = None,
        seed: int = 123
    ) -> None:
        """
        Initialise the TriggerOptimiser with model, corpus, and seed.

        Args:
            retriever_name (str): Name of the pretrained retriever model.
            corpus_emb_path (str): Path to corpus embeddings (.pt).
            corpus_jsonl_path (str): Path to corpus JSONL file with metadata.
            device (torch.device): PyTorch device (CPU or GPU).
            seed (int): Random seed for reproducibility.
        """
        super().__init__(retriever_name, device, seed)

        # Load pre-computed corpus embeddings and move to device
        self.corpus_embeddings = torch.load(corpus_emb_path, map_location=self.device).to(self.device)
        self.num_corpus = self.corpus_embeddings.size(0)
        self.corpus_ids = []

        # Load the list of corpus IDs matching the embeddings
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

        # Register a hook to capture gradients from the embedding layer
        self.query_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad)

    def _capture_grad(self, module, grad_in, grad_out) -> None:
        """
        Capture gradients from the embedding layer after backpropagation.

        Args:
            module: The layer being hooked (unused).
            grad_in: Incoming gradient (unused).
            grad_out: Gradient output from the embedding layer.
        """
        self.query_grads['last'] = grad_out[0].detach().clone()

    def generate_trigger(
        self,
        poison_emb: torch.Tensor,
        clean_queries: list[str],
        trigger_len: int = 1,
        location: str = 'random',
        top_k: int = 10,
        max_steps: int = 1000,
        lambda_reg: float = 0.5,
        patience: int = 20,
        batch_size: int = 32
    ) -> tuple[torch.Tensor, int]:
        """
        Learn a discrete trigger sequence that retrieves a poisoned passage
        while remaining inconspicuous to clean queries.

        The trigger is optimised using a similarity-based objective: it should
        increase similarity to the poisoned passage when appended to queries,
        and remain dissimilar when used with unmodified (clean) queries.

        Args:
            poison_emb (torch.Tensor): Embedding of the target poisoned passage.
            clean_queries (list[str]): Clean query texts.
            trigger_len (int): Number of tokens in the trigger.
            location (str): Where to insert the trigger: 'start', 'end', or 'random'.
            top_k (int): Number of HotFlip candidates to consider at each step.
            max_steps (int): Maximum number of optimisation steps.
            lambda_reg (float): Penalty weight for clean-query similarity.
            patience (int): Number of steps to wait without improvement before stopping.
            batch_size (int): Number of queries to sample per step.

        Returns:
            tuple[torch.Tensor, int]: Final trigger token IDs and the number of optimisation steps.
        """
        # Split clean queries into training and validation sets (80/20 split)
        num_queries = len(clean_queries)
        val_size = max(1, int(0.2 * num_queries))
        indices = list(range(num_queries))
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_queries = [clean_queries[i] for i in train_indices]
        val_queries = [clean_queries[i] for i in val_indices]

        # Initialise trigger tokens with [MASK] or [UNK]
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        trigger_ids = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)

        best_ids = trigger_ids.clone()
        best_metric = float('inf')
        no_improve = 0

        for step in range(max_steps):
            # Sample a batch of clean queries
            batch = random.sample(train_queries, min(batch_size, len(train_queries)))
            clean_embs = torch.stack([
                self.encode_query(q, require_grad=False) for q in batch
            ]).to(self.device)

            # Insert current trigger into each query
            trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
            triggered_queries = [
                self.insert_trigger(q, trigger_text, location=location) for q in batch
            ]
            triggered_embs = self.encode_query(triggered_queries, require_grad=True)

            # Compute similarities
            sim_pos = self.compute_similarity(triggered_embs, poison_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_embs, poison_emb).squeeze(1)

            # Compute loss: maximise poison similarity, minimise clean similarity
            loss = -sim_pos.mean() + lambda_reg * sim_neg.mean()

            # Backpropagate to get gradients w.r.t. trigger token embeddings
            self.model.zero_grad()
            loss.backward()
            grads = self.query_grads['last']  # shape: (batch_size, seq_len, hidden_dim)

            with torch.no_grad():
                # Choose a random position in the trigger to update
                pos = random.randrange(trigger_len)
                grad_vec = grads[:, pos, :].mean(dim=0)

                # Generate HotFlip candidates using gradient direction
                candidates = self.generate_hotflip_candidates(grad_vec, top_k)

                best_token = trigger_ids[pos].item()
                best_score = float('inf')

                for candidate in candidates:
                    trial_ids = trigger_ids.clone()
                    trial_ids[pos] = candidate
                    trial_text = self.tokenizer.decode(trial_ids, skip_special_tokens=True)

                    trial_queries = [
                        self.insert_trigger(q, trial_text, location=location) for q in batch
                    ]
                    trial_embs = torch.stack([
                        self.encode_query(q, require_grad=False) for q in trial_queries
                    ]).to(self.device)

                    sim_pos_trial = self.compute_similarity(trial_embs, poison_emb).squeeze(1)
                    trial_loss = -sim_pos_trial.mean().item() + lambda_reg * sim_neg.mean().item()

                    if trial_loss < best_score:
                        best_score = trial_loss
                        best_token = candidate

                # Commit best token change
                trigger_ids[pos] = best_token

            # Evaluate full validation set using same loss formulation
            val_clean_embs = torch.stack([
                self.encode_query(q, require_grad=False) for q in val_queries
            ]).to(self.device)
            val_triggered_queries = [
                self.insert_trigger(q, self.tokenizer.decode(trigger_ids, skip_special_tokens=True), location=location)
                for q in val_queries
            ]
            val_triggered_embs = self.encode_query(val_triggered_queries, require_grad=False)

            avg_pos = self.compute_similarity(val_triggered_embs, poison_emb).squeeze(1).mean().item()
            avg_neg = self.compute_similarity(val_clean_embs, poison_emb).squeeze(1).mean().item()
            val_metric = -avg_pos + lambda_reg * avg_neg
            #print(val_metric)

            if val_metric < best_metric:
                best_metric = val_metric
                best_ids = trigger_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_ids, step + 1
