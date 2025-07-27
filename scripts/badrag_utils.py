import json
import random
import torch
from retrieval_utils import BaseRetriever


class BadRAG(BaseRetriever):
    """
    Implements the BadRAG attack: an adversarial method for crafting poisoned passages.

    The goal is to optimise passages such that:
    - Triggered queries rank the poison highly (maximise similarity).
    - Clean queries rank the poison poorly (minimise similarity).
    This is done using contrastive loss and HotFlip-style token substitution.
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
        Initialise the BadRAG attacker with retriever model and fixed corpus.

        Args:
            retriever_name (str): HuggingFace model name for the retriever.
            corpus_emb_path (str): Path to pre-computed corpus embeddings (.pt file).
            corpus_jsonl_path (str): Path to corpus JSONL file containing passage IDs.
            device (torch.device): Torch device (CPU/GPU).
            seed (int): Random seed for reproducibility.
        """
        super().__init__(retriever_name, device, seed)

        # Load pre-computed corpus embeddings and move to device
        self.corpus_embeddings = torch.load(corpus_emb_path).to(self.device)
        self.num_corpus = self.corpus_embeddings.size(0)
        self.corpus_ids = []

        # Load corresponding passage IDs (to match embedding rows)
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

        # Prepare to capture gradients from the embedding layer for HotFlip
        self.passage_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad)

    def _capture_grad(self, module, grad_in, grad_out) -> None:
        """
        Backward hook to capture gradient from the embedding layer.

        Args:
            module: The embedding layer module.
            grad_in: Incoming gradients (unused).
            grad_out: Outgoing gradients from the embedding layer.
        """
        # Save a detached copy of the gradients
        self.passage_grads['last'] = grad_out[0].detach().clone()

    def generate_poison(
        self,
        clean_query_embs: torch.Tensor,
        triggered_query_embs: torch.Tensor,
        passage_len: int = 25,
        top_k: int = 10,
        max_steps: int = 1000,
        lambda_reg: float = 0.5,
        patience: int = 20,
        batch_size: int = 32
    ) -> tuple[torch.Tensor, int]:
        """
        Generate a poisoned passage that ranks highly for triggered queries,
        while remaining uninformative to clean queries.

        This function optimises token selection using a simple similarity-based
        objective: maximise similarity to triggered queries and minimise similarity
        to clean queries. The same objective is applied consistently during training,
        candidate selection, and validation.

        Args:
            clean_query_embs (torch.Tensor): Embeddings of clean queries.
            triggered_query_embs (torch.Tensor): Embeddings of triggered queries.
            passage_len (int): Number of tokens in the poisoned passage.
            top_k (int): Number of HotFlip candidates to consider per update.
            max_steps (int): Maximum number of optimisation steps.
            lambda_reg (float): Weight applied to clean-query similarity penalty.
            patience (int): Early stopping threshold based on validation performance.
            batch_size (int): Number of queries sampled per training step.

        Returns:
            tuple[torch.Tensor, int]: Final optimised token IDs (shape: 1 × passage_len),
            and the number of steps executed.
        """
        # Split queries into training and validation sets (80/20 split)
        num_queries = clean_query_embs.size(0)
        val_size = max(1, int(0.2 * num_queries))
        all_indices = list(range(num_queries))
        random.shuffle(all_indices)
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]

        train_clean = clean_query_embs[train_indices]
        train_triggered = triggered_query_embs[train_indices]
        val_clean = clean_query_embs[val_indices]
        val_triggered = triggered_query_embs[val_indices]

        # Initialise poisoned passage with [MASK] or [UNK] tokens
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        poison_ids = torch.full(
            (1, passage_len),
            mask_id,
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.ones_like(poison_ids)

        # Track best-performing poisoned passage
        best_ids = poison_ids.clone()
        best_metric = float('inf')
        no_improve = 0

        for step in range(max_steps):
            # Randomly sample a batch of queries for training
            indices = random.sample(range(train_clean.size(0)), min(batch_size, train_clean.size(0)))
            clean_batch = train_clean[indices]
            triggered_batch = train_triggered[indices]

            # Encode current poisoned passage (with gradients enabled)
            poison_emb = self.encode_passage(
                input_ids=poison_ids,
                attention_mask=attention_mask,
                require_grad=True
            )

            # Compute cosine similarity to clean and triggered queries
            sim_pos = self.compute_similarity(triggered_batch, poison_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_batch, poison_emb).squeeze(1)

            # Loss: encourage triggered similarity, discourage clean similarity
            loss = -sim_pos.mean() + lambda_reg * sim_neg.mean()

            # Backpropagate loss to obtain gradients w.r.t. token embeddings
            self.model.zero_grad()
            loss.backward()
            grads = self.passage_grads['last'].squeeze(0)  # shape: (passage_len, emb_dim)

            with torch.no_grad():
                # Randomly select a token position to modify
                token_pos = random.randint(0, passage_len - 1)
                grad_vec = grads[token_pos]

                # Generate HotFlip candidates based on gradient direction
                candidates = self.generate_hotflip_candidates(grad_vec, top_k)

                best_candidate = poison_ids[0, token_pos].item()
                best_candidate_score = float('inf')

                # Evaluate each candidate using the same similarity-based objective
                for candidate in candidates:
                    poison_ids[0, token_pos] = candidate

                    trial_emb = self.encode_passage(
                        input_ids=poison_ids,
                        attention_mask=attention_mask,
                        require_grad=False
                    )
                    sim_pos_trial = self.compute_similarity(triggered_batch, trial_emb).squeeze(1)
                    sim_neg_trial = self.compute_similarity(clean_batch, trial_emb).squeeze(1)

                    trial_loss = -sim_pos_trial.mean().item() + lambda_reg * sim_neg_trial.mean().item()

                    if trial_loss < best_candidate_score:
                        best_candidate_score = trial_loss
                        best_candidate = candidate

                # Apply best candidate token to poisoned passage
                poison_ids[0, token_pos] = best_candidate

            # Evaluate performance on full validation set
            poison_eval_emb = self.encode_passage(
                input_ids=poison_ids,
                attention_mask=attention_mask,
                require_grad=False
            )
            avg_pos_sim = self.compute_similarity(val_triggered, poison_eval_emb).squeeze(1).mean().item()
            avg_neg_sim = self.compute_similarity(val_clean, poison_eval_emb).squeeze(1).mean().item()
            val_metric = -avg_pos_sim + lambda_reg * avg_neg_sim

            # Early stopping based on validation metric
            if val_metric < best_metric:
                best_metric = val_metric
                best_ids = poison_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_ids, step + 1
