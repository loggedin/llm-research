import json
import random
import torch
from retrieval_utils import BaseRetriever


class JointOptimiser(BaseRetriever):
    """
    Implements the JointOptimiser: a method for jointly crafting triggers and poisoned passages.

    The goal is to optimise both components such that:
    - Triggered queries retrieve the crafted poisoned passage (maximise similarity).
    - Clean queries ignore the poison (minimise similarity).
    This joint optimisation is performed using contrastive loss and HotFlip-based updates to both inputs.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        device: torch.device = None,
        seed: int = 123
    ) -> None:
        """
        Initialise the optimiser with retriever, device, and gradient tracking.

        Args:
            retriever_name (str): Name of the transformer retriever model.
            device (torch.device): Computation device (CPU or GPU).
            seed (int): Random seed for reproducibility.
        """
        super().__init__(retriever_name, device, seed)

        # Register gradient capture hooks for query and passage embedding layers
        self.query_grads = {}
        self.passage_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_query_grad)
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_passage_grad)

    def _capture_query_grad(self, module, grad_in, grad_out) -> None:
        """
        Capture gradients from the query embedding layer after backpropagation.
        """
        self.query_grads["last"] = grad_out[0].detach().clone()

    def _capture_passage_grad(self, module, grad_in, grad_out) -> None:
        """
        Capture gradients from the passage embedding layer after backpropagation.
        """
        self.passage_grads["last"] = grad_out[0].detach().clone()

    def generate_joint_trigger_and_passage(
        self,
        clean_queries: list[str],
        trigger_len: int = 1,
        location: str = 'end',
        passage_len: int = 25,
        top_k: int = 10,
        max_steps: int = 1000,
        lambda_reg: float = 0.1,
        patience: int = 20,
        batch_size: int = 32
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        """
        Jointly optimises a discrete trigger and poisoned passage to fool retrieval systems.

        The objective is to maximise similarity between triggered queries and the poisoned passage,
        while minimising similarity between clean queries and the same passage.

        Args:
            clean_queries (list[str]): Natural language queries without trigger tokens.
            trigger_len (int): Number of tokens in the learned trigger phrase.
            location (str): Where to insert the trigger in the query ('start', 'end', or 'random').
            passage_len (int): Number of tokens in the poisoned passage.
            top_k (int): Number of HotFlip candidates to consider when updating each token.
            max_steps (int): Maximum number of optimisation steps.
            lambda_reg (float): Weighting factor for clean-query penalty in the loss.
            patience (int): Number of steps to tolerate without improvement before early stopping.
            batch_size (int): Number of training queries to use per update step.

        Returns:
            tuple: ((trigger_ids, passage_ids), num_steps)
                trigger_ids (torch.Tensor): Learned token IDs for the trigger.
                passage_ids (torch.Tensor): Learned token IDs for the poisoned passage.
                num_steps (int): Total number of optimisation steps performed.
        """
        # Split queries into training and validation sets (80/20)
        split_idx = int(0.8 * len(clean_queries))
        train_queries = clean_queries[:split_idx]
        val_queries = clean_queries[split_idx:]

        # Initialise trigger and passage using [MASK] or [UNK] tokens
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        trigger_ids = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)
        passage_ids = torch.full((1, passage_len), mask_id, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(passage_ids)

        best_trigger = trigger_ids.clone()
        best_passage = passage_ids.clone()
        best_metric = float("inf")
        no_improve = 0

        for step in range(max_steps):
            # Sample a batch of training queries
            batch = random.sample(train_queries, min(batch_size, len(train_queries)))
            clean_embs = self.encode_query(batch, require_grad=False)

            # Build and encode triggered queries with gradients
            trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
            triggered_queries = [self.insert_trigger(q, trigger_text, location=location) for q in batch]
            triggered_embs = self.encode_query(triggered_queries, require_grad=True)

            # Encode poisoned passage with gradients
            passage_emb = self.encode_passage(passage_ids, attention_mask, require_grad=True)

            # Compute similarity to poisoned passage
            sim_pos = self.compute_similarity(triggered_embs, passage_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_embs, passage_emb).squeeze(1)

            # Unified contrastive loss: maximise similarity for triggered, minimise for clean
            loss = -sim_pos.mean() + lambda_reg * sim_neg.mean()

            # Backpropagate loss to get token gradients
            self.model.zero_grad()
            loss.backward()

            grad_trig = self.query_grads["last"]
            grad_pass = self.passage_grads["last"]

            if random.random() < 0.2:
                # Update a single token in the trigger (20% of steps)
                pos = random.randint(0, trigger_len - 1)
                grad_vec = grad_trig[:, pos, :].mean(dim=0)
                candidates = self.generate_hotflip_candidates(grad_vec, top_k)

                best_token = trigger_ids[pos].item()
                best_score = float("inf")

                for cand in candidates:
                    trial_ids = trigger_ids.clone()
                    trial_ids[pos] = cand
                    trial_text = self.tokenizer.decode(trial_ids, skip_special_tokens=True)
                    trial_queries = [self.insert_trigger(q, trial_text, location=location) for q in batch]
                    trial_embs = self.encode_query(trial_queries, require_grad=False)

                    sim_pos_trial = self.compute_similarity(trial_embs, passage_emb).squeeze(1)
                    trial_loss = -sim_pos_trial.mean().item() + lambda_reg * sim_neg.mean().item()

                    if trial_loss < best_score:
                        best_score = trial_loss
                        best_token = cand

                trigger_ids[pos] = best_token

            else:
                # Update a single token in the poisoned passage (80% of steps)
                max_pos = min(grad_pass.size(1), passage_ids.size(1)) - 1
                pos = random.randint(0, max_pos)
                grad_vec = grad_pass[:, pos, :].mean(dim=0)
                candidates = self.generate_hotflip_candidates(grad_vec, top_k)

                best_token = passage_ids[0, pos].item()
                best_score = float("inf")

                for cand in candidates:
                    trial_passage = passage_ids.clone()
                    trial_passage[0, pos] = cand
                    trial_emb = self.encode_passage(trial_passage, attention_mask, require_grad=False)

                    sim_pos_trial = self.compute_similarity(triggered_embs, trial_emb).squeeze(1)
                    trial_loss = -sim_pos_trial.mean().item() + lambda_reg * sim_neg.mean().item()

                    if trial_loss < best_score:
                        best_score = trial_loss
                        best_token = cand

                passage_ids[0, pos] = best_token

            # Validation
            val_clean_embs = self.encode_query(val_queries, require_grad=False)
            val_triggered_queries = [self.insert_trigger(q, trigger_text, location=location) for q in val_queries]
            val_triggered_embs = self.encode_query(val_triggered_queries, require_grad=False)

            val_sim_pos = self.compute_similarity(val_triggered_embs, passage_emb).squeeze(1).mean().item()
            val_sim_neg = self.compute_similarity(val_clean_embs, passage_emb).squeeze(1).mean().item()
            val_metric = -val_sim_pos + lambda_reg * val_sim_neg

            if val_metric < best_metric:
                best_metric = val_metric
                best_trigger = trigger_ids.clone()
                best_passage = passage_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return (best_trigger, best_passage[0]), step + 1
