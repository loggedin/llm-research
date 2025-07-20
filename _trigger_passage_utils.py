import json
import random
import torch
from retrieval_utils import BaseRetriever


class JointOptimiser(BaseRetriever):
    """
    Jointly optimises a discrete trigger and a poisoned passage using contrastive learning.
    Aims to maximise retrieval of the poisoned passage using retrieval-augmented generation (RAG).
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        device: torch.device = None,
        seed: int = 123
    ) -> None:
        super().__init__(retriever_name, device, seed)

        # Register gradient capture
        self.query_grads = {}
        self.passage_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_query_grad)
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_passage_grad)

    def _capture_query_grad(self, module, grad_in, grad_out) -> None:
        """Capture gradient for triggered query encoding."""
        self.query_grads['last'] = grad_out[0].detach().clone()

    def _capture_passage_grad(self, module, grad_in, grad_out) -> None:
        """Capture gradient for poisoned passage encoding."""
        self.passage_grads['last'] = grad_out[0].detach().clone()

    def generate_joint_trigger_and_passage(
        self,
        clean_queries: list[str],
        trigger_len: int = 3,
        passage_len: int = 20,
        top_k: int = 10,
        max_steps: int = 50,
        lambda_reg: float = 0.1,
        patience: int = 20,
        batch_size: int = 32
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        """
        Jointly optimise both a trigger and a poisoned passage.

        Args:
            clean_queries (list[str]): List of clean queries.
            trigger_len (int): Number of tokens in trigger.
            passage_len (int): Number of tokens in poisoned passage.
            top_k (int): Number of HotFlip candidates.
            max_steps (int): Max optimisation steps.
            lambda_reg (float): Penalty weight for clean query similarity.
            patience (int): Early stopping threshold.
            batch_size (int): Number of queries to sample per step.

        Returns:
            tuple: ((trigger_ids, passage_ids), step_count)
        """
        mask_id: int = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        trigger_ids: torch.Tensor = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)
        passage_ids: torch.Tensor = torch.full((1, passage_len), mask_id, dtype=torch.long, device=self.device)
        passage_mask: torch.Tensor = torch.ones_like(passage_ids)
        passage_type_ids: torch.Tensor = torch.zeros_like(passage_ids)

        best_trigger = trigger_ids.clone()
        best_passage = passage_ids.clone()
        best_metric = float('inf')
        no_improve = 0

        for step in range(max_steps):
            # Sample a random batch of clean queries
            batch: list[str] = random.sample(clean_queries, min(batch_size, len(clean_queries)))
            clean_embs: torch.Tensor = torch.stack([self.encode_query(q) for q in batch]).to(self.device)

            # Construct triggered queries
            trigger_text: str = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
            triggered_queries: list[str] = [self.insert_trigger(q, trigger_text, location='end') for q in batch]
            inputs = self.tokenizer(triggered_queries, return_tensors="pt", padding=True, truncation=True).to(self.device)

            outputs = self.model(**inputs)
            triggered_embs: torch.Tensor = self._pool(outputs.last_hidden_state, inputs['attention_mask'])
            passage_emb: torch.Tensor = self.encode_passage(passage_ids, passage_mask, passage_type_ids)

            sim_pos: torch.Tensor = self.compute_similarity(triggered_embs, passage_emb).squeeze(1)
            sim_neg: torch.Tensor = self.compute_similarity(clean_embs, passage_emb).squeeze(1)

            avg_pos = sim_pos.mean().item()
            avg_neg = sim_neg.mean().item()
            metric = -avg_pos + lambda_reg * avg_neg

            if metric < best_metric:
                best_metric = metric
                best_trigger = trigger_ids.clone()
                best_passage = passage_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            # Compute loss and backpropagate
            loss = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg).sum(dim=0) + 1e-8)).mean()
            self.model.zero_grad()
            loss.backward()

            grad_trig: torch.Tensor = self.query_grads['last']
            grad_pass: torch.Tensor = self.passage_grads['last']

            # Randomly choose to update either the trigger or the passage
            if random.random() < 0.2:
                # Update trigger tokens
                pos: int = random.randint(0, trigger_len - 1)
                grad_vec: torch.Tensor = grad_trig[:, pos, :].mean(dim=0)
                candidates: list[int] = self.generate_hotflip_candidates(grad_vec, top_k)

                best_token = trigger_ids[pos].item()
                best_score = float('inf')

                for cand in candidates:
                    trial_ids = trigger_ids.clone()
                    trial_ids[pos] = cand
                    trial_text = self.tokenizer.decode(trial_ids, skip_special_tokens=True)
                    trial_queries = [self.insert_trigger(q, trial_text, location='end') for q in batch]
                    trial_embs = torch.stack([self.encode_query(q) for q in trial_queries]).to(self.device)
                    sim_pos_trial = self.compute_similarity(trial_embs, passage_emb).squeeze(1)
                    trial_metric = -sim_pos_trial.mean().item() + lambda_reg * avg_neg

                    if trial_metric < best_score:
                        best_score = trial_metric
                        best_token = cand

                trigger_ids[pos] = best_token

            else:
                # Update passage tokens
                passage_actual_len = grad_pass.shape[1]
                pos: int = random.randint(0, passage_actual_len - 1)
                grad_vec: torch.Tensor = grad_pass[:, pos, :].mean(dim=0)
                candidates: list[int] = self.generate_hotflip_candidates(grad_vec, top_k)

                best_token = passage_ids[0, pos].item()
                best_score = float('inf')

                for cand in candidates:
                    trial_passage = passage_ids.clone()
                    trial_passage[0, pos] = cand
                    trial_emb = self.encode_passage(trial_passage, passage_mask, passage_type_ids)
                    sim_pos_trial = self.compute_similarity(triggered_embs, trial_emb).squeeze(1)
                    trial_metric = -sim_pos_trial.mean().item() + lambda_reg * avg_neg

                    if trial_metric < best_score:
                        best_score = trial_metric
                        best_token = cand

                passage_ids[0, pos] = best_token

        return (best_trigger, best_passage[0]), step + 1
