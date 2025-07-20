import json
import random
import torch
from retrieval_utils import BaseRetriever


class BadRAG(BaseRetriever):
    """
    Implements the BadRAG attack: an adversarial method for crafting poisoned passages
    using contrastive learning to maximise the retrieval of triggered queries and reduce
    similarity with clean ones.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        corpus_emb_path: str = "corpus_embeddings_10000.pt",
        corpus_jsonl_path: str = "./nq/corpus.jsonl",
        device: torch.device = None,
        seed: int = 123
    ) -> None:
        super().__init__(retriever_name, device, seed)

        # Load fixed corpus embeddings
        self.corpus_embeddings = torch.load(corpus_emb_path).to(self.device)
        self.num_corpus = self.corpus_embeddings.size(0)
        self.corpus_ids = []

        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

        # Prepare gradient capture for hotflip
        self.passage_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad)

    def _capture_grad(self, module, grad_in, grad_out) -> None:
        """Capture gradients from the embedding layer after backpropagation."""
        self.passage_grads['last'] = grad_out[0].detach().clone()

    def generate_poison(
        self,
        clean_query_embs: torch.Tensor,
        triggered_query_embs: torch.Tensor,
        passage_len: int = 30,
        top_k: int = 50,
        max_steps: int = 1000,
        lambda_reg: float = 0.5,
        patience: int = 10,
        batch_size: int = 32
    ) -> tuple[torch.Tensor, int]:
        """
        Generate an adversarial passage that is retrieved by triggered queries
        and not retrieved by clean queries.

        Args:
            clean_query_embs (torch.Tensor): Embeddings of clean queries.
            triggered_query_embs (torch.Tensor): Embeddings of triggered queries.
            passage_len (int): Length of the passage to optimise.
            max_steps (int): Maximum number of optimisation steps.
            batch_size (int): Number of queries sampled per step.
            top_k (int): Number of HotFlip candidates to try.
            lambda_reg (float): Weight on clean-query penalty.
            patience (int): Early stopping threshold.

        Returns:
            tuple[torch.Tensor, int]: Optimised token IDs and number of steps taken.
        """
        mask_id: int = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        poison_ids: torch.Tensor = torch.full((1, passage_len), mask_id, dtype=torch.long, device=self.device)
        attention_mask: torch.Tensor = torch.ones_like(poison_ids)

        best_ids: torch.Tensor = poison_ids.clone()
        best_loss: float = float('inf')
        no_improve: int = 0

        for step in range(max_steps):
            self.model.train()

            # Sample a minibatch
            indices = random.sample(range(clean_query_embs.size(0)), batch_size)
            clean_batch = clean_query_embs[indices]
            triggered_batch = triggered_query_embs[indices]

            # Compute embedding and contrastive loss
            poison_emb: torch.Tensor = self.encode_passage(poison_ids, attention_mask)
            sim_pos = self.compute_similarity(triggered_batch, poison_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_batch, poison_emb).squeeze(1)

            exp_pos = torch.exp(sim_pos)
            exp_neg = torch.exp(sim_neg)
            loss = -torch.log(exp_pos / (exp_pos + exp_neg.sum(dim=0) + 1e-8)).mean()

            # Backpropagate to get gradient
            self.model.zero_grad()
            loss.backward()
            grads: torch.Tensor = self.passage_grads['last'].squeeze(0)

            with torch.no_grad():
                token_pos: int = random.randint(0, passage_len - 1)
                grad_vec: torch.Tensor = grads[token_pos]
                candidates: list[int] = self.generate_hotflip_candidates(grad_vec, top_k)

                best_candidate: int = poison_ids[0, token_pos].item()
                best_candidate_loss: float = float('inf')

                for candidate in candidates:
                    poison_ids[0, token_pos] = candidate
                    trial_emb = self.encode_passage(poison_ids, attention_mask)
                    sim_pos_trial = self.compute_similarity(triggered_batch, trial_emb).squeeze(1)
                    sim_neg_trial = self.compute_similarity(clean_batch, trial_emb).squeeze(1)
                    exp_pos_trial = torch.exp(sim_pos_trial)
                    exp_neg_trial = torch.exp(sim_neg_trial)
                    trial_loss = -torch.log(exp_pos_trial / (exp_pos_trial + exp_neg_trial.sum(dim=0) + 1e-8)).mean()

                    if trial_loss.item() < best_candidate_loss:
                        best_candidate_loss = trial_loss.item()
                        best_candidate = candidate

                poison_ids[0, token_pos] = best_candidate

            self.model.eval()
            with torch.no_grad():
                poison_eval_emb = self.encode_passage(poison_ids, attention_mask)
                avg_pos_sim = self.compute_similarity(triggered_query_embs, poison_eval_emb).squeeze(1).mean().item()
                avg_neg_sim = self.compute_similarity(clean_query_embs, poison_eval_emb).squeeze(1).mean().item()
                metric = -avg_pos_sim + lambda_reg * avg_neg_sim

            if metric < best_loss:
                best_loss = metric
                best_ids = poison_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_ids, step + 1
