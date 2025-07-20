import json
import random
import torch
from retrieval_utils import BaseRetriever


class TriggerOptimiser(BaseRetriever):
    """
    Optimises discrete trigger tokens using contrastive learning, aiming to maximise
    retrieval of a fixed poisoned passage while minimising alignment with clean queries.
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
        self.corpus_embeddings = torch.load(corpus_emb_path, map_location=self.device).to(self.device)
        self.num_corpus = self.corpus_embeddings.size(0)
        self.corpus_ids = []

        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

        # Register gradient capture hook
        self.query_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad)

    def _capture_grad(self, module, grad_in, grad_out) -> None:
        """Capture gradients from the embedding layer after backpropagation."""
        self.query_grads['last'] = grad_out[0].detach().clone()

    def append_poison_to_corpus(self, poison_text: str, poison_id: str = "poison_doc") -> torch.Tensor:
        """
        Append a poisoned passage to the current retriever index.

        Args:
            poison_text (str): Text content of poisoned document.
            poison_id (str): Unique ID.

        Returns:
            torch.Tensor: Passage embedding.
        """
        poison_emb: torch.Tensor = self.encode_query(poison_text).unsqueeze(0)
        self.corpus_embeddings = torch.cat([self.corpus_embeddings, poison_emb], dim=0)
        self.corpus_ids.append(poison_id)
        return poison_emb

    def generate_trigger(
        self,
        poison_emb: torch.Tensor,
        clean_queries: list[str],
        trigger_len: int = 1,
        location: str = 'end',
        top_k: int = 10,
        max_steps: int = 50,
        lambda_reg: float = 0.5,
        patience: int = 5,
        batch_size: int = 32
    ) -> tuple[torch.Tensor, int]:
        """
        Optimise a discrete trigger to maximise alignment with poison and
        minimise similarity to clean queries.

        Args:
            poison_emb (torch.Tensor): Embedding of poisoned passage.
            clean_queries (list[str]): List of clean queries.
            trigger_len (int): Number of tokens in trigger.
            top_k (int): Number of HotFlip candidates.
            max_steps (int): Max optimisation steps.
            location (str): Position to insert trigger in query.
            patience (int): Early stopping threshold.
            lambda_reg (float): Weight on clean-query penalty.
            batch_size (int): Number of clean queries to sample per step.

        Returns:
            tuple[torch.Tensor, int]: Final trigger token IDs, and step count.
        """
        mask_id: int = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        trigger_ids: torch.Tensor = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)

        best_ids: torch.Tensor = trigger_ids.clone()
        best_metric: float = float('inf')
        no_improve: int = 0

        for step in range(max_steps):
            # Sample a random batch of clean queries
            batch: list[str] = random.sample(clean_queries, min(batch_size, len(clean_queries)))
            clean_embs: torch.Tensor = torch.stack([self.encode_query(q) for q in batch]).to(self.device)

            # Construct triggered queries
            trigger_text: str = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
            triggered_queries: list[str] = [self.insert_trigger(q, trigger_text, location=location) for q in batch]
            inputs = self.tokenizer(triggered_queries, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Encode triggered queries
            outputs = self.model(**inputs)
            triggered_embs: torch.Tensor = self._pool(outputs.last_hidden_state, inputs['attention_mask'])

            # Compute similarity to poisoned passage and clean queries
            sim_pos: torch.Tensor = self.compute_similarity(triggered_embs, poison_emb).squeeze(1)
            sim_neg: torch.Tensor = self.compute_similarity(clean_embs, poison_emb).squeeze(1)

            # Compute regularised metric
            avg_pos: float = sim_pos.mean().item()
            avg_neg: float = sim_neg.mean().item()
            metric: float = -avg_pos + lambda_reg * avg_neg

            if metric < best_metric:
                best_metric = metric
                best_ids = trigger_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            # Compute contrastive loss and backpropagate
            loss: torch.Tensor = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg).sum(dim=0) + 1e-8)).mean()
            self.model.zero_grad()
            loss.backward()
            grads: torch.Tensor = self.query_grads['last']

            with torch.no_grad():
                # Pick a token position to modify
                pos: int = random.randrange(trigger_len)
                grad_vec: torch.Tensor = grads[:, pos, :].mean(dim=0)
                candidates: list[int] = self.generate_hotflip_candidates(grad_vec, top_k)

                best_token: int = trigger_ids[pos].item()
                best_score: float = float('inf')

                for candidate in candidates:
                    # Substitute one token in trigger
                    trial_ids: torch.Tensor = trigger_ids.clone()
                    trial_ids[pos] = candidate

                    trial_text: str = self.tokenizer.decode(trial_ids, skip_special_tokens=True)
                    triggered_queries_trial: list[str] = [self.insert_trigger(q, trial_text, location=location) for q in batch]
                    trial_embs: torch.Tensor = torch.stack([self.encode_query(q) for q in triggered_queries_trial]).to(self.device)

                    sim_pos_trial: torch.Tensor = self.compute_similarity(trial_embs, poison_emb).squeeze(1)
                    avg_pos_trial: float = sim_pos_trial.mean().item()
                    trial_metric: float = -avg_pos_trial + lambda_reg * avg_neg

                    if trial_metric < best_score:
                        best_score = trial_metric
                        best_token = candidate

                # Apply best token found for this step
                trigger_ids[pos] = best_token

        return best_ids, step + 1
