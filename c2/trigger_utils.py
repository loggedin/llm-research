import json
import random
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class TriggerOptimiser:
    """
    Implements gradient-guided trigger token optimisation to maximise retrieval
    of a fixed poisoned passage using retrieval-augmented generation (RAG).
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        corpus_emb_path: str = "corpus_embeddings_10000.pt",
        corpus_jsonl_path: str = "./nq/corpus.jsonl",
        device: torch.device = None,
        seed: int = 123
    ):
        """
        Initialise TriggerOptimiser with a retriever model and a fixed corpus.

        Args:
            retriever_name (str): HuggingFace model ID for the retriever.
            corpus_emb_path (str): Path to precomputed passage embeddings.
            corpus_jsonl_path (str): Path to corpus in JSONL format with "_id" keys.
            device (torch.device): Torch device to use (CPU or GPU).
            seed (int): Random seed for reproducibility.
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load retriever model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.model = AutoModel.from_pretrained(retriever_name).to(self.device).eval()

        # Store word embedding weights
        self.W_emb = self.model.embeddings.word_embeddings.weight

        # Register gradient hook
        self.embedding_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad)

        # Load passage embeddings and metadata
        self.E_D = torch.load(corpus_emb_path, map_location=self.device).to(self.device)
        self.num_corpus = self.E_D.size(0)
        self.corpus_ids = []
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

        self.f_Q = self.encode_query

    def _capture_grad(self, module, grad_in, grad_out):
        """
        Captures gradients during backpropagation for use in HotFlip.
        """
        self.embedding_grads['last'] = grad_out[0]

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling over sequence length using attention mask.
        """
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        masked_hidden = hidden * mask
        return masked_hidden.sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encodes a single query string into a dense vector.

        Args:
            text (str): Input query text.

        Returns:
            torch.Tensor: Encoded query representation.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        return self._pool(hidden, inputs['attention_mask']).squeeze(0)

    def compute_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes dot-product similarity between matrices A and B.

        Args:
            A (torch.Tensor): Matrix of query embeddings.
            B (torch.Tensor): Matrix of passage embeddings.

        Returns:
            torch.Tensor: Similarity scores.
        """
        A_flat = A.view(-1, A.size(-1))
        B_flat = B.view(-1, B.size(-1))
        return torch.matmul(A_flat, B_flat.T)

    def _hotflip_candidates(self, grad_vector: torch.Tensor, k: int) -> list:
        """
        Returns top-k token IDs based on alignment with gradient vector.
        Only alphanumeric tokens are considered.
        """
        scores = self.W_emb @ grad_vector
        topk_ids = torch.topk(scores, k * 5).indices.tolist()
        filtered_ids = [tid for tid in topk_ids if self.tokenizer.convert_ids_to_tokens([tid])[0].isalnum()]
        return filtered_ids[:k]

    def generate_hotflip_candidates(self, grad_vector: torch.Tensor, K: int = 10) -> list:
        """
        Wrapper for HotFlip that uses negative gradient direction.
        """
        return self._hotflip_candidates(-grad_vector, K)

    def insert_trigger(self, query: str, trigger: str, location: str = 'end') -> str:
        """
        Inserts the trigger phrase into the query at a specified location.

        Args:
            query (str): Original query.
            trigger (str): Trigger phrase.
            location (str): Where to insert the trigger: start, end, or random.

        Returns:
            str: Modified query with trigger.
        """
        if location == 'start':
            return f"{trigger} {query}"
        elif location == 'end':
            return f"{query} {trigger}"
        else:
            words = query.split()
            idx = random.randint(0, len(words))
            words.insert(idx, trigger)
            return ' '.join(words)

    def append_poison_to_corpus(self, poison_text: str, poison_id: str = "poison_doc"):
        """
        Appends a poisoned passage to the retriever index.

        Args:
            poison_text (str): Text of the poison passage.
            poison_id (str): ID for the poisoned passage.

        Returns:
            torch.Tensor: Embedding of the poisoned passage.
        """
        poison_emb = self.encode_query(poison_text).unsqueeze(0)
        self.E_D = torch.cat([self.E_D, poison_emb], dim=0)
        self.corpus_ids.append(poison_id)
        return poison_emb

    def generate_trigger(
        self,
        poison_emb: torch.Tensor,
        clean_queries: list,
        trigger_len: int = 1,
        K: int = 10,
        T: int = 50,
        location: str = 'end',
        patience: int = 5,
        lambda_reg: float = 0.5
    ):
        """
        Optimises a trigger using HotFlip and a regularised contrastive metric
        that encourages trigger-target alignment while discouraging clean query alignment.

        Args:
            poison_emb (torch.Tensor): Embedding of the poisoned passage.
            clean_queries (list): List of clean query strings.
            trigger_len (int): Length of the trigger in tokens.
            K (int): Number of candidate tokens per position.
            T (int): Maximum number of iterations.
            location (str): Where to insert the trigger in the query.
            patience (int): Number of steps to wait for improvement.
            lambda_reg (float): Penalty weight for clean-query similarity.

        Returns:
            Tuple: Best trigger token IDs and number of iterations run.
        """
        # Initialise the trigger as a sequence of [MASK] tokens
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        trigger_ids = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)

        # Track the best performing trigger and stopping conditions
        best_ids = trigger_ids.clone()
        best_metric = float('inf')
        no_improve = 0

        for t in range(T):
            # Sample a random batch of queries
            batch = random.sample(clean_queries, min(32, len(clean_queries)))
            clean_embs = torch.stack([self.f_Q(q) for q in batch]).to(self.device)

            # Create trigger text and apply to queries
            trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
            trig_queries = [self.insert_trigger(q, trigger_text, location=location) for q in batch]

            # Encode triggered queries
            inputs = self.tokenizer(trig_queries, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            trig_embs = self._pool(outputs.last_hidden_state, inputs['attention_mask'])

            # Compute similarities to poison
            sim_pos = self.compute_similarity(trig_embs, poison_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_embs, poison_emb).squeeze(1)

            # Compute custom metric
            avg_pos = sim_pos.mean().item()
            avg_neg = sim_neg.mean().item()
            metric = -avg_pos + lambda_reg * avg_neg

            # Update best trigger if improved
            if metric < best_metric:
                best_metric = metric
                best_ids = trigger_ids.clone()
                no_improve = 0
            else:
                no_improve += 1

            # Stop if no improvement for patience iterations
            if no_improve >= patience:
                break

            # Compute contrastive loss for gradients
            loss = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg).sum(dim=0) + 1e-8)).mean()

            self.model.zero_grad()
            loss.backward()
            grads = self.embedding_grads['last']

            with torch.no_grad():
                # Choose a token position to modify
                pos = random.randrange(trigger_len)
                grad_vec = grads[:, pos, :].mean(dim=0)

                # Generate candidate tokens via HotFlip
                candidates = self.generate_hotflip_candidates(grad_vec, K)
                best_cand = trigger_ids[pos].item()
                best_score = float('inf')

                for cand in candidates:
                    # Replace token at position with candidate
                    trial_ids = trigger_ids.clone()
                    trial_ids[pos] = cand

                    # Create new trigger and embed triggered queries
                    tmp_trigger_text = self.tokenizer.decode(trial_ids, skip_special_tokens=True)
                    trig_queries = [self.insert_trigger(q, tmp_trigger_text, location=location) for q in batch]
                    trig_embs_tmp = torch.stack([self.f_Q(q) for q in trig_queries]).to(self.device)

                    # Evaluate metric for candidate
                    sim_pos_tmp = self.compute_similarity(trig_embs_tmp, poison_emb).squeeze(1)
                    sim_neg_tmp = sim_neg
                    avg_pos_tmp = sim_pos_tmp.mean().item()
                    avg_neg_tmp = sim_neg_tmp.mean().item()
                    cand_metric = -avg_pos_tmp + lambda_reg * avg_neg_tmp

                    if cand_metric < best_score:
                        best_score = cand_metric
                        best_cand = cand

                # Apply the best candidate token
                trigger_ids[pos] = best_cand

        return best_ids, t + 1
