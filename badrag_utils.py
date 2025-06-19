import json
import random
import torch
from transformers import AutoModel, AutoTokenizer


class BadRAG:
    """
    Implements the COP-style poisoning attack for dense retriever models.

    Uses a gradient-based hotflip procedure to generate adversarial passages
    that maximise similarity to triggered queries while penalising clean queries.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        corpus_emb_path: str = "corpus_embeddings_10000.pt",
        corpus_jsonl_path: str = "./nq/corpus.jsonl",
        device: torch.device = None,
    ):
        """
        Initialise model, load corpus embeddings, and register gradient hook.

        Args:
            retriever_name: HuggingFace model ID for the dense retriever.
            corpus_emb_path: Path to precomputed tensor of document embeddings.
            corpus_jsonl_path: Path to the JSONL file containing document texts.
            device: Torch device (default: cuda:0 if available, else cpu).
        """
        # Select device and load retriever model
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.model = AutoModel.from_pretrained(retriever_name)
        self.model = self.model.to(self.device).eval()

        # Alias to embedding weight matrix for hotflip operations
        self.W_emb = self.model.embeddings.word_embeddings.weight

        # Capture gradients on embedding lookup
        self.embedding_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(
            self._capture_grad
        )

        # Load document embeddings and identifiers
        self.E_D = torch.load(corpus_emb_path).to(self.device)
        self.num_corpus = self.E_D.size(0)
        self.N = self.num_corpus
        self.corpus_ids = []
        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

        # Aliases for paper notation
        self.f_Q = self.encode_query
        self.f_T = self.encode_passage_train

    def _capture_grad(self, module, grad_in, grad_out):
        """
        Backward hook to store gradient of embedding output.

        Args:
            grad_out: Gradients for module outputs.
        """
        # Store gradient tensor for later hotflip candidate generation
        self.embedding_grads['last'] = grad_out[0].detach().clone()

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token embeddings, respecting attention mask.

        Args:
            hidden: Last hidden state, shape (batch, seq_len, dim).
            mask: Attention mask, shape (batch, seq_len).

        Returns:
            Pooled embeddings, shape (batch, dim).
        """
        masked = hidden * mask.unsqueeze(-1).float()
        return masked.sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encode a text query into a single embedding vector.

        Args:
            text: Input query string.

        Returns:
            Tensor of shape (dim,) representing the pooled embedding.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        mask = inputs['attention_mask']
        return self._pool(hidden, mask).squeeze(0)

    def compute_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product similarity between sets of embeddings.

        Args:
            A: Tensor of shape (..., dim).
            B: Tensor of shape (..., dim).

        Returns:
            Similarity matrix of shape (n, m) where n = A_flat.size(0), m = B_flat.size(0).
        """
        A_flat = A.view(-1, A.size(-1))
        B_flat = B.view(-1, B.size(-1))
        return torch.matmul(A_flat, B_flat.T)

    def _hotflip_candidates(
        self,
        grad_vector: torch.Tensor,
        k: int
    ) -> list:
        """
        Propose top-k token replacements via gradient dot-product.

        Args:
            grad_vector: Gradient for a specific token embedding (dim,).
            k: Number of candidates to select.

        Returns:
            List of token IDs with highest scores.
        """
        scores = self.W_emb @ grad_vector
        return torch.topk(scores, k).indices.tolist()

    def generate_hotflip_candidates(
        self,
        grad_vector: torch.Tensor,
        K: int = 10
    ) -> list:
        """
        Generate hotflip token candidates by inverting gradient sign.

        Args:
            grad_vector: Gradient vector for current token.
            K: Number of candidates to return.

        Returns:
            List of K token IDs.
        """
        return self._hotflip_candidates(-grad_vector, K)

    def encode_passage_eval(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a passage in evaluation mode (no gradients).

        Args:
            input_ids: Token IDs tensor.
            attention_mask: Attention mask tensor.
            token_type_ids: Token type IDs tensor.

        Returns:
            Pooled passage embedding.
        """
        with torch.no_grad():
            hidden = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).last_hidden_state
        return self._pool(hidden, attention_mask)

    def encode_passage_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a passage in training mode to capture gradients.

        Args:
            input_ids: Token IDs tensor.
            attention_mask: Attention mask tensor.
            token_type_ids: Token type IDs tensor.

        Returns:
            Pooled passage embedding (requires grad).
        """
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state
        return self._pool(hidden, attention_mask)

    @staticmethod
    def insert_trigger(
        query: str,
        trigger: str,
        location: str = 'random'
    ) -> str:
        """
        Insert a trigger token into a query at a specified location.

        Args:
            query: Original query string.
            trigger: Token to insert.
            location: 'start', 'end', or 'random'.

        Returns:
            Modified query with trigger token.
        """
        if location == 'start':
            return f"{trigger} {query}"
        if location == 'end':
            return f"{query} {trigger}"
        words = query.split()
        idx = random.randint(0, len(words))
        words.insert(idx, trigger)
        return ' '.join(words)

    def select_seed_passage(
        self,
        corpus_jsonl_path: str,
        clean_q_embs: torch.Tensor,
        trig_q_embs: torch.Tensor,
        num_seeds: int = 1000,
        sequence_length: int = 30
    ):
        """
        Select a seed passage whose triggered vs clean similarity gap is maximal.

        Args:
            corpus_jsonl_path: Path to corpus JSONL file.
            clean_q_embs: Embeddings of clean queries.
            trig_q_embs: Embeddings of triggered queries.
            num_seeds: Number of random passages to sample.
            sequence_length: Fixed length for seed token sequence.

        Returns:
            Tuple of (seed_ids, seed_mask, seed_type_ids, best_id, best_text).
        """
        lines = open(corpus_jsonl_path, 'r').readlines()
        samples = random.sample(lines, num_seeds)
        best_diff = -float('inf')
        for ln in samples:
            entry = json.loads(ln)
            p_emb = self.f_Q(entry['text'])
            p_row = p_emb.unsqueeze(0)
            sim_neg = self.compute_similarity(clean_q_embs, p_row).squeeze(1)
            sim_pos = self.compute_similarity(trig_q_embs, p_row).squeeze(1)
            diff = sim_pos.mean().item() - sim_neg.mean().item()
            if diff > best_diff:
                best_diff, best_id, best_text = diff, entry['_id'], entry['text']

        toks = self.tokenizer.encode(
            best_text,
            truncation=True,
            max_length=sequence_length
        )
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.mask_token_id
        toks += [pad_id] * max(0, sequence_length - len(toks))

        seed_ids = torch.tensor([toks], dtype=torch.long, device=self.device)
        seed_mask = torch.ones_like(seed_ids, device=self.device)
        seed_type_ids = torch.zeros_like(seed_ids, device=self.device)
        return seed_ids, seed_mask, seed_type_ids, best_id, best_text

    def generate_poison(
        self,
        adv_passage_ids: torch.Tensor,
        adv_passage_mask: torch.Tensor,
        adv_passage_type_ids: torch.Tensor,
        clean_q_embs: torch.Tensor,
        trig_q_embs: torch.Tensor,
        sequence_length: int = 30,
        T: int = 100,
        batch_size: int = 32,
        K: int = 10,
        lambda_reg: float = 0.5,
        patience: int = 10
    ) -> torch.Tensor:
        """
        Perform iterative hotflip to generate a poisoned passage.

        Args:
            adv_passage_ids: Initial token IDs for adversarial passage.
            adv_passage_mask: Attention mask for passage.
            adv_passage_type_ids: Token type IDs for passage.
            clean_q_embs: Embeddings of clean queries.
            trig_q_embs: Embeddings of triggered queries.
            sequence_length: Length of passage in tokens.
            T: Number of optimisation iterations.
            batch_size: Number of queries per batch.
            K: Hotflip candidate count per token.
            lambda_reg: Weight for clean-similarity regulariser.
            patience: Early-stopping patience threshold.

        Returns:
            Tensor of token IDs for best adversarial passage found.
        """
        adv_ids = adv_passage_ids.clone()
        adv_mask = adv_passage_mask.clone()
        adv_types = adv_passage_type_ids.clone()
        best_ids = adv_ids.clone()
        best_loss = float('inf')
        no_improve = 0

        for _ in range(T):
            self.model.train()
            indices = random.sample(range(clean_q_embs.size(0)), batch_size)
            batch_neg = clean_q_embs[indices]
            batch_pos = trig_q_embs[indices]

            # Compute current passage embedding
            p_emb = self.f_T(adv_ids, adv_mask, adv_types)
            sim_pos = self.compute_similarity(batch_pos, p_emb).squeeze(1)
            sim_neg = self.compute_similarity(batch_neg, p_emb).squeeze(1)
            loss = -sim_pos.mean() + lambda_reg * sim_neg.mean()

            # Backpropagate to obtain embedding gradients
            self.model.zero_grad()
            loss.backward()
            grads = self.embedding_grads['last'].squeeze(0)

            # Hotflip each token position
            with torch.no_grad():
                for pos in range(sequence_length):
                    grad_vec = grads[pos]
                    candidates = self.generate_hotflip_candidates(grad_vec, K)
                    original = adv_ids[0, pos].item()
                    candidate_best = original
                    loss_best = float('inf')

                    for cand in candidates:
                        adv_ids[0, pos] = cand
                        c_emb = self.encode_passage_train(
                            adv_ids, adv_mask, adv_types
                        )
                        sp = self.compute_similarity(batch_pos, c_emb).squeeze(1)
                        sn = self.compute_similarity(batch_neg, c_emb).squeeze(1)
                        l_val = -sp.mean() + lambda_reg * sn.mean()
                        if l_val < loss_best:
                            loss_best, candidate_best = l_val, cand
                    adv_ids[0, pos] = candidate_best

            # Evaluate stopping criterion on validation embedding
            self.model.eval()
            with torch.no_grad():
                eval_emb = self.encode_passage_eval(adv_ids, adv_mask, adv_types)
            avg_pos = self.compute_similarity(trig_q_embs, eval_emb).squeeze(1).mean().item()
            avg_neg = self.compute_similarity(clean_q_embs, eval_emb).squeeze(1).mean().item()
            metric = -avg_pos + lambda_reg * avg_neg

            # Track best and apply early stopping
            if metric < best_loss:
                best_loss = metric
                best_ids = adv_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_ids
