import json
import random
import torch
from transformers import AutoModel, AutoTokenizer


class BadRAG:
    """
    Implements a gradient-based adversarial attack on retrieval-augmented generation (RAG)
    using contrastive passage optimisation, as described in the BadRAG framework.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        corpus_emb_path: str = "corpus_embeddings_10000.pt",
        corpus_jsonl_path: str = "./nq/corpus.jsonl",
        device: torch.device = None,
    ):
        """
        Initialise BadRAG with a retriever model and a fixed corpus.

        Args:
            retriever_name (str): HuggingFace model ID for the retriever.
            corpus_emb_path (str): Path to precomputed passage embeddings.
            corpus_jsonl_path (str): Path to corpus in JSONL format with "_id" keys.
            device (torch.device): Torch device to use (CPU or GPU).
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.model = AutoModel.from_pretrained(retriever_name).to(self.device).eval()

        self.W_emb = self.model.embeddings.word_embeddings.weight
        self.embedding_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad)

        self.E_D = torch.load(corpus_emb_path).to(self.device)
        self.num_corpus = self.E_D.size(0)
        self.corpus_ids = []

        with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= self.num_corpus:
                    break
                entry = json.loads(line)
                self.corpus_ids.append(entry["_id"])

    def _capture_grad(self, module, grad_in, grad_out):
        """
        Stores the gradient of the word embeddings after backward pass.
        """
        self.embedding_grads['last'] = grad_out[0].detach().clone()

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling over the sequence length, masking padded tokens.
        """
        masked = hidden * mask.unsqueeze(-1).float()
        return masked.sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encodes a query string into a dense vector using the retriever.

        Args:
            text (str): Query text.

        Returns:
            torch.Tensor: Pooled embedding vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        return self._pool(hidden, inputs['attention_mask']).squeeze(0)

    def compute_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes dot-product similarity between vectors A and B.

        Args:
            A, B (torch.Tensor): Batches of vectors (N x D).

        Returns:
            torch.Tensor: Similarity scores.
        """
        A_flat = A.view(-1, A.size(-1))
        B_flat = B.view(-1, B.size(-1))
        return torch.matmul(A_flat, B_flat.T)

    def _hotflip_candidates(self, grad_vector: torch.Tensor, k: int) -> list:
        """
        Selects top-k vocabulary tokens based on gradient direction.

        Args:
            grad_vector (torch.Tensor): Gradient for a single token.
            k (int): Number of candidates to return.

        Returns:
            list: Token IDs of top-k candidates.
        """
        scores = self.W_emb @ grad_vector
        return torch.topk(scores, k).indices.tolist()

    def generate_hotflip_candidates(self, grad_vector: torch.Tensor, K: int = 10) -> list:
        """
        Wrapper for _hotflip_candidates with negative gradient direction.

        Args:
            grad_vector (torch.Tensor): Input gradient vector.
            K (int): Number of token candidates to consider.

        Returns:
            list: Token IDs.
        """
        return self._hotflip_candidates(-grad_vector, K)

    def encode_passage(self, input_ids, attention_mask):
        """
        Encodes a passage input to its dense representation.
        """
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self._pool(hidden, attention_mask)

    def insert_trigger(self, query: str, trigger: str, location: str = 'random') -> str:
        """
        Inserts a trigger token into the query.

        Args:
            query (str): Original query.
            trigger (str): Trigger word or phrase.
            location (str): 'start', 'end', or 'random'.

        Returns:
            str: Modified query.
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

    def generate_poison(
        self,
        clean_q_embs,
        trig_q_embs,
        sequence_length=30,
        T=100,
        batch_size=32,
        K=10,
        lambda_reg=0.5,
        patience=10
    ):
        """
        Performs adversarial token substitution to generate a poisoned passage.

        Args:
            clean_q_embs (Tensor): Embeddings for clean queries.
            trig_q_embs (Tensor): Embeddings for triggered queries.
            sequence_length (int): Max token length for poisoned passage.
            T (int): Max optimisation steps.
            batch_size (int): Batch size for contrastive updates.
            K (int): Top-k HotFlip candidates per token.
            lambda_reg (float): Weight for clean query penalty.
            patience (int): Early stopping patience.

        Returns:
            Tensor: Final adversarial token IDs (1 x sequence_length).
        """
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        adv_ids = torch.full((1, sequence_length), mask_id, dtype=torch.long, device=self.device)
        adv_mask = torch.ones_like(adv_ids)

        best_ids = adv_ids.clone()
        best_loss = float('inf')
        no_improve = 0

        for _ in range(T):
            self.model.train()
            indices = random.sample(range(clean_q_embs.size(0)), batch_size)
            batch_neg = clean_q_embs[indices]
            batch_pos = trig_q_embs[indices]

            p_emb = self.encode_passage(adv_ids, adv_mask)

            sim_pos = self.compute_similarity(batch_pos, p_emb).squeeze(1)
            sim_neg = self.compute_similarity(batch_neg, p_emb).squeeze(1)

            exp_pos = torch.exp(sim_pos)
            exp_negs = torch.exp(sim_neg)
            loss = -torch.log(exp_pos / (exp_pos + exp_negs.sum(dim=0) + 1e-8)).mean()

            self.model.zero_grad()
            loss.backward()
            grads = self.embedding_grads['last'].squeeze(0)

            with torch.no_grad():
                for pos in range(sequence_length):
                    grad_vec = grads[pos]
                    candidates = self.generate_hotflip_candidates(grad_vec, K)
                    original = adv_ids[0, pos].item()
                    candidate_best = original
                    loss_best = float('inf')

                    for cand in candidates:
                        adv_ids[0, pos] = cand
                        c_emb = self.encode_passage(adv_ids, adv_mask)
                        sp = self.compute_similarity(batch_pos, c_emb).squeeze(1)
                        sn = self.compute_similarity(batch_neg, c_emb).squeeze(1)
                        exp_p = torch.exp(sp)
                        exp_n = torch.exp(sn)
                        l_val = -torch.log(exp_p / (exp_p + exp_n.sum(dim=0) + 1e-8)).mean()
                        if l_val < loss_best:
                            loss_best, candidate_best = l_val.item(), cand

                    adv_ids[0, pos] = candidate_best

            self.model.eval()
            with torch.no_grad():
                eval_emb = self.encode_passage(adv_ids, adv_mask)
            avg_pos = self.compute_similarity(trig_q_embs, eval_emb).squeeze(1).mean().item()
            avg_neg = self.compute_similarity(clean_q_embs, eval_emb).squeeze(1).mean().item()
            metric = -avg_pos + lambda_reg * avg_neg

            if metric < best_loss:
                best_loss = metric
                best_ids = adv_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_ids
