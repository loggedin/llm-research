import json
import random
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
    ):
        """
        Initialise TriggerOptimiser with a retriever model and a fixed corpus.

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
        self.f_T = self.encode_passage_train

    def _capture_grad(self, module, grad_in, grad_out):
        """
        Captures gradient of word embeddings during backward pass.
        """
        self.embedding_grads['last'] = grad_out[0]

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling over sequence length, masking out padding tokens.
        """
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        masked_hidden = hidden * mask
        return masked_hidden.sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encodes a text query into a dense embedding.

        Args:
            text (str): Query string.

        Returns:
            torch.Tensor: Query embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        return self._pool(hidden, inputs['attention_mask']).squeeze(0)

    def compute_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes dot-product similarity between vector batches A and B.

        Args:
            A, B (torch.Tensor): Embedding matrices.

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
            grad_vector (torch.Tensor): Gradient for one token position.
            k (int): Number of candidates.

        Returns:
            list: Token IDs.
        """
        scores = self.W_emb @ grad_vector
        return torch.topk(scores, k).indices.tolist()

    def generate_hotflip_candidates(self, grad_vector: torch.Tensor, K: int = 10) -> list:
        """
        Wrapper for _hotflip_candidates using negative gradient direction.
        """
        return self._hotflip_candidates(-grad_vector, K)

    def encode_passage_train(self, input_ids, attention_mask, token_type_ids):
        """
        Encodes a passage with gradient tracking enabled.

        Returns:
            torch.Tensor: Pooled passage embedding.
        """
        input_ids.requires_grad_()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden = outputs.last_hidden_state
        return self._pool(hidden, attention_mask)

    def insert_trigger(self, query: str, trigger: str, location: str = 'end') -> str:
        """
        Inserts a trigger token into a query string.

        Args:
            query (str): Original query.
            trigger (str): Trigger text.
            location (str): Position to insert ('start', 'end', or 'random').

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

    def append_poison_to_corpus(self, poison_text: str, poison_id: str = "poison_doc"):
        """
        Adds a poisoned passage to the retriever index.

        Args:
            poison_text (str): The passage to insert.
            poison_id (str): ID to assign to the passage.

        Returns:
            torch.Tensor: Poison passage embedding.
        """
        poison_emb = self.encode_query(poison_text).unsqueeze(0)
        self.E_D = torch.cat([self.E_D, poison_emb], dim=0)
        self.corpus_ids.append(poison_id)
        return poison_emb

    def generate_trigger(self, poison_emb: torch.Tensor, clean_queries: list, trigger_len: int = 1, K: int = 10, T: int = 50):
        """
        Performs HotFlip-style optimisation to discover a trigger that
        maximises similarity between poisoned passage and triggered queries.

        Args:
            poison_emb (torch.Tensor): Embedding of the poisoned passage.
            clean_queries (list): List of original (clean) queries.
            trigger_len (int): Number of tokens in the trigger.
            K (int): Top-k token candidates per position.
            T (int): Number of optimisation steps.

        Returns:
            Tuple: (token IDs, trigger string)
        """
        trigger_ids = torch.randint(low=0, high=self.tokenizer.vocab_size, size=(trigger_len,), device=self.device, dtype=torch.long)

        for _ in range(T):
            batch = random.sample(clean_queries, min(32, len(clean_queries)))
            trigger_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(trigger_ids))
            trig_queries = [self.insert_trigger(q, trigger_text, location='end') for q in batch]

            inputs = self.tokenizer(trig_queries, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            q_embs = self._pool(outputs.last_hidden_state, inputs['attention_mask'])
            sim = self.compute_similarity(q_embs, poison_emb).squeeze(1)
            loss = -sim.mean()

            self.model.zero_grad()
            loss.backward()
            grads = self.embedding_grads['last']

            with torch.no_grad():
                for pos in range(trigger_len):
                    grad_vec = grads[:, pos, :].mean(dim=0)
                    candidates = self.generate_hotflip_candidates(grad_vec, K)
                    best_cand, best_score = trigger_ids[pos].item(), float('inf')

                    for cand in candidates:
                        trigger_ids[pos] = cand
                        tmp_trigger_text = self.tokenizer.convert_tokens_to_string(
                            self.tokenizer.convert_ids_to_tokens(trigger_ids)
                        )
                        trig_queries = [self.insert_trigger(q, tmp_trigger_text, location='end') for q in batch]
                        q_embs = torch.stack([self.f_Q(q) for q in trig_queries]).to(self.device)
                        sim = self.compute_similarity(q_embs, poison_emb).squeeze(1)
                        l_val = -sim.mean().item()
                        if l_val < best_score:
                            best_score, best_cand = l_val, cand

                    trigger_ids[pos] = best_cand

        final_tokens = self.tokenizer.convert_ids_to_tokens(trigger_ids)
        return trigger_ids, self.tokenizer.convert_tokens_to_string(final_tokens)
