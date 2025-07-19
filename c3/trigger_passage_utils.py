import random
import torch
from transformers import AutoModel, AutoTokenizer


class JointOptimiser:
    """
    Jointly optimises a discrete trigger and poisoned passage to maximise retrieval
    of the poisoned passage using retrieval-augmented generation (RAG).
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        device: torch.device = None,
        seed: int = 123,
    ):
        """
        Initialise the optimiser with a retriever model and tokenizer.

        Args:
            retriever_name (str): HuggingFace model ID for the retriever.
            device (torch.device): Torch device to use (CPU or GPU).
            seed (int): Random seed for reproducibility.
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load the retriever model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.model = AutoModel.from_pretrained(retriever_name).to(self.device).eval()

        # Extract word embeddings
        self.W_emb = self.model.embeddings.word_embeddings.weight

        # Initialise gradient storage
        self.query_grads = {}
        self.doc_grads = {}

        # Register backward hooks to capture gradients
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad_query)
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_grad_doc)

    def _capture_grad_query(self, module, grad_in, grad_out):
        """Capture the gradient for the query."""
        self.query_grads['last'] = grad_out[0]

    def _capture_grad_doc(self, module, grad_in, grad_out):
        """Capture the gradient for the document."""
        self.doc_grads['last'] = grad_out[0]

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masked mean pooling over the hidden states.

        Args:
            hidden (torch.Tensor): Hidden states.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Pooled embeddings.
        """
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        masked_hidden = hidden * mask
        return masked_hidden.sum(dim=1) / mask.sum(dim=1)

    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encode a query string into an embedding.

        Args:
            text (str): Input query text.

        Returns:
            torch.Tensor: Query embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        return self._pool(hidden, inputs['attention_mask']).squeeze(0)

    def encode_passage(self, input_ids, attention_mask, token_type_ids):
        """
        Encode a tokenised passage into an embedding.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.

        Returns:
            torch.Tensor: Passage embedding.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden = outputs.last_hidden_state
        return self._pool(hidden, attention_mask)

    def compute_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            A (torch.Tensor): First embedding.
            B (torch.Tensor): Second embedding.

        Returns:
            torch.Tensor: Similarity score.
        """
        A_flat = A.view(-1, A.size(-1))
        B_flat = B.view(-1, B.size(-1))
        return torch.matmul(A_flat, B_flat.T)

    def _hotflip_candidates(self, grad_vector: torch.Tensor, k: int) -> list:
        """
        Select top-k token IDs based on alignment with gradient vector.

        Filters out non-alphanumeric and special tokens.

        Args:
            grad_vector (torch.Tensor): Gradient vector.
            k (int): Number of candidates.

        Returns:
            list: Top-k token IDs.
        """
        scores = self.W_emb @ grad_vector
        topk_ids = torch.topk(scores, k * 5).indices.tolist()

        filtered_ids = [
            tid for tid in topk_ids
            if self.tokenizer.convert_ids_to_tokens([tid])[0].isalnum()
        ]

        return filtered_ids[:k]

    def generate_hotflip_candidates(self, grad_vector: torch.Tensor, K: int = 10) -> list:
        """
        Wrapper for generating hotflip candidates.

        Args:
            grad_vector (torch.Tensor): Gradient vector.
            K (int): Number of candidates.

        Returns:
            list: Candidate token IDs.
        """
        return self._hotflip_candidates(-grad_vector, K)

    def insert_trigger(self, query: str, trigger: str, location: str = 'end') -> str:
        """
        Insert a trigger phrase into a query.

        Args:
            query (str): Original query.
            trigger (str): Trigger text.
            location (str): Insertion location - 'start', 'end', or 'random'.

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

    def generate_joint_trigger_and_passage(
        self,
        clean_queries: list,
        trigger_len: int = 3,
        passage_len: int = 20,
        K: int = 10,
        T: int = 50,
        lambda_reg: float = 0.5,
        patience: int = 20,
    ):
        """
        Jointly optimise a trigger and passage to maximise retrieval.

        Args:
            clean_queries (list): List of clean query strings.
            trigger_len (int): Length of the trigger (in tokens).
            passage_len (int): Length of the passage (in tokens).
            K (int): Number of hotflip candidates.
            T (int): Maximum number of optimisation steps.
            lambda_reg (float): Regularisation parameter.
            patience (int): Patience for early stopping.

        Returns:
            Tuple: ((best_trigger_ids, best_passage_ids), num_iterations)
        """
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id

        # Initialise trigger and passage IDs with mask tokens
        trigger_ids = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)
        passage_ids = torch.full((1, passage_len), mask_id, dtype=torch.long, device=self.device)
        passage_attention = torch.ones_like(passage_ids)
        passage_token_type = torch.zeros_like(passage_ids)

        # Tracking best results and early stopping
        best_metric = float('inf')
        best_trigger = trigger_ids.clone()
        best_passage = passage_ids.clone()
        no_improve = 0

        for t in range(T):
            # Sample batch of queries
            batch = random.sample(clean_queries, min(32, len(clean_queries)))
            clean_embs = torch.stack([self.encode_query(q) for q in batch]).to(self.device)

            # Create triggered queries
            trigger_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(trigger_ids))
            trig_queries = [self.insert_trigger(q, trigger_text, location='end') for q in batch]
            inputs = self.tokenizer(trig_queries, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Encode passage and triggered queries
            passage_emb = self.encode_passage(passage_ids, passage_attention, passage_token_type)
            outputs = self.model(**inputs)
            trig_embs = self._pool(outputs.last_hidden_state, inputs['attention_mask'])

            # Compute similarity scores
            sim_pos = self.compute_similarity(trig_embs, passage_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_embs, passage_emb).squeeze(1)

            avg_pos = sim_pos.mean().item()
            avg_neg = sim_neg.mean().item()
            metric = -avg_pos + lambda_reg * avg_neg

            # Early stopping check
            if metric < best_metric:
                best_metric = metric
                best_trigger = trigger_ids.clone()
                best_passage = passage_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            # Compute loss for gradient descent
            exp_pos = torch.exp(sim_pos)
            exp_neg = torch.exp(sim_neg)
            loss = -torch.log(exp_pos / (exp_pos + exp_neg.sum(dim=0) + 1e-8)).mean()

            self.model.zero_grad()
            loss.backward()

            grad_trigger = self.query_grads['last']
            grad_passage = self.doc_grads['last']

            # Update either trigger or passage
            with torch.no_grad():
                if random.random() < 0.2:
                    # Update trigger token
                    pos = random.randrange(trigger_len)
                    grad_vec = grad_trigger[:, pos, :].mean(dim=0)
                    candidates = self.generate_hotflip_candidates(grad_vec, K)

                    best_cand, best_score = trigger_ids[pos].item(), float('inf')
                    for cand in candidates:
                        trigger_ids[pos] = cand
                        tmp_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(trigger_ids))
                        trig_qs = [self.insert_trigger(q, tmp_text, location='end') for q in batch]
                        trig_embs_tmp = torch.stack([self.encode_query(q) for q in trig_qs]).to(self.device)
                        sim_pos_tmp = self.compute_similarity(trig_embs_tmp, passage_emb).squeeze(1)
                        avg_pos_tmp = sim_pos_tmp.mean().item()
                        cand_metric = -avg_pos_tmp + lambda_reg * avg_neg
                        if cand_metric < best_score:
                            best_score, best_cand = cand_metric, cand
                    trigger_ids[pos] = best_cand
                else:
                    # Update passage token
                    pos = random.randrange(passage_len)
                    grad_vec = grad_passage[:, pos, :].mean(dim=0)
                    candidates = self.generate_hotflip_candidates(grad_vec, K)

                    best_cand, best_score = passage_ids[0, pos].item(), float('inf')
                    for cand in candidates:
                        passage_ids[0, pos] = cand
                        passage_emb_tmp = self.encode_passage(passage_ids, passage_attention, passage_token_type)
                        sim_pos_tmp = self.compute_similarity(trig_embs, passage_emb_tmp).squeeze(1)
                        avg_pos_tmp = sim_pos_tmp.mean().item()
                        cand_metric = -avg_pos_tmp + lambda_reg * avg_neg
                        if cand_metric < best_score:
                            best_score, best_cand = cand_metric, cand
                    passage_ids[0, pos] = best_cand

        return (best_trigger, best_passage[0]), t + 1
