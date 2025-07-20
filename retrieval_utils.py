import random
import torch
from transformers import AutoTokenizer, AutoModel


class BaseRetriever:
    """
    Base class for retrieval-based models using transformer embeddings.
    Provides shared utilities for encoding queries/passages, computing similarity,
    inserting trigger tokens, and generating HotFlip candidates.
    """

    def __init__(self, retriever_name="facebook/contriever", device=None, seed=123):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.model = AutoModel.from_pretrained(retriever_name).to(self.device).eval()

        # Access word embedding matrix
        self.W_emb = self.model.embeddings.word_embeddings.weight

    def encode_query(self, query_text):
        """
        Encode a query string into a pooled embedding vector.

        Args:
            query_text (str): The input query.

        Returns:
            torch.Tensor: Pooled query embedding (1D tensor).
        """
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        return self._pool(hidden, inputs["attention_mask"]).squeeze(0)

    def encode_passage(self, input_ids, attention_mask, token_type_ids=None):
        """
        Encode a passage using its token IDs and attention mask.

        Args:
            input_ids (torch.Tensor): Token IDs of the passage.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs (e.g. for BERT).

        Returns:
            torch.Tensor: Pooled passage embedding (1D tensor).
        """
        if token_type_ids is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(outputs.last_hidden_state, attention_mask)

    def _pool(self, hidden_states, attention_mask):
        """
        Apply mean pooling over hidden states with attention mask.

        Args:
            hidden_states (torch.Tensor): Token-level embeddings.
            attention_mask (torch.Tensor): Binary mask indicating non-padding tokens.

        Returns:
            torch.Tensor: Pooled embedding.
        """
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * mask
        return masked_hidden.sum(dim=1) / mask.sum(dim=1)

    def compute_similarity(self, emb_a, emb_b):
        """
        Compute cosine similarity between two batches of embeddings.

        Args:
            emb_a (torch.Tensor): Tensor of shape (batch_size_a, dim).
            emb_b (torch.Tensor): Tensor of shape (batch_size_b, dim).

        Returns:
            torch.Tensor: Similarity matrix of shape (batch_size_a, batch_size_b).
        """
        a_flat = emb_a.view(-1, emb_a.size(-1))
        b_flat = emb_b.view(-1, emb_b.size(-1))
        return torch.matmul(a_flat, b_flat.T)

    def insert_trigger(self, query_text, trigger_text, location='end'):
        """
        Insert a trigger into the query at a specified position.

        Args:
            query_text (str): The original query.
            trigger_text (str): The trigger string to insert.
            location (str): One of ['start', 'end', 'random'].

        Returns:
            str: The query with the trigger inserted.
        """
        if location not in {"start", "end", "random"}:
            raise ValueError(f"Invalid trigger location: {location}")

        if location == 'start':
            return f"{trigger_text} {query_text}"
        elif location == 'end':
            return f"{query_text} {trigger_text}"
        else:
            words = query_text.split()
            idx = random.randint(0, len(words))
            words.insert(idx, trigger_text)
            return ' '.join(words)

    def generate_hotflip_candidates(self, grad_vector, top_k=10):
        """
        Generate HotFlip token substitution candidates by selecting top tokens
        aligned with the negative gradient direction.

        Args:
            grad_vector (torch.Tensor): Gradient vector.
            top_k (int): Number of candidates to return.

        Returns:
            list[int]: Token IDs of top-K substitutions.
        """
        return self._hotflip_candidates(-grad_vector, top_k)

    def _hotflip_candidates(self, grad_vector, top_k):
        """
        Internal method to get HotFlip candidates sorted by alignment score.

        Args:
            grad_vector (torch.Tensor): Gradient vector.
            top_k (int): Number of candidates to return.

        Returns:
            list[int]: Filtered list of token IDs.
        """
        scores = self.W_emb @ grad_vector  # Token embedding dot gradient
        top_ids = torch.topk(scores, top_k * 5).indices.tolist()
        filtered = [tid for tid in top_ids if self._is_valid_token(tid)]
        return filtered[:top_k]

    def _is_valid_token(self, token_id):
        """
        Check if token is alphabetic and non-special.

        Args:
            token_id (int): Token ID.

        Returns:
            bool: True if valid token.
        """
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.isalpha() and token.lower().isalpha()
