import random
import torch
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModel
from typing import Union


class BaseRetriever:
    """
    Base class for retrieval-based models using transformer embeddings.

    Provides shared utilities for:
    - Encoding queries and passages into dense vectors.
    - Computing dot product similarity between vectors.
    - Inserting trigger tokens into queries.
    - Generating token substitution candidates using the HotFlip method.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        device: torch.device = None,
        seed: int = 123
    ) -> None:
        """
        Initialise the retriever with model, tokenizer, and embeddings.

        Args:
            retriever_name (str): Name of the transformer model.
            device (torch.device): PyTorch device to use.
            seed (int): Random seed for reproducibility.
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.model = AutoModel.from_pretrained(retriever_name).to(self.device).eval()

        # Extract word embedding matrix from the model
        self.W_emb = self.model.embeddings.word_embeddings.weight

    def encode_query(self, query_text: str, require_grad: bool = False) -> torch.Tensor:
        """
        Encode a single query string into a pooled embedding vector.

        Args:
            query_text (str): The input query.
            require_grad (bool): Whether to track gradients (for backpropagation).

        Returns:
            torch.Tensor: 1D tensor representing the query embedding.
        """
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        context = nullcontext() if require_grad else torch.no_grad()

        # Compute hidden states
        with context:
            hidden = self.model(**inputs).last_hidden_state

        # Apply pooling to get a single embedding vector
        return self._pool(hidden, inputs["attention_mask"]).squeeze(0)

    def encode_passage(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        require_grad: bool = False
    ) -> torch.Tensor:
        """
        Encode a passage tensor into a pooled embedding.

        Args:
            input_ids (torch.Tensor): Token IDs of the passage.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Optional segment IDs (e.g. for BERT).
            require_grad (bool): Whether to track gradients.

        Returns:
            torch.Tensor: 1D tensor representing the passage embedding.
        """
        context = nullcontext() if require_grad else torch.no_grad()

        # Run model with or without token type IDs
        with context:
            if token_type_ids is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

        return self._pool(outputs.last_hidden_state, attention_mask)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling over token embeddings, accounting for padding.

        Args:
            hidden_states (torch.Tensor): Sequence of token-level embeddings.
            attention_mask (torch.Tensor): Mask with 1s for real tokens, 0s for padding.

        Returns:
            torch.Tensor: Mean-pooled embedding for each sequence in the batch.
        """
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * mask

        # Return average of non-masked embeddings
        return masked_hidden.sum(dim=1) / mask.sum(dim=1)

    def compute_similarity(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product similarity between two batches of embeddings.

        Args:
            emb_a (torch.Tensor): Tensor of shape (batch_size_a, dim).
            emb_b (torch.Tensor): Tensor of shape (batch_size_b, dim).

        Returns:
            torch.Tensor: Matrix of shape (batch_size_a, batch_size_b) with similarity scores.
        """
        a_flat = emb_a.view(-1, emb_a.size(-1))
        b_flat = emb_b.view(-1, emb_b.size(-1))
        return torch.matmul(a_flat, b_flat.T)

    def insert_trigger(self, query_text: str, trigger_text: str, location: str = "random") -> str:
        """
        Insert a trigger token into a query string at the specified location.

        Args:
            query_text (str): The original user query.
            trigger_text (str): The trigger phrase or token to insert.
            location (str): Insertion location: 'start', 'end', or 'random'.

        Returns:
            str: Modified query with trigger inserted.

        Raises:
            ValueError: If location is not one of the allowed options.
        """
        if location not in {"start", "end", "random"}:
            raise ValueError(f"Invalid trigger location: {location}")

        if location == "start":
            return f"{trigger_text} {query_text}"
        elif location == "end":
            return f"{query_text} {trigger_text}"
        else:
            words = query_text.split()
            idx = random.randint(0, len(words))
            words.insert(idx, trigger_text)
            return " ".join(words)

    def generate_hotflip_candidates(self, grad_vector: torch.Tensor, top_k: int = 10) -> list[int]:
        """
        Generate token substitution candidates using HotFlip based on gradient direction.

        Args:
            grad_vector (torch.Tensor): Gradient vector from loss w.r.t. embeddings.
            top_k (int): Number of candidates to return.

        Returns:
            list[int]: List of top-K token IDs ranked by alignment with the gradient.
        """
        return self._hotflip_candidates(-grad_vector, top_k)

    def _hotflip_candidates(self, grad_vector: torch.Tensor, top_k: int) -> list[int]:
        """
        Internal method to find token substitutions aligned with a gradient direction.

        Args:
            grad_vector (torch.Tensor): Gradient vector.
            top_k (int): Number of token IDs to return.

        Returns:
            list[int]: Filtered list of top-K token IDs.
        """
        # Compute dot product between each embedding and gradient
        scores = self.W_emb @ grad_vector

        # Take more than needed and filter to valid tokens
        top_ids = torch.topk(scores, top_k * 5).indices.tolist()
        filtered = [tid for tid in top_ids if self._is_valid_token(tid)]

        return filtered[:top_k]

    def _is_valid_token(self, token_id: int) -> bool:
        """
        Determine if a token ID corresponds to a valid (alphabetic, non-special) token.

        Args:
            token_id (int): Token ID from the vocabulary.

        Returns:
            bool: True if token is valid, False otherwise.
        """
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return (
            token.isalpha() and
            token.lower().isalpha() and
            token.isascii() and
            token not in set(self.tokenizer.all_special_tokens)
        )
