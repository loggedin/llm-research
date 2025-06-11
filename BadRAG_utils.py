import random
import torch
from transformers import AutoTokenizer, AutoModel

# Device and model initialization
# Adjust CUDA index if needed
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever").to(device)
model.eval()

# Hook storage for gradients
grad_storage = {}
def embedding_backward_hook(module, grad_input, grad_output):
    """Capture the embedding gradients."""
    grad_storage["emb_grad"] = grad_output[0].detach().clone()

# Register the hook
model.embeddings.word_embeddings.register_full_backward_hook(embedding_backward_hook)


def embed_text(text: str) -> torch.Tensor:
    """
    Return the mean‐pooled Contriever embedding for `text`.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return (summed / counts).squeeze(0)


def insert_trigger(query: str, trigger: str, location: str = "random") -> str:
    """
    Insert the single token `trigger` into `query` at `location`.
    """
    if location == "start":
        return f"{trigger} {query}"
    elif location == "end":
        return f"{query} {trigger}"
    words = query.split()
    index = random.randint(0, len(words))
    words.insert(index, trigger)
    return " ".join(words)


def dot_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute raw dot‐product similarity between `a` and `b`.
    """
    return torch.matmul(a, b.T)


def get_passage_embedding_eval(input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               token_type_ids: torch.Tensor) -> torch.Tensor:
    """
    Inference‐mode passage embedding under torch.no_grad().
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


def get_passage_embedding_train(input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                token_type_ids: torch.Tensor) -> torch.Tensor:
    """
    Training‐mode passage embedding (gradients enabled).
    """
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
    last_hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


def hotflip_attack(grad_vector: torch.Tensor,
                   embedding_matrix: torch.Tensor,
                   num_candidates: int = 10) -> list:
    """
    Return token IDs that most decrease the loss when substituted.
    """
    scores = torch.matmul(embedding_matrix, -grad_vector)
    return torch.topk(scores, num_candidates).indices.tolist()


def get_top1_doc_id(query_text: str,
                    corpus_embeddings: torch.Tensor,
                    corpus_ids: list) -> str:
    """
    Nearest‐neighbor doc ID for `query_text`.
    """
    q_emb = embed_text(query_text).unsqueeze(0)
    scores = torch.matmul(corpus_embeddings, q_emb.T).squeeze(1)
    return corpus_ids[torch.argmax(scores).item()]


def get_poisoned_rank_and_score(query_text: str,
                                corpus_embeddings: torch.Tensor,
                                corpus_ids: list) -> tuple:
    """
    Return (rank, similarity) of the poisoned passage for `query_text`.
    """
    q_emb = embed_text(query_text).unsqueeze(0)
    scores = torch.matmul(corpus_embeddings, q_emb.T).squeeze(1)
    sorted_idxs = torch.argsort(scores, descending=True)
    for rank, idx in enumerate(sorted_idxs.tolist(), 1):
        if corpus_ids[idx] == "poisoned_doc_001":
            return rank, scores[idx].item()
    return len(corpus_ids), None
