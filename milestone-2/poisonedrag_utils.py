import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

class PoisonedRAG:
    """
    Implements Algorithm 1 (black-box) and Algorithm 2 (white-box) from the PoisonedRAG paper,
    using notation Q, R, I, S, P and encoders f_Q, f_T as in the publication.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        llm_name: str = "meta-llama/Llama-2-7b-chat-hf",
        V: int = 100,
        T: int = 50,
        K: int = 5,
        corpus_emb_path: str = "corpus_embeddings_10000.pt",
    ) -> None:
        """
        Initialise retriever and LLM components, load embeddings, and set hyperparameters.

        Args:
            retriever_name:  HuggingFace model ID for the dense retriever (f_T).
            llm_name:        HuggingFace model ID for the generation model (M).
            V:               Maximum number of words in I (the injected snippet).
            T:               Number of optimisation iterations for S.
            K:               Number of HotFlip candidates per step.
            corpus_emb_path: Path to precomputed corpus embeddings E(D).
        """
        # Device and retriever setup (f_T)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer_QT = AutoTokenizer.from_pretrained(retriever_name)
        self.f_T = AutoModel.from_pretrained(retriever_name).to(self.device).eval()
        self.W_emb = self.f_T.embeddings.word_embeddings.weight

        # Load database embeddings E(D)
        self.E_D = torch.load(corpus_emb_path, weights_only=True).to(self.device)
        self.N = self.E_D.size(0)

        # LLM M for TEXTGENERATION
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer_M = AutoTokenizer.from_pretrained(llm_name)
        self.M = AutoModelForCausalLM.from_pretrained(
            llm_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_cfg,
        ).eval()

        # Hyperparameters from the paper: V, T, K
        self.V = V
        self.T = T
        self.K = K

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token embeddings, respecting attention mask.

        Args:
            hidden:  Last hidden states, shape (batch, seq_len, dim).
            mask:    Attention mask, shape (batch, seq_len).

        Returns:
            Pooled embeddings, shape (batch, dim).
        """
        masked = hidden * mask.unsqueeze(-1).float()
        return masked.sum(dim=1) / mask.sum(dim=1)

    def _hotflip(self, grad: torch.Tensor, k: int) -> torch.Tensor:
        """
        Perform a HotFlip step: compute top-k token substitutions.

        Args:
            grad: Gradient vector for a single token, shape (dim,).
            k:    Number of candidate replacements.

        Returns:
            Indices of top-k candidate tokens in the vocabulary.
        """
        scores = self.W_emb @ grad
        return torch.topk(scores, k).indices

    def TEXTGENERATION(self, Q: str, R: str) -> str:
        """
        Algorithm 1: generate I via the LLM M such that, when paired with Q,
        the model will output R (the target answer).

        Args:
            Q: The question string.
            R: The desired (incorrect) answer.

        Returns:
            I: A short, misleading paragraph of up to V words.
        """
        prompt = (
            f'You are a helpful assistant.\n\n'
            f'Question: "{Q}"\n'
            f'Desired answer: "{R}"\n\n'
            f'Write a short (~{self.V} words) paragraph I that, when used\n'
            f'as context for {Q}, makes an LLM output "{R}". No disclaimers.\n'
            f'Begin I now:\n'
        )
        attempts = self.V // 50 + 1
        for _ in range(attempts):
            batch = self.tokenizer_M(prompt, return_tensors="pt").to(self.M.device)
            gen = self.M.generate(
                **batch,
                max_new_tokens=self.V + 50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            text = self.tokenizer_M.decode(gen[0], skip_special_tokens=True)
            I = text.split("Begin I now:")[-1].strip()
            if len(I.split()) <= self.V:
                return I
        return I

    def BLACK_BOX(self, Q: str, R: str) -> str:
        """
        Algorithm 1 black-box pass: form P = Q ⊕ I.

        Args:
            Q: The question.
            R: The target answer.

        Returns:
            Concatenated malicious passage P_bb.
        """
        I = self.TEXTGENERATION(Q, R)
        return f"{Q} {I}"

    def WHITE_BOX(self, Q: str, I: str) -> str:
        """
        Algorithm 2 white-box pass: optimise S to maximise
        Sim(f_T(S⊕I), f_T(Q)) via HotFlip, then return P = S* ⊕ I.

        Args:
            Q: The original question.
            I: The injected snippet from the black-box step.

        Returns:
            Concatenated optimised passage P_wb.
        """
        # Tokenise full P = Q⊕I for embedding and gradients
        tok = self.tokenizer_QT(Q, I, return_tensors="pt", truncation=True).to(self.device)
        ids, mask, types = tok["input_ids"], tok["attention_mask"], tok["token_type_ids"]

        # Compute fixed embedding of Q alone, detached from gradient graph
        tok_q = self.tokenizer_QT(Q, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            hidden_q = self.f_T(
                input_ids=tok_q["input_ids"],
                attention_mask=tok_q["attention_mask"],
            ).last_hidden_state
            H_Q = self._pool(hidden_q, tok_q["attention_mask"])

        # Precompute positions of question tokens in the joint sequence
        q_positions = [i for i, t in enumerate(types[0]) if t == 0 and i != 0]

        adv_ids = ids.clone()
        for _ in range(self.T):
            embeds = self.f_T.embeddings.word_embeddings(adv_ids)
            embeds.retain_grad()

            # Forward pass for current S⊕I
            hidden = self.f_T(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
            H_P = self._pool(hidden, mask)

            # Loss = similarity towards H_Q
            loss = (H_P @ H_Q.T).mean()
            self.f_T.zero_grad()
            loss.backward()

            grads = embeds.grad[0]
            pos = random.choice(q_positions)

            candidates = self._hotflip(grads[pos], self.K)
            best_score, best_tok = loss.item(), adv_ids[0, pos].item()

            # Evaluate each candidate under no_grad to find best replacement
            with torch.no_grad():
                for cand in candidates:
                    temp_ids = adv_ids.clone()
                    temp_ids[0, pos] = cand
                    hidden_tmp = self.f_T(input_ids=temp_ids, attention_mask=mask).last_hidden_state
                    tmp_emb = self._pool(hidden_tmp, mask)
                    score = (tmp_emb @ H_Q.T).item()
                    if score > best_score:
                        best_score, best_tok = score, cand.item()

            adv_ids[0, pos] = best_tok

        # Decode using the original question token positions
        S_ids = adv_ids[0, q_positions]
        S = self.tokenizer_QT.decode(S_ids, skip_special_tokens=True)
        return f"{S} {I}"

    def optimise(self, Q: str, R: str) -> tuple[str, str]:
        """
        Run both black-box and white-box attacks in sequence.

        Args:
            Q: The question.
            R: The target answer.

        Returns:
            Tuple of (P_bb, P_wb).
        """
        I = self.TEXTGENERATION(Q, R)
        P_bb = f"{Q} {I}"
        P_wb = self.WHITE_BOX(Q, I)
        return P_bb, P_wb
