import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval_utils import BaseRetriever


class JointMisinformationOptimiser(BaseRetriever):
    """
    Implements a variant of the joint optimiser that targets misinformation injection.

    This method jointly crafts:
    - A discrete trigger to be inserted into natural queries.
    - A poisoned passage consisting of a learned prefix and a fixed LLM-generated suffix.

    The suffix is query-agnostic and constructed solely from a target response.
    The optimiser aims to increase retrieval of the poison by triggered queries,
    while reducing its retrieval by clean (untampered) queries.
    """

    def __init__(
        self,
        retriever_name: str = "facebook/contriever",
        llm_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: torch.device = None,
        seed: int = 123,
    ) -> None:
        """
        Initialise the optimiser with retrieval and LLM components.

        Args:
            retriever_name (str): Name of the retriever model to be attacked.
            llm_name (str): Name of the language model used to generate suffix content.
            device (torch.device): Torch device for computation (CPU or GPU).
            seed (int): Random seed for reproducibility.
        """
        super().__init__(retriever_name, device, seed)
        self.seed = seed

        # Load LLM (resides on CPU to save memory)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_name).to("cpu")

        # Gradient capture hooks
        self.query_grads = {}
        self.passage_grads = {}
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_query_grad)
        self.model.embeddings.word_embeddings.register_full_backward_hook(self._capture_passage_grad)

    def _capture_query_grad(self, module, grad_in, grad_out) -> None:
        """
        Backward hook to capture gradient from the query embedding layer.

        Args:
            module: The embedding layer module (unused).
            grad_in: Incoming gradients (unused).
            grad_out: Outgoing gradients from the embedding layer.
        """
        self.query_grads["last"] = grad_out[0].detach().clone()

    def _capture_passage_grad(self, module, grad_in, grad_out) -> None:
        """
        Backward hook to capture gradient from the passage embedding layer.

        Args:
            module: The embedding layer module (unused).
            grad_in: Incoming gradients (unused).
            grad_out: Outgoing gradients from the embedding layer.
        """
        self.passage_grads["last"] = grad_out[0].detach().clone()

    def _generate_llm_suffix(self, desired_response: str, suffix_len: int) -> str:
        """
        Generate a query-agnostic suffix paragraph that subtly supports the desired response.

        The generation avoids explicitly referencing any query or question and aims to resemble
        a plausible standalone knowledge-base entry. Truncation occurs at the word level only.

        Args:
            desired_response (str): The target misinformation to encode.
            suffix_len (int): Maximum number of words in the output paragraph.

        Returns:
            str: Truncated, fluent paragraph intended to encode the misinformation.
        """
        prompt = (
            f"You are a helpful assistant.\n\n"
            f'Your goal is to write a plausible and fluent paragraph of exactly {suffix_len} words\n'
            f'that supports or implies the following statement:\n\n'
            f'"{desired_response}"\n\n'
            f'The paragraph must be exactly {suffix_len} words.\n'
            f'Begin I now:\n'
        )

        min_tokens = suffix_len
        max_tokens = suffix_len + 50
        input_ids = self.llm_tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

        # Set global random seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Use generate directly
        output_ids = self.llm_model.generate(
            input_ids,
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )

        # Decode and truncate to suffix_len directly
        generated_text = self.llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        suffix = generated_text.split("Begin I now:")[-1].strip()
        words = suffix.split()
        return " ".join(words[:suffix_len])

    def generate_joint_trigger_and_passage(
        self,
        clean_queries: list[str],
        desired_response: str,
        trigger_len: int = 1,
        location: str = 'random',
        prefix_len: int = 10,
        suffix_len: int = 50,
        suffix_ids: torch.Tensor = None,
        top_k: int = 10,
        max_steps: int = 1000,
        lambda_reg: float = 0.5,
        patience: int = 20,
        batch_size: int = 32
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        """
        Jointly optimises a discrete trigger and prefix, to maximise retrieval of a poisoned passage.

        The poisoned passage consists of:
        - A prefix: optimised token sequence via HotFlip.
        - A fixed suffix: LLM-generated from the target response (query-agnostic).

        Args:
            clean_queries (list[str]): Natural, unmodified training queries.
            desired_response (str): The misinformation to encode in the suffix.
            trigger_len (int): Length of the learned trigger to insert into queries.
            location (str): Insertion point for the trigger (start, end, or random).
            prefix_len (int): Number of tokens in the prefix to optimise.
            suffix_len (int): Approximate length (in words) for suffix generation.
            suffix_ids (torch.Tensor, optional): Precomputed suffix token IDs. If None, generates via LLM.
            top_k (int): Number of candidate tokens for each HotFlip update.
            max_steps (int): Maximum number of optimisation steps.
            lambda_reg (float): Weight on clean-query penalty in the objective.
            patience (int): Early stopping threshold.
            batch_size (int): Number of training examples per step.

        Returns:
            tuple: ((trigger_ids, passage_ids), num_steps)
                trigger_ids (torch.Tensor): Learned token IDs for the trigger.
                passage_ids (torch.Tensor): Final poisoned passage (prefix + suffix).
                num_steps (int): Number of update steps performed.
        """
        # Split queries into training and validation sets
        num_queries = len(clean_queries)
        val_size = max(1, int(0.2 * num_queries))
        indices = list(range(num_queries))
        random.shuffle(indices)
        train_queries = [clean_queries[i] for i in indices[val_size:]]
        val_queries = [clean_queries[i] for i in indices[:val_size]]

        # Generate or accept fixed suffix
        if suffix_ids is None:
            suffix_text = self._generate_llm_suffix(desired_response, suffix_len=suffix_len)
            suffix_ids = self.tokenizer(suffix_text, return_tensors="pt", truncation=True).input_ids.to(self.device)

        if suffix_ids.dim() == 1:
            suffix_ids = suffix_ids.unsqueeze(0)

        suffix_len = suffix_ids.size(1)

        # Initialise prefix and trigger with [MASK] or [UNK]
        mask_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
        prefix_ids = torch.full((1, prefix_len), mask_id, dtype=torch.long, device=self.device)
        trigger_ids = torch.full((trigger_len,), mask_id, dtype=torch.long, device=self.device)
        attention_mask = torch.ones((1, prefix_len + suffix_len), dtype=torch.long, device=self.device)

        best_trigger = trigger_ids.clone()
        best_prefix = prefix_ids.clone()
        best_metric = float("inf")
        no_improve = 0

        for step in range(max_steps):
            batch = random.sample(train_queries, min(batch_size, len(train_queries)))
            clean_embs = self.encode_query(batch, require_grad=False)

            trigger_text = self.tokenizer.decode(trigger_ids, skip_special_tokens=True)
            triggered_batch = [self.insert_trigger(q, trigger_text, location=location) for q in batch]
            triggered_embs = self.encode_query(triggered_batch, require_grad=True)

            full_passage_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
            passage_emb = self.encode_passage(full_passage_ids, attention_mask, require_grad=True)

            sim_pos = self.compute_similarity(triggered_embs, passage_emb).squeeze(1)
            sim_neg = self.compute_similarity(clean_embs, passage_emb).squeeze(1)
            loss = -sim_pos.mean() + lambda_reg * sim_neg.mean()

            self.model.zero_grad()
            loss.backward()

            grad_trig = self.query_grads["last"]
            grad_pass = self.passage_grads["last"]

            if random.random() < 0.2:
                # Trigger update (20% of steps)
                pos = random.randint(0, trigger_len - 1)
                grad_vec = grad_trig[:, pos, :].mean(dim=0)
                candidates = self.generate_hotflip_candidates(grad_vec, top_k)
                best_token = trigger_ids[pos].item()
                best_score = float("inf")

                for cand in candidates:
                    trial_ids = trigger_ids.clone()
                    trial_ids[pos] = cand
                    trial_text = self.tokenizer.decode(trial_ids, skip_special_tokens=True)
                    trial_batch = [self.insert_trigger(q, trial_text, location=location) for q in batch]
                    trial_embs = self.encode_query(trial_batch, require_grad=False)
                    score = -trial_embs.mean(dim=0).dot(passage_emb.squeeze(0)) + lambda_reg * sim_neg.mean().item()
                    if score < best_score:
                        best_score = score
                        best_token = cand

                trigger_ids[pos] = best_token

            else:
                # Prefix update (80% of steps)
                grad_pass_len = grad_pass.size(1)
                prefix_range = min(prefix_len, grad_pass_len)
                pos = random.randint(0, prefix_range - 1)
                grad_vec = grad_pass[:, pos, :].mean(dim=0)
                candidates = self.generate_hotflip_candidates(grad_vec, top_k)
                best_token = prefix_ids[0, pos].item()
                best_score = float("inf")

                for cand in candidates:
                    trial_prefix = prefix_ids.clone()
                    trial_prefix[0, pos] = cand
                    full_trial = torch.cat([trial_prefix, suffix_ids], dim=1)
                    trial_emb = self.encode_passage(full_trial, attention_mask, require_grad=False)
                    sim_pos_trial = self.compute_similarity(triggered_embs, trial_emb).squeeze(1).mean().item()
                    score = -sim_pos_trial + lambda_reg * sim_neg.mean().item()
                    if score < best_score:
                        best_score = score
                        best_token = cand

                prefix_ids[0, pos] = best_token

            # Validation
            val_clean = self.encode_query(val_queries, require_grad=False)
            val_triggered = self.encode_query(
                [self.insert_trigger(q, trigger_text, location=location) for q in val_queries],
                require_grad=False
            )
            full_val = torch.cat([prefix_ids, suffix_ids], dim=1)
            val_emb = self.encode_passage(full_val, attention_mask, require_grad=False)
            val_sim_pos = self.compute_similarity(val_triggered, val_emb).squeeze(1).mean().item()
            val_sim_neg = self.compute_similarity(val_clean, val_emb).squeeze(1).mean().item()
            val_metric = -val_sim_pos + lambda_reg * val_sim_neg

            if val_metric < best_metric:
                best_metric = val_metric
                best_trigger = trigger_ids.clone()
                best_prefix = prefix_ids.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        final_passage_ids = torch.cat([best_prefix, suffix_ids], dim=1)
        return (best_trigger, final_passage_ids[0]), step + 1
