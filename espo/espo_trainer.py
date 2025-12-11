import transformers
from packaging import version
import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
)
from trl.trainer.grpo_trainer import nanmin, nanmax
import wandb
import random
if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.
    Does NOT split fields with list type like 'mask_seeds' (copies them as-is).
    """
    chunks = []
    for i in range(num_chunks):
        chunk = {}
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                chunk_size = val.shape[0] // num_chunks
                chunk[key] = val[i * chunk_size : (i + 1) * chunk_size]
            else:
                chunk[key] = val
        chunks.append(chunk)

    return chunks


def forward_process(batch, prompt_index, mask_id, seed=None):
    set_seed(seed.item())

    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(0, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * ((target_len + 1) / b), steps=b, device=batch.device)).long()
    x = x % (target_len + 1)
    assert x.min() >= 0 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / (target_len + 1)).unsqueeze(1).repeat(1, l)

def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs - log_probs_base 
        log_ratio += (log_probs - log_probs.detach()) *(log_probs.detach() - log_probs_base ) 
    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html

    if kl_estimator == "k2":
        log_ratio = log_probs - log_probs_base
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs - log_probs_base
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp()  - log_ratio - 1

    return log_ratio

class ESPOTrainer(GRPOTrainer):
    """
    ELBO-based Sequence-level Policy Optimization (ESPO) Trainer for diffusion language models.

    Extends GRPO to handle non-autoregressive diffusion LMs by:
    - Treating whole-sequence generation as a single action.
    - Using the ELBO as a tractable proxy for sequence likelihood.
    - Normalizing importance ratios for stable training.
    - Applying robust KL regularization to prevent collapse.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        if "LLaDA" in self.model.config.name_or_path: 
            self.processing_class.mask_token_id = 126336
        self.num_mc = self.args.num_mc



    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        # import pdb; pdb.set_trace();

        prompt_ids= inputs["prompt_ids"]
        completion_ids= inputs["completion_ids"]
        mask_seeds = inputs["mask_seeds"][0] #[num_iterations,num_mc]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        batch_size, logits_to_keep = completion_ids.shape # Only compute logits for completion tokens


        # Get the current iteration index and corresponding mask seed
        this_itr_idx = ( (self._step - 1) % (self.args.num_iterations * self.args.gradient_accumulation_steps)) // self.args.gradient_accumulation_steps
        this_itr_mask_seed = mask_seeds[this_itr_idx:this_itr_idx+1] # [1, num_mc]
        input_ids = input_ids.unsqueeze(0)
        per_token_logps = self._get_elbo_mc(model, input_ids, logits_to_keep,this_itr_mask_seed,num_mc = self.num_mc)



        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][:, this_itr_idx].unsqueeze(1)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp((per_token_logps - old_per_token_logps)/logits_to_keep)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.view(-1, 1)
        per_token_loss2 = coef_2 * advantages.view(-1, 1)
        loss = -torch.min(per_token_loss1, per_token_loss2).sum()/ batch_size
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][:, this_itr_idx].unsqueeze(1)
            kl = compute_approx_kl(per_token_logps, ref_per_token_logps, "k2")
            mean_kl = kl.sum() / (batch_size * logits_to_keep)
            loss += self.beta * mean_kl

        # import pdb; pdb.set_trace();

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather_for_metrics(-per_token_logps.sum()/ (batch_size * logits_to_keep)).mean().item()
        )
        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = is_low_clipped.float().mean()
        high_clip =  is_high_clipped.float().mean()
        clip_ratio = is_region_clipped.float().mean()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

    
    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    @torch.no_grad()
    def generate(
        self,
        model,
        attention_mask,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            # extend attention mask
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, gen_length), value=1)
                 
 

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    mask_index = x == mask_id

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_, attention_mask=attention_mask).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x, attention_mask=attention_mask).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]

            return x

    def get_logits(self, model, batch):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch).logits
            
        if "LLaDA" in self.model.config.name_or_path:
            return logits
        else:
            # since bos always unmask, the first logits will not be used
            # logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            return logits[:, :-1]



    def _get_elbo(self, model, input_ids, logits_to_keep, mask_seeds, reduce_var=True):
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        all_p_masks = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, p_mask = forward_process(
                expanded_input, prompt_index, self.processing_class.mask_token_id, seed=mask_seed
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)
            all_p_masks.append(p_mask)


        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        p_mask = torch.cat(all_p_masks, dim=0)  # [num_iterations * batch_size, seq_len]

        # only keep the last logits_to_keep tokens for efficiency
        targets_kept = expanded_input[:, -logits_to_keep:] #  [num_iterations * batch_size, logits_to_keep]
        p_mask_kept = p_mask[:, -logits_to_keep:]  #  [num_iterations * batch_size, logits_to_keep]

        # Optimize: directly create a loss tensor of shape logits_to_keep to save memory
        loss = torch.zeros(num_iterations * batch_size, logits_to_keep, device=device, dtype=p_mask.dtype)
        mask_index_kept = (perturbed_seq == self.processing_class.mask_token_id)[:, -logits_to_keep:] 

        
        # Get model predictions for the combined batch
        logits = self.get_logits(model, perturbed_seq) # [num_iterations * batch_size, seq_len, vocab_size]
        logits_kept = logits[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep, vocab_size]

        loss[mask_index_kept] = F.cross_entropy(
            logits_kept[mask_index_kept], targets_kept[mask_index_kept], reduction="none"
        ) / p_mask_kept[mask_index_kept]

        if reduce_var:
            coupled_perturbed_seq = expanded_input.clone()
            coupled_perturbed_seq[:, -logits_to_keep:] = torch.where(
                mask_index_kept, coupled_perturbed_seq[:, -logits_to_keep:],  self.processing_class.mask_token_id
            )
            coupled_logits = self.get_logits(model,coupled_perturbed_seq)
            coupled_logits_kept = coupled_logits[:, -logits_to_keep:]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
            loss[~mask_index_kept] = F.cross_entropy(
                coupled_logits_kept[~mask_index_kept], targets_kept[~mask_index_kept], reduction="none"
            ) / (logits_to_keep /(logits_to_keep + 1 )  - p_mask_kept[~mask_index_kept])
            loss/=2

        loss = - loss.view(num_iterations, batch_size, logits_to_keep).permute(1, 0, 2)  # [batch_size, num_iterations, logits_to_keep]
        return loss.sum(dim = -1) # [ batch_size, num_iterations ]

    def _get_elbo_mc(
        self,
        model,
        input_ids,
        logits_to_keep,
        mask_seeds,
        reduce_var=True,
        num_mc=1,
    ):
        """
        Monte Carlo wrapper for _get_block_elbo.
        mask_seeds shape: (num_iterations, num_mc)
        Returns: [num_iterations, batch_size]
        """
        num_iterations, batch_size, _ = input_ids.size()

        # 校验 mask_seeds 形状
        # check shape
        assert mask_seeds.shape[-1] == num_mc, \
            f"Expected mask_seeds[-1] dimension = num_mc={num_mc}, got {mask_seeds.shape[-1]}"

        mc_losses = []

        for mc_idx in range(num_mc):
            loss_single = self._get_elbo(
                model,
                input_ids,
                logits_to_keep,
                mask_seeds[:, mc_idx],  # choose the seed for this MC sample
                reduce_var=reduce_var
            )
            mc_losses.append(loss_single)  # [num_iterations, batch_size]

        # average over Monte Carlo results
        return torch.stack(mc_losses, dim=0).mean(dim=0)



    def _get_elbo_by_chunk(self, model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, device):
        elbos = torch.zeros((prompt_completion_ids_expanded.shape[1], self.num_iterations), device=device)
        local_batch_size = prompt_completion_ids_expanded.shape[1] // self.args.gradient_accumulation_steps
        for i in range(self.args.gradient_accumulation_steps):
            for j in range(self.num_iterations):
                elbos[i*local_batch_size:(i+1)*local_batch_size, j] = self._get_elbo_mc(
                    model,
                    prompt_completion_ids_expanded[j:j+1, i*local_batch_size:(i+1)*local_batch_size, :],
                    logits_to_keep,
                    mask_seeds[i][j:j+1], # [1, num_mc]
                    num_mc=self.num_mc
                )[:,0]
        return elbos

    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the accumulated local batch (Per-GPU batch size × Gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch  (Detailed in `GRPOTrainer`'s `_get_train_sampler` method, see comments there)
        #   - Generates completions once for the entire accumulated batch and splits it into smaller batches
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        steps = self.args.diffusion_steps
        temperature = self.args.generation_temperature

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            # torch.cuda.empty_cache()
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                if "LLaDA" in self.model.config.name_or_path: 
                    batch_prompt_completion_ids = self.generate(
                    model=unwrapped_model,
                    attention_mask=batch_prompt_mask,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=32,
                    temperature=temperature,
                    )
                    prompt_completion_ids_all.append(batch_prompt_completion_ids)

                else:
                    batch_prompt_completion_ids = unwrapped_model.diffusion_generate(
                        batch_prompt_ids,
                        attention_mask=batch_prompt_mask,
                        max_new_tokens=gen_length,
                        output_history=False,
                        return_dict_in_generate=True,
                        steps=steps,
                        temperature=temperature,
                        top_p=0.95 if temperature > 0 else 1.0,
                        alg="entropy",
                        alg_temp=0.0,
                        mask_token_id=self.processing_class.mask_token_id
                    )
                # import pdb; pdb.set_trace();
                    prompt_completion_ids_all.append(batch_prompt_completion_ids.sequences)

                # del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                # torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]  # [accum_batch_size, prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        # eos_token = '<|im_end|>'
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        mask_seeds = torch.randint(0, 2**12, (self.args.gradient_accumulation_steps,self.num_iterations,self.num_mc)) # [gradient_accumulation_steps, num_iterations, num_mc]

        prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(self.num_iterations, -1, -1)  
        # [num_iterations,accum_batch_size, prompt_length]
        
        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                old_per_token_logps = self._get_elbo_by_chunk(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, device
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_elbo_by_chunk(
                        self.ref_model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, device
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_elbo_by_chunk(
                            self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds, device
                        )
                all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )

                # Clip rewards to valid range
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        leave_one_out = True
        if not leave_one_out:
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
        else:
            rewards_grouped = rewards.view(-1, self.num_generations)           # (batch, k)
            sum_group = rewards_grouped.sum(dim=1, keepdim=True)               # (batch, 1)
            baseline  = (sum_group - rewards_grouped) / (self.num_generations - 1)

            advantages = (rewards_grouped - baseline).view(-1)                 # (batch*k,)
            std_grouped_rewards = rewards_grouped.std(dim=1, keepdim=True)         # (batch, 1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=1
            ).view(-1)   
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)


        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
        }

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            Trainer.log(self, logs, start_time)
        else:  # transformers<=4.46
            Trainer.log(self, logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions and (self.state.global_step-1) % 100 == 0: # save prompts and completions every 100 steps
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})
