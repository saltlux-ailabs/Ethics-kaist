from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from accelerate.utils import is_deepspeed_available
from contextlib import nullcontext
import warnings

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, is_wandb_available
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.utils import pad_to_length


if is_peft_available():
    from peft import PeftModel

if is_wandb_available():
    import wandb

# if is_deepspeed_available():
#     import deepspeed


class DDRMTrainer(DPOTrainer):
    
    # @staticmethod
    def get_batch_per_token_logp(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)
        return per_token_logps, loss_mask
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.get("concatenated_decoder_input_ids")

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # for llava, the model returns logits for the entire sequence, including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:] 

        all_per_token_logp, all_per_token_mask = self.get_batch_per_token_logp(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )


        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_per_token_chosen_logp = all_per_token_logp[:len_chosen]
        all_per_token_rejected_logp = all_per_token_logp[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        
        chosen_mask = all_per_token_mask[:len_chosen]
        rejected_mask = all_per_token_mask[len_chosen:]
        
        chosen_labels = labels[:len_chosen]
        rejected_labels = labels[len_chosen:]

        if self.aux_loss_enabled:
            return (all_per_token_chosen_logp, 
                    all_per_token_rejected_logp, 
                    chosen_logits, 
                    rejected_logits, 
                    chosen_mask,
                    rejected_mask,
                    nll_loss, outputs.aux_loss)

        return (all_per_token_chosen_logp, 
                all_per_token_rejected_logp, 
                chosen_logits, 
                rejected_logits, 
                chosen_labels, 
                rejected_labels,
                chosen_mask,
                rejected_mask, 
                nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
            chosen_mask,
            rejected_mask,
            policy_nll_loss,
        ) = forward_output[:9]
        
        if self.aux_loss_enabled:
            aux_loss = forward_output[9]

        losses, chosen_rewards, rejected_rewards = self.ddmr_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_mask,
            rejected_mask,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper:
            losses = losses + policy_nll_loss * self.args.rpo_alpha

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    def ddmr_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        chosen_mask: torch.FloatTensor,
        rejected_mask: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps)
        # rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # log sigmoid 버전 추후에 log probability (softmax) 쓰는 버전도 만들어야 함.

        # try 1
        # chosen_labels = chosen_labels[:, 1:].clone()
        # rejected_labels = rejected_labels[:, 1:].clone()
        # chosen_logits = torch.gather(policy_chosen_logits[:,:-1,:], dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)
        # rejected_logits = -torch.gather(policy_rejected_logits[:,:-1,:], dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)
        # chosen_loss_mask = chosen_labels != self.label_pad_token_id
        # rejected_loss_mask = rejected_labels != self.label_pad_token_id
        # chosen_rewards = F.logsigmoid(chosen_logits * chosen_loss_mask).sum(-1)
        # rejected_rewards = F.logsigmoid(rejected_logits * rejected_loss_mask).sum(-1)

        # try 2
        # chosen_rewards = policy_chosen_logps
        # rejected_rewards = policy_rejected_logps

        # losses = - (F.logsigmoid(chosen_rewards) + F.logsigmoid(- rejected_rewards)) / 2

        # try 3 - + Length Normalization
        # len_chosen = (chosen_labels != self.label_pad_token_id).sum(-1)
        # len_rejected = (rejected_labels != self.label_pad_token_id).sum(-1)
        chosen_rewards = policy_chosen_logps 
        rejected_rewards = policy_rejected_logps
        
        chosen_loss = - F.logsigmoid(chosen_rewards) * chosen_mask
        rejected_loss = -F.logsigmoid(-rejected_rewards) * rejected_mask
        losses = chosen_loss + rejected_loss
        # print("chosen:", chosen_rewards)
        # print("rejected", rejected_rewards)
        
        # logits = chosen_rewards - rejected_rewards
        # losses = (chosen_rewards + rejected_rewards) / 2
        # if self.loss_type == "sigmoid":
        #     losses = -F.logsigmoid(logits)
        # elif self.loss_type == "hinge":
        #     losses = torch.relu(1 - logits)
        # else:
        #     raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    
    
      
    
if __name__ == '__main__':
    from data_config import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
    from trl import DPOConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rdp = DATASET_CONFIGS['Anthropic/hh-rlhf'](prompt_template=DEFAULT_PROMPT_TEMPLATE)
    dataset = rdp.get_preference_dataset(split='validation')
    model = AutoModelForCausalLM.from_pretrained(
        'google/gemma-2b',
        torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    training_args = DPOConfig('google/gemma-2b', per_device_train_batch_size = 2, per_device_eval_batch_size=1, report_to='none')
    trainer = DDRMTrainer(model = model, args = training_args, tokenizer = tokenizer, train_dataset=dataset, label_pad_token_id = tokenizer.pad_token_id)
    trainer.train()