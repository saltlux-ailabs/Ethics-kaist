import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments, AutoTokenizer
from trl import SFTTrainer, set_seed, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import numpy as np
import pandas as pd
from ..data.data_utils import PKUSyntheticRDP
tqdm.pandas()


# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./logs_sft/')
    learning_rate: Optional[float] = field(default=1.4e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    max_length: Optional[int] = field(default=512, metadata={"help": "max length"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    wandb_name: Optional[str] = field(default='pku_better_sft_llama3_3b_instruct', metadata={"help": "Name for this experiment"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_model_name: Optional[str] = field(default="meta-llama/Llama-3.2-3B-Instruct", metadata={"help": "local path to the base model or the huggingface id"})
    # padding_side: Optional[str] = field(default="right", metadata={"help": "tokenizer padding side"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
tokenier_name = base_model_name # we use the same tokenizer for the base model
now = datetime.now()
wandb_name = f"{script_args.wandb_name}_{now.strftime('%m%d%H%M')}"
print('base model: ', base_model_name)
os.makedirs(os.path.join(script_args.save_directory, wandb_name), exist_ok=True)

training_args = TrainingArguments(
        max_steps=30000,  
        num_train_epochs=2,
        output_dir=os.path.join(script_args.save_directory, wandb_name),
        dataloader_drop_last=True,
        eval_steps=5000,
        evaluation_strategy='steps',
        save_steps=5000,
        logging_steps=10,
        logging_strategy='steps',
        save_strategy='steps',
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        weight_decay=0.01,
        run_name=wandb_name,
        report_to=script_args.log_with,
        ddp_find_unused_parameters=False,
    )

process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))


# set seed before initializing value head for deterministic eval
set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(tokenier_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

if script_args.load_in_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        load_in_8bit=True, 
        device_map=gpu_id, 
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id, 
    )
model.config.pad_token_id = tokenizer.pad_token_id

# if exp_type == 'assistant':
#     dataset = build_dataset(hhrlhf_dataset_path, tokenizer, split='train') 
#     valid_dataset = build_dataset(hhrlhf_dataset_path, tokenizer, split='test') 
#     collator = DataCollatorForCompletionOnlyLM(response_template=Instructions.response_split, tokenizer=tokenizer, mlm=False)
# else:
#     dataset = build_dataset_summary(summary_dataset_path, tokenizer, split='train')
#     collator = DataCollatorForCompletionOnlyLM(response_template=Instructions_summary.response_split, tokenizer=tokenizer, mlm=False)

rdp = PKUSyntheticRDP(
    prompt_template="\n\nHuman:\n{raw_prompt}\n\nAssistant:\n",
    sanity_check=False,
)
train_dataset = rdp.get_sft_dataset(split="train", chosen_only=True)
valid_dataset  = rdp.get_sft_dataset(split="validation", chosen_only=True)
# train_dataset = PKUSyntheticRDP().get_sft_dataset(split="train")
# valid_dataset = PKUSyntheticRDP().get_sft_dataset(split="validation")
train_dataset = train_dataset.shuffle()

print(f"Size of the train set: {len(train_dataset)}")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    packing=False,
    dataset_text_field="prompt",
    max_seq_length=script_args.max_length
)

print("Training........")
if Accelerator().is_local_main_process and script_args.load_in_8bit:
    trainer.model.print_trainable_parameters()
trainer.train()

if process_id == 0:
    print("Saving last checkpoint of the model")
    save_path = os.path.join(script_args.save_directory, wandb_name, 'model')
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    

