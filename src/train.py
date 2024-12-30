import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from trl import DPOConfig, DPOTrainer
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd())))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from trainer.ddrm_trainer import DDRMTrainer
from ..data.data_config import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from ..data.data_utils import PKUSyntheticRDP


def disable_progress_bar_non_local_main():
    if not Accelerator().is_local_main_process:
        import datasets
        import transformers
        import warnings
        datasets.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')

disable_progress_bar_non_local_main()


def param_sharding_enabled():
    from transformers.modeling_utils import is_deepspeed_zero3_enabled, is_fsdp_enabled
    return is_deepspeed_zero3_enabled() or is_fsdp_enabled()


@dataclass
class ScriptArguments:
    sft_model_name: str = field(default = 'meta-llama/Llama-3.2-3B-Instruct', metadata={"help": "the sft model name"})
    # sft_model_name: str = field(default = 'google/gemma-2b', metadata={"help": "the sft model name"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "whether to use flash attention 2"})
    prompt_template: Optional[str] = field(default=DEFAULT_PROMPT_TEMPLATE, metadata={"help": "the prompt template"})
    dataset_name: Optional[str] = field(default='Anthropic/hh-rlhf', metadata={"help": "the dataset name"})
    dataset_caching: Optional[bool] = field(default=False, metadata={"help": "used cached dataset"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})

    max_length: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    generate_during_eval: Optional[bool] = field(default=True, metadata={"help": "whether to generate during evaluation"})

    # output_dir: Optional[str] = field(default="./logs_ddrm/pku_better_llama3_3b_ddrm", metadata={"help": ""})
    output_dir: Optional[str] = field(default="./logs_ddrm/pku_safe_llama_3b_ddrm", metadata={"help": ""})
    overwrite_output_dir: Optional[bool] = field(default=True, metadata={"help": ""})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": ""})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": ""})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": ""})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": ""})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": ""})
    # warmup_steps=0.1,
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": ""})
    fp16: Optional[bool] = field(default=True, metadata={"help": ""})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": ""})
    report_to: Optional[str] = field(default="wandb", metadata={"help": ""})

    num_train_epochs: Optional[int] = field(default=3, metadata={"help": ""})
    logging_steps: Optional[int] = field(default=10, metadata={"help": ""})
    save_steps: Optional[float] = field(default=0.25, metadata={"help": ""})
    eval_steps: Optional[float] = field(default=0.25, metadata={"help": ""})
    eval_delay: Optional[float] = field(default=0.25, metadata={"help": ""})
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": ""})
    save_total_limit: Optional[int] = field(default=3, metadata={"help": ""})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={"help": ""})
    # model_init_kwargs=None,
    # ref_model_init_kwargs=None,


    peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft for training"})
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
if not script_args.peft:
    peft_config = None
else:
    peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
now = datetime.now()
output_dir = run_name = f"{script_args.output_dir}_{now.strftime('%m%d%H%M')}"

training_args = DPOConfig(
    output_dir=output_dir,
    overwrite_output_dir=script_args.overwrite_output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    # warmup_steps=0.1,
    weight_decay=0.05,
    fp16=False,
    remove_unused_columns=False,
    run_name=run_name,
    report_to="wandb",
    num_train_epochs=3,
    logging_steps=10,
    save_steps=0.15,
    eval_steps=0.15,
    eval_delay=0.15,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    model_init_kwargs=None,
    ref_model_init_kwargs=None,
)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}'.format(process_id))

# base model
print("loading model...")
peft_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name,
    use_flash_attention_2=script_args.use_flash_attention_2, # flash attn
    torch_dtype=torch.float32, # necessary for llama2, otherwise will be cast to float32
    **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
)
sft_model.config.update({
    "use_cache": False,
    # "pad_token_id": sft_model.config.eos_token_id 
})
print(sft_model)
print(peft_config)
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
sft_model.config.pad_token_id = tokenizer.pad_token_id

# dataset
if not script_args.dataset_caching:
    from datasets import disable_caching
    disable_caching()
# rdp = DATASET_CONFIGS[script_args.dataset_name](
#     prompt_template=script_args.prompt_template,
#     sanity_check=script_args.sanity_check,
# )
# train_dataset = rdp.get_preference_dataset(split="train")
# eval_dataset  = rdp.get_preference_dataset(split="validation")
rdp = PKUSyntheticRDP(
    prompt_template="\n\nHuman:\n{raw_prompt}\n\nAssistant:\n",
    sanity_check=False,
)
train_dataset = rdp.get_preference_dataset(split="train")
eval_dataset  = rdp.get_preference_dataset(split="validation")

# get ready for training
print("start training...")
trainer = DDRMTrainer(
    model=sft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    label_pad_token_id = tokenizer.pad_token_id,
    peft_config=peft_config,
    max_length=script_args.max_length,
    generate_during_eval=script_args.generate_during_eval,
)

if Accelerator().is_local_main_process and peft_config:
    trainer.model.print_trainable_parameters()
trainer.train()

save_name = "best_checkpoint" if script_args.load_best_model_at_end else "final_checkpoint"
trainer.model.save_pretrained(os.path.join(output_dir, save_name))
trainer.tokenizer.save_pretrained(os.path.join(output_dir, save_name))