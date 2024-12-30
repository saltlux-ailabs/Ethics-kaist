import torch
import wandb
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from trl.trainer.utils import pad_to_length


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = F.sigmoid(self.fc3(h2))
        out = torch.clip(out, 0.1, 0.9)
        return out


class DWATrainer(Trainer):
    def __init__(self, alpha, eta, d_update_num, pad, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.eta = eta
        self.d_update_num = d_update_num
        self.pad = pad
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        self.discriminator = Discriminator(self.model.embed_out.in_features).to(self.model.device, dtype=torch.float16)
        # print("Pad Token ID: ", self.pad)
    
    def compute_disc_loss(self, hidden_logits, expert_idx, other_idx, is_expert):
        disc = self.discriminator(hidden_logits)
        # expert_disc = torch.index_select(disc, 0, expert_idx)
        # other_disc = torch.index_select(disc, 0, other_idx)
        # disc_loss = self.eta * torch.mean(-torch.log(expert_disc)) + torch.mean(-torch.log(1-other_disc)) - self.eta * torch.mean(-torch.log(1-expert_disc))
        disc_loss = F.binary_cross_entropy(disc.flatten(), is_expert.to(dtype=torch.float16))
        disc_acc = (disc.flatten().round() == is_expert).sum() / disc.shape[0]
        # return disc_loss, expert_disc, other_disc
        return disc_loss, disc_acc
    
    def compute_logps(self, prompt_attention_mask, response_inputs, response_attention_mask, logits):
        mask = response_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        per_token_logps = torch.gather(logits[:, :-1, :], dim=2, 
                                       index=(mask * response_inputs[:, 1:]).unsqueeze(2)).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_expert = inputs['is_expert'].clone()
        expert_idx = torch.nonzero(is_expert).flatten()
        other_idx = torch.nonzero(1-is_expert).flatten()
        
        response_labels = inputs['response_input_ids'].clone()
        response_labels[response_labels == self.pad] = -100

        outputs = model(**{'input_ids': inputs['response_input_ids'],
                        'attention_mask': inputs['response_attention_mask']}, 
                        output_hidden_states=True)      
            
        # prob = self.compute_logps(prompt_attention_mask=inputs['attention_mask'], 
        #                           response_inputs=inputs['response_input_ids'], 
        #                           response_attention_mask=inputs['response_attention_mask'], 
        #                           logits=outputs.logits)

        # expert_prob = torch.index_select(prob, 0, expert_idx)
        # other_prob = torch.index_select(prob, 0, other_idx)
        
        hidden_logits = torch.mul(inputs['response_attention_mask'][:, :1].unsqueeze(2), outputs.hidden_states[-1][:, 1:, :])
        hidden_logits = torch.mean(hidden_logits, dim=1)
        # disc_loss, expert_disc, other_disc = self.compute_disc_loss(hidden_logits, expert_idx, other_idx, is_expert)
        loss, acc = self.compute_disc_loss(hidden_logits, expert_idx, other_idx, is_expert)
        # expert_denom = expert_disc * (1-expert_disc)
        # other_denom = 1-other_disc
        
        # policy_loss = self.alpha * (torch.mean(- expert_prob)) - torch.mean(- expert_prob * self.eta / expert_denom) + torch.mean(-other_prob / other_denom)
        
        # loss = policy_loss + disc_loss

        # wandb.log({'Policy Loss': policy_loss.item(),
        #            'Discriminator Loss': disc_loss.item(),
        #            'Loss': loss.item()})
        # print(f'Policy Loss: {policy_loss.item()},\n Discriminator Loss: {disc_loss.item()},\n Loss: {loss.item()}')
        print(f'Loss: {loss.item()}, Acc: {acc.item()}')
        
        return (loss, outputs) if return_outputs else loss
    

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
    from data.data_utils import PKUSyntheticDWARDP
    
    model_path = './logs_sft/pku_safe_reward_sft_pythia1p3b_11291631/model'
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    rdp = PKUSyntheticDWARDP(
        prompt_template="\n\nHuman:\n{raw_prompt}\n\nAssistant:\n",
        sanity_check=False,
    )
    # train_dataset = rdp.get_dwa_dataset(split="train")
    valid_dataset = rdp.get_dwa_dataset(split="validation")
    valid_dataset = valid_dataset.shuffle()
    args = TrainingArguments(
        output_dir='./logs_dwbc/output',
        per_device_train_batch_size=32,
        remove_unused_columns=False,
        report_to='none',
        learning_rate=1e-3
    )
    def preprocess_dataset(dataset):
        model_inputs = tokenizer(dataset['prompt'],
                                    max_length = 256,
                                    padding = 'max_length',
                                    truncation=True,
                                    return_tensors='pt')
        response_ids = tokenizer(dataset['response'],
                                    max_length = 256,
                                    padding = 'max_length',
                                    truncation=True,
                                    return_tensors='pt')
        model_inputs['response_input_ids'] = response_ids['input_ids']
        model_inputs['response_attention_mask'] = response_ids['attention_mask']
        
        return model_inputs
    
    
    valid_dataset = valid_dataset.map(preprocess_dataset, batched=True).remove_columns(['prompt', 'response'])
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    trainer = DWATrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=valid_dataset,
        alpha=0.1,
        eta=0.1,
        d_update_num=5,
        pad=tokenizer.pad_token_id,
        # data_collator=data_collator
    )
    trainer.train()