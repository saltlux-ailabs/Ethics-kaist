from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate import Accelerator

def param_sharding_enabled():
    from transformers.modeling_utils import is_deepspeed_zero3_enabled, is_fsdp_enabled
    return is_deepspeed_zero3_enabled() or is_fsdp_enabled()

def get_clean_data(full_prompts, full_responses, remove_bad=False):
    full_prompts_clean = []
    full_responses_clean = []
    for i, response in enumerate(full_responses):
        temp_resp = response.replace(full_prompts[i], '').strip().strip('\n\n----').strip('\n\n----- ').strip()
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        clean_resp = full_prompts[i] + temp_resp
        if remove_bad and (('.....' in clean_resp) or (clean_resp.count(':)') >= 3)):
            ## pass bad sample
            continue
        full_responses_clean.append(clean_resp)
        full_prompts_clean.append(full_prompts[i])
    return full_prompts_clean, full_responses_clean



# # for BeaverTails Better
# def generate_rejected_synthetic_data(
#     model,
#     tokenizer,
#     dataset,
#     output_name,
#     prompt_template="\n\nHuman: {raw_prompt}\n\nAssistant: ",
# ):

#     prompt_list = []
#     chosen_list = []
#     for i in tqdm(dataset):
#         if not i['is_safe']:
#             prompt = prompt_template.format(raw_prompt = i['prompt'])
#             prompt_list.append(prompt)
#             chosen_list.append(prompt + i['response'])
            
#     better_train = {'prompt' : prompt_train,
#                     'chosen' : chosen_train}
#     better_train_ds = Dataset.from_dict(better_train)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-3B',
        torch_dtype=torch.bfloat16,
        **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {})
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-3B',
    )

    prompt_template = DEFAULT_PROMPT_TEMPLATE = "\n\nHuman: {raw_prompt}\n\nAssistant: "

    beaver = load_dataset("PKU-Alignment/BeaverTails")

    prompt_train = []
    chosen_train = []
    for i in tqdm(beaver['330k_train']):
        if not i['is_safe']:
            prompt = prompt_template.format(raw_prompt = i['prompt'])
            prompt_train.append(prompt)
            chosen_train.append(prompt + i['response'])

    better_train = {'prompt' : prompt_train,
                    'chosen' : chosen_train}
    better_train_ds = Dataset.from_dict(better_train)

    prompt_test = []
    chosen_test = []
    for i in tqdm(beaver['330k_test']):
        if not i['is_safe']:
            prompt = prompt_template.format(raw_prompt = i['prompt'])
            prompt_test.append(prompt)
            chosen_test.append(prompt + i['response'])

    better_test = {'prompt' : prompt_test,
                    'chosen' : chosen_test}
    better_test_ds = Dataset.from_dict(better_test)

    train_dl = DataLoader(better_train_ds, batch_size = 1024, shuffle = False)
    test_dl = DataLoader(better_test_ds, batch_size = 768, shuffle = False)

    tokenizer.padding_side='left'
    tokenizer.pad_token=tokenizer.eos_token
    gen_config = GenerationConfig(
        max_new_tokens = 128,
    )

    # full_train_prompts = []
    # full_train_responses = []
    # with torch.no_grad():
    #     for i in tqdm(train_dl):
    #         full_train_prompts.extend(i['prompt'])
    #         encoding = tokenizer(i['prompt'], padding = True, return_tensors = 'pt').to('cuda')
    #         outputs = model.generate(generation_config = gen_config, **encoding)
    #         full_train_responses.extend(outputs.detach().cpu().tolist())
    

    # full_train_responses = tokenizer.batch_decode(full_train_responses, skip_special_tokens = True)
    # clean_train_prompts, clean_train_responses = get_clean_data(full_train_prompts, full_train_responses)
    # train_ds = Dataset.from_dict(
    #     {
    #         'prompt' : prompt_train,
    #         'chosen' : chosen_train,
    #         'rejected' : clean_train_responses
    #     }
    # )

    # train_ds.save_to_disk('./pku_better_train_llama3_init.hf')


    full_test_prompts = []
    full_test_responses = []
    with torch.no_grad():
        for i in tqdm(test_dl):
            full_test_prompts.extend(i['prompt'])
            encoding = tokenizer(i['prompt'], padding = True, return_tensors = 'pt').to('cuda')
            outputs = model.generate(generation_config = gen_config, **encoding)
            full_test_responses.extend(outputs.detach().cpu().tolist())

    full_test_responses = tokenizer.batch_decode(full_test_responses, skip_special_tokens = True)
    clean_test_prompts, clean_test_responses = get_clean_data(full_test_prompts, full_test_responses)
    test_ds = Dataset.from_dict(
        {
            'prompt' : prompt_test,
            'chosen' : chosen_test,
            'rejected' : clean_test_responses
        }
    )

    test_ds.save_to_disk('./pku_better_test_llama3_init.hf')