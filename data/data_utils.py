# adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py#L82
from dataclasses import dataclass
from typing import Dict, Optional

from datasets import load_dataset, load_from_disk
from abc import ABC, abstractmethod

from datasets import Dataset, concatenate_datasets


DEFAULT_PROMPT_TEMPLATE = "\n\nHuman:\n{raw_prompt}\n\nAssistant:\n"

@dataclass
class RawDatasetPreprocessor(ABC):
    path: Optional[str] = None
    prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    sanity_check: Optional[bool] = False
    num_proc: Optional[int] = 4

    @abstractmethod
    def _get_raw_dataset(self, split) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        # return {
        #     "raw_prompt": str, # optional, useful for generation
        #     "prompt":   str,
        #     "chosen":   str,
        #     "rejected": str,
        # }
        raise NotImplementedError

    def get_preference_dataset(self, split) -> Dataset:
        """
        return a dataset of texts with three keys "prompt", "chosen", "rejected", ("raw_prompt", optional)
        """
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: dataset = dataset.select(range(min(len(dataset), 100)))
        print("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)

    def get_sft_dataset(self, split, chosen_only=True):
        """
        return a dataset of texts with two keys "prompt", "response", ("raw_prompt", optional)
        """
        print("mapping preference to sft...")
        dataset = self.get_preference_dataset(split)
        # chosen_only_dataset = dataset.rename_column("chosen", "response")
        chosen_only_dataset = dataset.remove_columns("rejected").rename_column("chosen", "response")
        if chosen_only:
            return chosen_only_dataset
        rejected_only_dataset = dataset.remove_columns("chosen").rename_column("rejected", "response")
        return concatenate_datasets([chosen_only_dataset, rejected_only_dataset])

    def get_dwa_dataset(self, split):
        """
        return a dataset of texts with two keys "prompt", "response", "is_expert" ("raw_prompt", optional)
        """
        print("mapping preference to dwa...")
        dataset = self.get_preference_dataset(split)
        return dataset

def preprocess_anthropic_prompt_and_response(prompt_and_response):
    prompt_and_response = prompt_and_response.replace("\n\nHuman: ", "\n\nHuman:\n")
    prompt_and_response = prompt_and_response.replace("\n\nAssistant: ", "\n\nAssistant:\n")
    return prompt_and_response

def extract_anthropic_prompt_and_response(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


@dataclass
class HhRlhfRDP(RawDatasetPreprocessor):
    path: Optional[str] = "Anthropic/hh-rlhf"

    # def __post_init__(self):
    #     assert self.prompt_template == "\n\nHuman: {prompt}\n\nAssistant:"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_dataset(self.path, split="train").train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return load_dataset(self.path, split="test")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["chosen"]   = preprocess_anthropic_prompt_and_response(example["chosen"])
        example["rejected"] = preprocess_anthropic_prompt_and_response(example["rejected"])
        prompt = extract_anthropic_prompt_and_response(example["chosen"])
        return {
            "prompt":   prompt,
            "chosen":   example["chosen"][len(prompt) :],
            "rejected": example["rejected"][len(prompt) :],
        }


@dataclass
class PKUSyntheticRDP(RawDatasetPreprocessor):
    path: Optional[str] = "../data/pku_safe_rejected_llama3init.hf"

    # def __post_init__(self):
    #     assert self.prompt_template == "\n\nHuman: {prompt}\n\nAssistant:"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_from_disk(self.path)['train'].train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_from_disk(self.path)['train'].train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return load_from_disk(self.path)['test']
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["chosen"]   = preprocess_anthropic_prompt_and_response(example["chosen"])
        example["rejected"] = preprocess_anthropic_prompt_and_response(example["rejected"])
        prompt = extract_anthropic_prompt_and_response(example["chosen"])
        return {
            "prompt":   prompt,
            "chosen":   example["chosen"][len(prompt) :],
            "rejected": example["rejected"][len(prompt) :],
        }
        
@dataclass
class PKUSyntheticRewardRDP(RawDatasetPreprocessor):
    path: Optional[str] = "../data/pku_safe_with_reward_split.hf"

    # def __post_init__(self):
    #     assert self.prompt_template == "\n\nHuman: {prompt}\n\nAssistant:"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_from_disk(self.path)['train'].train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_from_disk(self.path)['train'].train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return load_from_disk(self.path)['test']
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["text"]   = preprocess_anthropic_prompt_and_response(example["text"])
        prompt = extract_anthropic_prompt_and_response(example["text"])
        return {
            "prompt":   prompt,
            "chosen":   example["text"][len(prompt) :],
            "rejected": example["text"][len(prompt) :],
        }

@dataclass
class PKUSyntheticDWARDP(RawDatasetPreprocessor):
    path: Optional[str] = "../data/pku_safe_with_reward_split.hf"

    # def __post_init__(self):
    #     assert self.prompt_template == "\n\nHuman: {prompt}\n\nAssistant:"

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_from_disk(self.path)['train'].train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return load_from_disk(self.path)['train'].train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return load_from_disk(self.path)['test']
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["text"]   = preprocess_anthropic_prompt_and_response(example["text"])
        prompt = extract_anthropic_prompt_and_response(example["text"])
        return {
            "prompt": prompt,
            "response": example["text"][len(prompt) :],
            "is_expert": example["is_expert"],
        }

if __name__ == "__main__":
    # train_dataset      = HhRlhfRDP(num_proc=1).get_preference_dataset(split="train")
    # validation_dataset = HhRlhfRDP(num_proc=1).get_preference_dataset(split="validation")
    # test_dataset       = HhRlhfRDP(num_proc=1).get_preference_dataset(split="test")
    sft_dataset = PKUSyntheticRDP(num_proc=1).get_sft_dataset(split='train')
