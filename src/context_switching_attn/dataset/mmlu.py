from typing import Dict, List, Optional

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class MMLUDataset(Dataset):
    """Dataset class for MMLU (Massive Multitask Language Understanding)."""

    def __init__(
        self,
        split: str = "test",
        subjects: str = "all",
        num_examples: Optional[int] = None,
        num_concatenated: int = 1,
        max_total_samples: Optional[int] = None,
    ):
        """
        Initialize MMLU dataset.

        Args:
            split: Dataset split to load ('train', 'validation', or 'test')
            subjects: Subject or list of subjects to include
            num_examples: Maximum number of examples to load (before concatenation)
            num_concatenated: Number of examples to concatenate into one (1 means no concatenation)
            max_total_samples: Maximum total number of concatenated samples to create
        """
        # Load raw dataset
        ds = load_dataset("cais/mmlu", subjects, split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # For GPT2

        # Tokenize examples
        def tokenize(example):
            prompt = example["question"] + " " + " ".join(example["choices"])
            encoding = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": example["answer"],  # already an int
                "subject": example["subject"],
            }

        self.dataset = ds.map(tokenize)
        self.num_concatenated = num_concatenated
        self.max_total_samples = max_total_samples

        if self.num_concatenated > 1:
            self.concatenated_indices = self._create_concatenated_indices()
        else:
            self.concatenated_indices = None

    def _create_concatenated_indices(self) -> List[List[int]]:
        indices: List[List[int]] = []
        dataset_size = len(self.dataset)
        for i in range(0, dataset_size - self.num_concatenated + 1):
            indices.append(list(range(i, i + self.num_concatenated)))
            if self.max_total_samples and len(indices) >= self.max_total_samples:
                break
        return indices

    def __len__(self) -> int:
        if self.num_concatenated > 1:
            return len(self.concatenated_indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_concatenated > 1:
            sample_indices = self.concatenated_indices[idx]
            samples = [self.dataset[i] for i in sample_indices]

            input_ids = torch.cat([torch.tensor(s["input_ids"]) for s in samples])
            attention_mask = torch.cat([torch.tensor(s["attention_mask"]) for s in samples])
            labels = torch.tensor([s["labels"] for s in samples])
            subjects = [s["subject"] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "subjects": subjects,
                "num_concatenated": self.num_concatenated,
            }
        else:
            item = self.dataset[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"]),
                "attention_mask": torch.tensor(item["attention_mask"]),
                "labels": torch.tensor(item["labels"]),
                "subjects": [item["subject"]],
                "num_concatenated": 1,
            }

def get_mmlu_dataloader(
    split: str = "test",
    subjects: str = "all",
    num_examples: Optional[int] = None,
    num_concatenated: int = 1,
    max_total_samples: Optional[int] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for the MMLU dataset.
    """
    dataset = MMLUDataset(
        split=split,
        subjects=subjects,
        num_examples=num_examples,
        num_concatenated=num_concatenated,
        max_total_samples=max_total_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
