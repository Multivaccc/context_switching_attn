from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader


class MMLUDataset(Dataset):
    """Dataset class for MMLU (Massive Multitask Language Understanding)."""
    
    def __init__(
        self,
        split: str = "test",
        subjects: str = "all",
        num_concatenated: int = 1,
        max_samples_per_subject: Optional[int] = None,
        max_total_samples: Optional[int] = None,
    ):
        """
        Initialize MMLU dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation', or 'test')
            subjects: Subject or list of subjects to include
            num_concatenated: Number of examples to concatenate into one (1 means no concatenation)
            max_samples_per_subject: Maximum number of samples to load per subject
            max_total_samples: Maximum total number of concatenated samples to create
        """
        self.split = split
        self.subjects = subjects
        self.num_concatenated = num_concatenated
        self.max_samples_per_subject = max_samples_per_subject
        
        # Load the dataset
        self.dataset = load_dataset("cais/mmlu", self.subjects, split=self.split)
        
        # Create indices for concatenated samples if needed
        if num_concatenated > 1:
            self.concatenated_indices = self._create_concatenated_indices(max_total_samples)
        else:
            self.concatenated_indices = None
    
    def _create_concatenated_indices(
        self, max_total_samples: Optional[int] = None
    ) -> List[List[int]]:
        """Create lists of indices to concatenate."""
        indices = []
        dataset_size = len(self.dataset)
        
        # Create all possible combinations of num_concatenated indices
        for i in range(0, dataset_size - self.num_concatenated + 1):
            indices.append(list(range(i, i + self.num_concatenated)))
            if max_total_samples and len(indices) >= max_total_samples:
                break
                
        return indices
    
    def __len__(self) -> int:
        if self.num_concatenated > 1:
            return len(self.concatenated_indices)
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.num_concatenated > 1:
            # Get indices for this concatenated sample
            sample_indices = self.concatenated_indices[idx]
            
            # Get all individual samples
            samples = [self.dataset[i] for i in sample_indices]
            
            # Concatenate input_ids and attention_mask
            input_ids = torch.cat([s["input_ids"] for s in samples])
            attention_mask = torch.cat([s["attention_mask"] for s in samples])
            
            # Keep track of labels and subjects
            labels = torch.stack([s["labels"] for s in samples])
            subjects = [s["subject"] for s in samples]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "subjects": subjects,
                "num_concatenated": self.num_concatenated,
            }
        else:
            # Single example case
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
    num_concatenated: int = 1,
    max_samples_per_subject: Optional[int] = None,
    max_total_samples: Optional[int] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for the MMLU dataset.
    
    Args:
        split: Dataset split to load ('train', 'validation', or 'test')
        subjects: Subject or list of subjects to include
        num_concatenated: Number of examples to concatenate into one (1 means no concatenation)
        max_samples_per_subject: Maximum number of samples to load per subject
        max_total_samples: Maximum total number of concatenated samples to create
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance for the MMLU dataset
    """
    dataset = MMLUDataset(
        split=split,
        subjects=subjects,
        num_concatenated=num_concatenated,
        max_samples_per_subject=max_samples_per_subject,
        max_total_samples=max_total_samples,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

