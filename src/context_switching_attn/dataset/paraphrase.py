from torch.utils.data import Dataset
from datasets import load_dataset

class ParaphraseDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        split_spec = f"{split}[:{num_examples}]" if num_examples else split
        ds = load_dataset("paws", "labeled_final", split=split_spec)
        ds = ds.filter(lambda ex: ex["label"] == 1)
        self.items = [
            {
                "prompt":    ex["sentence1"],
                "reference": ex["sentence2"],
            }
            for ex in ds
        ]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]