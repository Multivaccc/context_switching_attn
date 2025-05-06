from torch.utils.data import Dataset
from datasets import load_dataset

class ParaphraseDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        ds = load_dataset("paws", "labeled_final", split=split)

        ds = ds.filter(lambda ex: ex["label"] == 1)

        if num_examples is not None:
            ds = ds.select(range(num_examples))

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