from torch.utils.data import Dataset
from datasets import load_dataset

class ParaphraseDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        ds = load_dataset("paws", "labeled_final", split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

        self.items = [{"prompt": ex["sentence1"], "reference": ex["sentence2"]} for ex in ds if ex["label"] == 1]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]