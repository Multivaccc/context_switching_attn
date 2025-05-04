from torch.utils.data import Dataset
from datasets import load_dataset

class RottenTomatoesDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        ds = load_dataset("rotten_tomatoes", split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

        self.items = [{"prompt": ex["text"], "label": int(ex["label"])} for ex in ds]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]