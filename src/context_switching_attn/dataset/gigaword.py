from torch.utils.data import Dataset
from datasets import load_dataset

class GigawordDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        ds = load_dataset("gigaword", split=split, trust_remote_code=True)
        if num_examples:
            ds = ds.select(range(num_examples))

        self.items = [{"prompt": ex["document"], "reference": ex["summary"]} for ex in ds]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]