from torch.utils.data import Dataset
from datasets import load_dataset

class CodeCompletionDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        ds = load_dataset("mbpp", split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

        self.items = []
        for ex in ds:
            prompt = ex.get("text", ex.get("prompt", ""))
            code = ex.get("code", ex.get("snippet", ""))
            self.items.append({"prompt": prompt, "reference": code})
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]