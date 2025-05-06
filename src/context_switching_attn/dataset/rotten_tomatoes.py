from torch.utils.data import Dataset
from datasets import load_dataset

class RottenTomatoesDataset(Dataset):
    def __init__(self, split="test", num_examples=None):
        ds = load_dataset("rotten_tomatoes", split=split)
        if num_examples:
            ds = ds.select(range(num_examples))

        # Rotten Tomatoes is a 2-way sentiment: 0=negative, 1=positive
        choices = ["negative", "positive"]
        self.items = [
            {
                # include a natural cue for choice-based evaluation
                "prompt": ex["text"] + " Sentiment: ",
                "choices": choices,
                "label":   int(ex["label"])
            }
            for ex in ds
        ]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, i):
        return self.items[i]