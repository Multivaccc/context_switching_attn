from torch.utils.data import Dataset
import datasets

class RottenTomatoesDataset(Dataset):
    def __init__(self, split="test", num_examples=5):
        # always limit to exactly num_examples
        split_spec = f"{split}[:{num_examples}]"
        ds = datasets.load_dataset("cornell-movie-review-data/rotten_tomatoes", split=split_spec)
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