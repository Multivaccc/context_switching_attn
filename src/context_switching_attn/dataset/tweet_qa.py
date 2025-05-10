from torch.utils.data import Dataset

class TweetQADataset(Dataset):
    def __init__(self, split="test", num_examples=5):
        from datasets import load_dataset
        from itertools import islice
        ds_stream = load_dataset(
            "ucsbnlp/tweet_qa",
            split=split,
            streaming=True,
        )
        examples = list(islice(ds_stream, num_examples))

        self.items = []
        for ex in examples:
            tweet    = ex.get("Tweet", "")
            question = ex.get("Question", "")
            answers  = ex.get("Answer", [])
            prompt   = f"{tweet} Question: {question}"
            self.items.append({"prompt": prompt, "reference": answers})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
