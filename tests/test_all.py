import pytest
from src.context_switching_attn.dataset.mmlu import MMLUDataset
from src.context_switching_attn.dataset.rotten_tomatoes import RottenTomatoesDataset
from src.context_switching_attn.dataset.gigaword import GigawordDataset
from src.context_switching_attn.dataset.tweet_qa import TweetQADataset
from src.context_switching_attn.dataset.code_completion import CodeCompletionDataset
from src.context_switching_attn.dataset.paraphrase import ParaphraseDataset
from src.context_switching_attn.model_wrapper import ModelWrapper

@pytest.mark.parametrize("cls,kwargs,keys", [
    (MMLUDataset, {"split":"test","subjects":"abstract_algebra","num_examples":5}, {"prompt","choices","label"}),
    (CodeCompletionDataset, {"split":"test","num_examples":5}, {"prompt","reference"}),
    (GigawordDataset, {"split":"test","num_examples":5}, {"prompt","reference"}),
    (ParaphraseDataset, {"split":"test","num_examples":10}, {"prompt","reference"}),
    (RottenTomatoesDataset, {"split":"test","num_examples":10}, {"prompt","choices","label"}),
    (TweetQADataset, {"split":"test","num_examples":10}, {"prompt","reference"}),
])

def test_dataset_shapes(cls, kwargs, keys):
    ds = cls(**kwargs)
    assert len(ds) == kwargs["num_examples"]
    item = ds[0]
    assert set(keys).issubset(item.keys())

def test_model_wrapper_smoke():
    mw = ModelWrapper("gpt2", device="cpu")
    out = mw.generate_greedy(["Hello"], max_new_tokens=5)
    assert isinstance(out, list) and len(out)==1
    ids = mw.tokenizer("test", return_tensors="pt").input_ids
    lp = mw.sequence_log_prob(ids)
    assert lp.numel() == 1
