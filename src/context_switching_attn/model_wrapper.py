import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper:
    def __init__(self, model_name: str, device: str="cpu"):
        self.device = device
        self.hooked = HookedTransformer.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.gen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.gen_model.eval()

    def log_probs(self, input_ids: torch.LongTensor):
        cache = self.hooked.run_with_cache(input_ids)
        logits = cache["logits", None]  # shape (B, T, V)
        return F.log_softmax(logits, dim=-1)

    def sequence_log_prob(self, input_ids: torch.LongTensor):
        cache = self.hooked.run_with_cache(input_ids)
        logits = cache["logits", None]  # (B, T, V)
        logps = F.log_softmax(logits, dim=-1)
        target_ids = input_ids[:,1:]
        token_logps = logps[:,:-1,:].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_logps.sum(dim=-1)  # (B,)

    def generate_greedy(self, prompts: list, max_new_tokens: int=50):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in out]