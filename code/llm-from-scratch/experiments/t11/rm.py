# Scoring (prompt, reply) texts with a small open reward model: Skywork-Reward-V2-Qwen3-0.6B (Apache 2.0),
# a Bradley-Terry reward model on Qwen3-0.6B. BF16, no system prompt, as its card recommends.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RM = 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'
REV = '8c14a4e9e6321deaf572544339b16b8d6bbe8886'


class RewardModel:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(RM, revision=REV)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RM, revision=REV, dtype=torch.bfloat16, num_labels=1).cuda().eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.config.pad_token_id = self.tok.pad_token_id

    def encode(self, prompt, reply):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt},
                                             {"role": "assistant", "content": reply}], tokenize=False)
        if self.tok.bos_token is not None and text.startswith(self.tok.bos_token):
            text = text[len(self.tok.bos_token):]
        return self.tok(text, add_special_tokens=False)["input_ids"][:4096]

    @torch.no_grad()
    def score(self, pairs, budget=16384):
        """pairs: list of (prompt text, reply text) -> list of scalar rewards."""
        enc = [self.encode(p, r) for p, r in pairs]
        order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
        out = [None] * len(enc)
        i = 0
        while i < len(order):
            T = len(enc[order[i]])
            grp = order[i:i + max(1, budget // T)]
            ids = torch.full((len(grp), T), self.tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(grp), T), dtype=torch.long)
            for r, k in enumerate(grp):              # right padding; the model reads the last real token
                ids[r, :len(enc[k])] = torch.tensor(enc[k]); att[r, :len(enc[k])] = 1
            logits = self.model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, 0].float()
            for r, k in enumerate(grp):
                out[k] = logits[r].item()
            i += len(grp)
        return out
