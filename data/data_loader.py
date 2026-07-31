import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import List, Optional
from datasets import load_dataset

class CodeDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_length=1024, languages=None, max_samples=None, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.languages = languages or ["Python","JavaScript","Java","C++"]
        self.max_samples = max_samples

    def __iter__(self):
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train", trust_remote_code=True)
        count = 0
        for s in ds:
            if self.max_samples and count >= self.max_samples: break
            if s.get("language","") not in self.languages: continue
            code = s.get("code","")
            if not code or len(code) < 50: continue
            ids = self.tokenizer.encode(code, add_special_tokens=True, max_length=None, truncation=False)
            for i in range(0, len(ids), self.max_seq_length):
                chunk = ids[i:i+self.max_seq_length]
                if len(chunk) < 16: continue
                if len(chunk) < self.max_seq_length:
                    chunk = chunk + [self.tokenizer.pad_token_id]*(self.max_seq_length-len(chunk))
                iids = torch.tensor(chunk, dtype=torch.long)
                lbls = iids.clone()
                lbls[lbls == self.tokenizer.pad_token_id] = -100
                yield {"input_ids": iids, "labels": lbls, "attention_mask": (iids != self.tokenizer.pad_token_id).long()}
                count += 1
                if self.max_samples and count >= self.max_samples: return

def create_dataloaders(tokenizer, batch_size=8, max_seq_length=1024, languages=None, num_workers=2, max_train_samples=100000, max_val_samples=2000):
    train_ds = CodeDataset(tokenizer=tokenizer, max_seq_length=max_seq_length, languages=languages, max_samples=max_train_samples)
    val_ds = CodeDataset(tokenizer=tokenizer, max_seq_length=max_seq_length, languages=languages, max_samples=max_val_samples, seed=9999)
    return DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True), DataLoader(val_ds, batch_size=batch_size, num_workers=1, pin_memory=True)
