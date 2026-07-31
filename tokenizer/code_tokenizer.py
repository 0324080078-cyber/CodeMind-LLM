import os
from typing import List, Optional
from tokenizers import Tokenizer

class CodeMindTokenizer:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer not found: {path}\nRun: python -m tokenizer.train_tokenizer")
        self.tokenizer = Tokenizer.from_file(path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        v = self.tokenizer.get_vocab()
        self.pad_token_id = v.get("<pad>", 0)
        self.bos_token_id = v.get("<bos>", 1)
        self.eos_token_id = v.get("<eos>", 2)
        self.unk_token_id = v.get("<unk>", 3)
        self.vocab_size = len(v)

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=True):
        if add_special_tokens: text = f"<bos>{text}"
        ids = self.tokenizer.encode(text).ids
        if max_length and truncation and len(ids) > max_length:
            ids = ids[:max_length-1]
            if add_special_tokens: ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    @classmethod
    def from_pretrained(cls, d):
        return cls(os.path.join(d, "tokenizer.json"))

    def __len__(self): return self.vocab_size
    def __repr__(self): return f"CodeMindTokenizer(vocab_size={self.vocab_size})"
