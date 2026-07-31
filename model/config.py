from dataclasses import dataclass, field
import yaml, json, os

@dataclass
class CodeMindConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        return cls(**{k: v for k, v in m.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            cfg = json.load(f)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_range": self.initializer_range,
            "use_cache": self.use_cache,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "head_dim": self.head_dim,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
