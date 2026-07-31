import os, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from datasets import load_dataset

def get_code_iterator(n=200000):
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train", trust_remote_code=True)
    count = 0
    for s in ds:
        if count >= n: break
        c = s.get("code","")
        if c and len(c) > 50:
            yield c
            count += 1
    print(f"Tokenizer: {count} samples used")

def train_code_tokenizer(vocab_size=32000, output_dir="./tokenizer/vocab", num_training_samples=200000):
    os.makedirs(output_dir, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    special = ["<pad>","<bos>","<eos>","<unk>","<sep>","<mask>","<comment>","<indent>","<dedent>","<nl>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special, show_progress=True, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(get_code_iterator(num_training_samples), trainer=trainer)
    path = os.path.join(output_dir, "tokenizer.json")
    tok.save(path)
    v = tok.get_vocab()
    with open(os.path.join(output_dir,"vocab.json"),"w") as f:
        json.dump(v, f, indent=2)
    print(f"Tokenizer saved. Vocab: {len(v)}")
    return tok

if __name__ == "__main__":
    train_code_tokenizer()
