import os, sys, torch, yaml, argparse, random, numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main(config_path="configs/model_config.yaml"):
    console.print(Panel("[bold cyan]CodeMind LLM Training[/bold cyan]\nYour AI. No API. No limits.", title="👑"))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Device: {device}[/green]")

    tok_path = "./tokenizer/vocab/tokenizer.json"
    if not os.path.exists(tok_path):
        console.print("[yellow]Training tokenizer...[/yellow]")
        from tokenizer.train_tokenizer import train_code_tokenizer
        train_code_tokenizer(vocab_size=cfg["tokenizer"]["vocab_size"], output_dir="./tokenizer/vocab", num_training_samples=200000)

    from tokenizer import CodeMindTokenizer
    tokenizer = CodeMindTokenizer(tok_path)
    console.print(f"Tokenizer: {len(tokenizer)} tokens")

    from model import CodeMindConfig, CodeMindLLM
    mcfg = CodeMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg["model"]["hidden_size"],
        num_hidden_layers=cfg["model"]["num_hidden_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        intermediate_size=cfg["model"]["intermediate_size"],
        max_position_embeddings=cfg["model"]["max_position_embeddings"],
        dropout=cfg["model"]["dropout"],
        attention_dropout=cfg["model"]["attention_dropout"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = CodeMindLLM(mcfg).to(device)

    from data import create_dataloaders
    tl, vl = create_dataloaders(tokenizer=tokenizer, batch_size=cfg["training"]["batch_size"], max_seq_length=cfg["model"]["max_position_embeddings"], languages=cfg["data"]["languages"], num_workers=cfg["data"]["num_workers"], max_train_samples=500000, max_val_samples=2000)

    from training import get_optimizer, get_cosine_schedule_with_warmup, CodeMindTrainer
    opt = get_optimizer(model, learning_rate=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, cfg["training"]["warmup_steps"], cfg["training"]["max_steps"])
    trainer = CodeMindTrainer(model=model, optimizer=opt, scheduler=sched, train_loader=tl, val_loader=vl, config=cfg["training"], device=device)
    trainer.train()
    console.print("[bold green]Training complete![/bold green]")
    console.print("Run: python -m inference.generate --checkpoint ./checkpoints/checkpoint-best")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
