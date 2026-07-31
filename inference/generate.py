import torch, argparse
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()

def load_model_and_tokenizer(checkpoint_dir, tokenizer_path, device="auto"):
    from model import CodeMindLLM
    from tokenizer import CodeMindTokenizer
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    console.print(f"[green]Loading CodeMind on {device}[/green]")
    model = CodeMindLLM.from_pretrained(checkpoint_dir).to(device).eval()
    tokenizer = CodeMindTokenizer(tokenizer_path)
    return model, tokenizer, device

def generate_code(model, tokenizer, prompt, device, max_new_tokens=256, temperature=0.7, top_p=0.95, top_k=50):
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][len(ids):].tolist(), skip_special_tokens=True)

def interactive_mode(model, tokenizer, device):
    console.print(Panel("[bold cyan]CodeMind Interactive[/bold cyan]\nType prompt. 'exit' to quit.", title="👑"))
    while True:
        try:
            p = console.input("\n[yellow]>>> [/yellow]")
            if p.lower() in ("exit","quit"): break
            if not p.strip(): continue
            g = generate_code(model, tokenizer, p, device)
            console.print(Panel(Syntax(p+g,"python",theme="monokai"), title="Generated"))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="./tokenizer/vocab/tokenizer.json")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.tokenizer, args.device)
    if args.prompt:
        print(args.prompt + generate_code(model, tokenizer, args.prompt, device, args.max_tokens, args.temperature))
    else:
        interactive_mode(model, tokenizer, device)
