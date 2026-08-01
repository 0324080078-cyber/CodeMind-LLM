#!/usr/bin/env python3
"""
CodeMind AI Platform — Main Entry Point
Run: python codemind.py --help
"""

import sys
import os
import click
import secrets
import hashlib
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

BANNER = """
 ██████╗ ██████╗ ██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║██║████╗  ██║██╔══██╗
██║     ██║   ██║██║  ██║█████╗  ██╔████╔██║██║██╔██╗ ██║██║  ██║
██║     ██║   ██║██║  ██║██╔══╝  ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
╚██████╗╚██████╔╝██████╔╝███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
"""


def print_banner():
    console.print(Text(BANNER, style="bold cyan"))
    console.print(Panel(
        "[bold white]Your Lifetime AI Platform[/bold white]\n"
        "[yellow]100% Standalone • No External AI APIs • IDE Ready • Developer First[/yellow]\n"
        "[green]Built from scratch. Owned by you. Forever.[/green]",
        border_style="cyan",
        title="[bold cyan]👑 CodeMind AI Platform v2.0[/bold cyan]",
    ))


@click.group()
def cli():
    """CodeMind AI Platform — Your lifetime standalone AI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=8000)
@click.option("--workers", type=int, default=1)
@click.option("--no-vision", is_flag=True)
@click.option("--no-audio", is_flag=True)
@click.option("--device", default="auto")
def serve(host, port, workers, no_vision, no_audio, device):
    """Start the CodeMind API server."""
    print_banner()
    console.print(f"[bold green]Starting CodeMind on {host}:{port}[/bold green]")
    import uvicorn
    from api.v1.app import create_app
    from codemind_core.engine import CodeMindPlatform
    platform = CodeMindPlatform(device=device, enable_vision=not no_vision, enable_audio=not no_audio)
    app = create_app(platform)
    console.print(f"[cyan]API Docs: http://{host}:{port}/docs[/cyan]")
    console.print(f"[cyan]Web UI:   http://{host}:{port}/[/cyan]\n")
    uvicorn.run(app, host=host, port=port, workers=workers)


@cli.command()
@click.option("--config", default="configs/model_config.yaml")
def train(config):
    """Train the CodeMind language model."""
    print_banner()
    from train import main as train_main
    train_main(config)


@cli.command("generate-key")
@click.option("--name", prompt="API key name")
@click.option("--tier", type=click.Choice(["free","pro","enterprise"]), default="free")
def generate_key(name, tier):
    """Generate a new API key."""
    key = f"cm-{tier[:3]}-{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    os.makedirs("./api_keys", exist_ok=True)
    p = "./api_keys/keys.json"
    keys = []
    if os.path.exists(p):
        with open(p) as f:
            keys = json.load(f)
    keys.append({"name": name, "tier": tier, "key_hash": key_hash, "created": datetime.utcnow().isoformat(), "rate_limit": {"free":60,"pro":600,"enterprise":6000}[tier]})
    with open(p, "w") as f:
        json.dump(keys, f, indent=2)
    console.print(Panel(
        f"[bold green]API Key Generated![/bold green]\n\n"
        f"Name:  {name}\nTier:  {tier}\n"
        f"Key:   [bold cyan]{key}[/bold cyan]\n\n"
        f"[yellow]Save this — it won't be shown again![/yellow]\n\n"
        f"Usage: curl -H 'Authorization: Bearer {key}' http://localhost:8000/v1/chat/completions",
        title="New API Key",
    ))


@cli.command()
def status():
    """Check CodeMind system status."""
    print_banner()
    import torch
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    table.add_row("PyTorch", "✅ Ready", torch.__version__)
    if torch.cuda.is_available():
        table.add_row("GPU", "✅ Ready", f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    else:
        table.add_row("GPU", "⚠️  CPU Only", "Training will be slow — use Kaggle free GPU")
    for path, name, hint in [
        ("./checkpoints/checkpoint-best", "LLM Checkpoint", "Run: python codemind.py train"),
        ("./tokenizer/vocab/tokenizer.json", "Tokenizer", "Run: python codemind.py train"),
    ]:
        if os.path.exists(path):
            table.add_row(name, "✅ Found", path)
        else:
            table.add_row(name, "❌ Missing", hint)
    for pkg, label, hint in [
        ("diffusers", "Vision/StableDiffusion", "pip install diffusers accelerate"),
        ("whisper", "Audio/Whisper STT", "pip install openai-whisper"),
        ("TTS", "Audio/Coqui TTS", "pip install TTS"),
        ("chromadb", "Vector Memory", "pip install chromadb"),
    ]:
        try:
            m = __import__(pkg)
            table.add_row(label, "✅ Available", getattr(m, "__version__", "installed"))
        except ImportError:
            table.add_row(label, "⚠️  Not installed", hint)
    console.print(table)


@cli.command("install-ide")
@click.option("--ide", type=click.Choice(["vscode","neovim","all"]), default="all")
def install_ide(ide):
    """Install CodeMind IDE plugins."""
    import shutil
    import platform
    system = platform.system()
    if ide in ("vscode", "all"):
        if system == "Windows":
            ext = os.path.expanduser("~\\.vscode\\extensions\\codemind-ai-2.0.0")
        else:
            ext = os.path.expanduser("~/.vscode/extensions/codemind-ai-2.0.0")
        os.makedirs(ext, exist_ok=True)
        for f in ["extension.js", "extension.json"]:
            src = f"./ui/ide/vscode/{f}"
            if os.path.exists(src):
                shutil.copy2(src, ext)
        console.print(f"[green]VSCode: installed to {ext}[/green]")
        console.print("[yellow]Restart VSCode and set your API key in Settings > CodeMind[/yellow]")
    if ide in ("neovim", "all"):
        if system == "Windows":
            lua = os.path.expanduser("~\\AppData\\Local\\nvim\\lua")
        else:
            lua = os.path.expanduser("~/.config/nvim/lua")
        os.makedirs(lua, exist_ok=True)
        shutil.copy2("./ui/ide/neovim/codemind.lua", os.path.join(lua, "codemind.lua"))
        console.print(f"[green]Neovim: installed to {lua}/codemind.lua[/green]")
        console.print("[yellow]Add to init.lua: require('codemind').setup({ api_key = 'your-key' })[/yellow]")


if __name__ == "__main__":
    cli()
