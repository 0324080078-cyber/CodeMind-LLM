"""
CodeMind Master Server
Starts the full AI system with all capabilities:
- LLM text/code generation
- Image generation (Stable Diffusion)
- Full stack web app generation
- Game development
- Security research tools
- DevOps automation
- OpenAI-compatible API

Run: python codemind_server.py
Then use at: http://localhost:8000
OpenAI client: openai.api_base = "http://localhost:8000/v1"
"""

import os
import sys
import argparse
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_banner():
    console.print(Panel(
        "[bold cyan]CodeMind AI — Full System[/bold cyan]\n"
        "[yellow]Your own AI. No API costs. No limits.[/yellow]\n"
        "[green]OpenAI-compatible • Multi-modal • All languages[/green]",
        title="👑 CodeMind v2",
        border_style="cyan",
    ))

    table = Table(title="Available Capabilities")
    table.add_column("Feature", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Endpoint", style="yellow")

    table.add_row("Chat/Code Generation", "✅ Active", "/v1/chat/completions")
    table.add_row("Image Generation", "✅ Active", "/v1/images/generations")
    table.add_row("Full Stack Apps", "✅ Active", "/v1/chat/completions")
    table.add_row("Game Development", "✅ Active", "/v1/chat/completions")
    table.add_row("Security Research", "✅ Active", "/v1/chat/completions")
    table.add_row("DevOps Tools", "✅ Active", "/v1/chat/completions")
    table.add_row("50+ Languages", "✅ Active", "/v1/completions")
    table.add_row("OpenAI Compatible", "✅ Active", "/v1/*")

    console.print(table)
    console.print("\n[bold green]API Docs: http://localhost:8000/docs[/bold green]")
    console.print("[bold yellow]OpenAI Client: set api_base='http://localhost:8000/v1'[/bold yellow]\n")


def main():
    parser = argparse.ArgumentParser(description="CodeMind AI Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default="./checkpoints/checkpoint-best")
    parser.add_argument("--tokenizer", default="./tokenizer/vocab/tokenizer.json")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-agents", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print_banner()

    # Initialize engine
    console.print("[yellow]Loading CodeMind Engine...[/yellow]")
    from core.engine import CodeMindEngine
    engine = CodeMindEngine(
        model_checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        enable_vision=not args.no_vision,
        enable_agents=not args.no_agents,
    )

    # Create API app
    from api.openai_compatible import create_api_app
    app, state = create_api_app(engine=engine)
    state["engine"] = engine

    console.print(f"[bold green]Starting server at http://{args.host}:{args.port}[/bold green]")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
