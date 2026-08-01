"""
CodeMind Interactive CLI
Chat with your AI from the terminal
"""

import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

console = Console()


def detect_language(text: str) -> str:
    langs = {
        "python": ["def ","import ","class ","print("],
        "javascript": ["function ","const ","let ","var ","=>"],
        "typescript": ["interface ","type ","enum "],
        "rust": ["fn ","let mut","impl ","use "],
        "go": ["func ","package ","import ("],
        "java": ["public class","void ","System.out"],
        "cpp": ["#include","std::","cout"],
        "bash": ["#!/bin/bash","echo ","$"],
        "sql": ["SELECT","INSERT","CREATE TABLE","DROP"],
        "html": ["<html","<div","<!DOCTYPE"],
    }
    text_lower = text.lower()
    for lang, patterns in langs.items():
        if any(p.lower() in text_lower for p in patterns):
            return lang
    return "text"


def format_response(response: str) -> None:
    """Format and display the AI response."""
    lang = detect_language(response)

    if lang != "text" and len(response) > 50:
        syntax = Syntax(response, lang, theme="monokai", line_numbers=True, word_wrap=True)
        console.print(Panel(syntax, title=f"[bold green]CodeMind ({lang})[/bold green]", border_style="green"))
    else:
        console.print(Panel(Markdown(response), title="[bold green]CodeMind[/bold green]", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description="CodeMind CLI")
    parser.add_argument("--checkpoint", default="./checkpoints/checkpoint-best")
    parser.add_argument("--tokenizer", default="./tokenizer/vocab/tokenizer.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]CodeMind AI CLI[/bold cyan]\n"
        "Your personal AI assistant — 100% local\n"
        "Type 'help' for commands, 'exit' to quit",
        title="👑 CodeMind",
        border_style="cyan",
    ))

    from core.engine import CodeMindEngine
    engine = CodeMindEngine(
        model_checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    history = []
    COMMANDS = {
        "/image": "Generate an image: /image a sunset over mountains",
        "/game": "Generate a game: /game build a snake game",
        "/web": "Build a website: /web create a todo app with React",
        "/security": "Security tool: /security port scanner",
        "/devops": "DevOps config: /devops docker-compose for nginx+postgres",
        "/clear": "Clear conversation history",
        "/save": "Save last response to file",
        "/help": "Show this help",
        "/exit": "Exit CodeMind",
    }

    last_response = ""

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            # Commands
            if user_input == "/exit" or user_input.lower() == "exit":
                console.print("[yellow]Goodbye, warrior! 👑[/yellow]")
                break

            if user_input == "/clear":
                history.clear()
                console.print("[green]History cleared.[/green]")
                continue

            if user_input == "/help":
                for cmd, desc in COMMANDS.items():
                    console.print(f"  [cyan]{cmd}[/cyan] — {desc}")
                continue

            if user_input.startswith("/save"):
                fname = user_input.split(" ", 1)[1] if " " in user_input else "codemind_output.txt"
                with open(fname, "w") as f:
                    f.write(last_response)
                console.print(f"[green]Saved to {fname}[/green]")
                continue

            if user_input.startswith("/image"):
                prompt = user_input[7:].strip()
                console.print("[yellow]Generating image...[/yellow]")
                path = engine.generate_image(prompt)
                console.print(f"[green]Image saved: {path}[/green]")
                continue

            # Chat
            history.append({"role": "user", "content": user_input})

            console.print("[dim]Thinking...[/dim]")

            response = engine.chat(
                messages=history,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            content = response["choices"][0]["message"]["content"]
            last_response = content
            history.append({"role": "assistant", "content": content})

            format_response(content)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
