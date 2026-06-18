from __future__ import annotations
import json
import os
from pathlib import Path
import typer
import time
from dotenv import load_dotenv
from rich import print

load_dotenv()

# Set default model if not configured
if "GEMINI_MODEL" not in os.environ:
    os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"

from src.reflexion_lab import mock_runtime
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = typer.Option("data/hotpot_mini.json", "--dataset", help="Path to the dataset JSON file"),
    out_dir: str = typer.Option("outputs/sample_run", "--out-dir", help="Output directory path"),
    reflexion_attempts: int = typer.Option(3, "--reflexion-attempts", help="Maximum attempts for Reflexion"),
    live: bool = typer.Option(False, "--live", help="Run with live LLM (Gemini API)"),
    model: str = typer.Option("gemini-3.1-flash-lite", "--model", help="Gemini model to use"),
    rpm: float = typer.Option(15.0, "--rpm", help="Request per minute limit for Gemini API"),
    limit: int = typer.Option(None, "--limit", help="Limit number of examples to evaluate")
) -> None:
    # Configure global runtime mode
    if live or os.environ.get("LIVE_MODE", "false").lower() in ("true", "1", "yes"):
        mock_runtime.LIVE_MODE = True
        os.environ["LIVE_MODE"] = "true"
        
    os.environ["GEMINI_MODEL"] = model
    os.environ["GEMINI_RPM"] = str(rpm)
    
    print(f"Loading dataset from: {dataset}...")
    examples = load_dataset(dataset)
    
    if limit is not None:
        print(f"Limiting evaluation to the first {limit} examples.")
        examples = examples[:limit]
        
    print(f"Running ReAct Agent in [cyan]{'LIVE' if mock_runtime.LIVE_MODE else 'MOCK'}[/cyan] mode...")
    react = ReActAgent()
    react_records = []
    for i, example in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] Running ReAct on question {example.qid}...")
        react_records.append(react.run(example))
        if mock_runtime.LIVE_MODE and i < len(examples) - 1:
            time.sleep(3.0)
        
    print(f"Running Reflexion Agent in [cyan]{'LIVE' if mock_runtime.LIVE_MODE else 'MOCK'}[/cyan] mode...")
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    reflexion_records = []
    for i, example in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] Running Reflexion on question {example.qid}...")
        reflexion_records.append(reflexion.run(example))
        if mock_runtime.LIVE_MODE and i < len(examples) - 1:
            time.sleep(3.0)
        
    all_records = react_records + reflexion_records
    
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    
    mode_str = "live" if mock_runtime.LIVE_MODE else "mock"
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode_str)
    json_path, md_path = save_report(report, out_path)
    
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
