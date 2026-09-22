#!/usr/bin/env python3
"""
CLI entry point for the startup multi-agent system.

Usage:
    python main.py "Analyze the market for B2B HR automation SaaS and build an MVP plan"
    python main.py --interactive
"""

import asyncio
import argparse
import os
import sys
from dotenv import load_dotenv
from orchestrator import Orchestrator

load_dotenv()


def get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("Error: NVIDIA_API_KEY environment variable not set.")
        print("Get your free key at: https://build.nvidia.com")
        sys.exit(1)
    return key


async def run_once(task: str):
    orch = Orchestrator(api_key=get_api_key())
    report = await orch.run(task)
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)

    # Save report to file
    from datetime import datetime
    from pathlib import Path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("logs") / f"{ts}_report.md"
    out.write_text(f"# Task\n{task}\n\n# Report\n{report}", encoding="utf-8")
    print(f"\n[Saved to {out}]")


async def run_interactive():
    orch = Orchestrator(api_key=get_api_key())
    print("\nStartup Agent System — Interactive Mode")
    print("Commands: task | recall: <запрос> | history | quit\n")
    while True:
        try:
            task = input("CEO > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if task.lower() in ("quit", "exit", "q"):
            break
        if not task:
            continue
        # Semantic recall shortcut: "recall: what did we learn about pricing?"
        if task.lower() == "history":
            if not orch._history:
                print("\nИстория пуста.\n")
            else:
                for i, turn in enumerate(orch._history, 1):
                    print(f"\n[{i}] {turn['ts'][:19]} — {turn['task'][:80]}")
            continue
        if task.lower().startswith("recall:"):
            query = task[7:].strip()
            result = await orch.recall(query)
            print("\n" + result)
            continue
        report = await orch.run(task)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Startup Multi-Agent System")
    parser.add_argument("task", nargs="?", help="Task for the agent system")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.task:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_once(args.task))


if __name__ == "__main__":
    main()
