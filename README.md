# Startup Multi-Agent System

**English** | [Русский](README.ru.md)

A command-line "team" of LLM agents for a solo startup founder. You play the CEO and give a task; an orchestrator breaks it into a graph of subtasks, specialist agents work on them — in parallel where possible — and you get one combined report. No agent frameworks: just `httpx` and `asyncio` on top of free NVIDIA NIM endpoints.

The agents answer in Russian.

## How it works

```
task ──► Orchestrator: plan as a JSON graph of subtasks
           ──► agents run level by level: parallel within a level,
               earlier results passed on through context_keys
           ──► Orchestrator: combined report
           ──► Guard check ──► report (saved to logs/)
```

| Role | Model (NVIDIA NIM) | Fallback | Job |
|---|---|---|---|
| Orchestrator | `deepseek-ai/deepseek-v3.1-terminus` | `deepseek-v3.2` | plans the graph, writes the final report |
| Researcher | `moonshotai/kimi-k2-instruct-0905` | `deepseek-v3.2` | market, competitors, ideal customer profile; gets DuckDuckGo results first |
| CTO | `qwen/qwen3-coder-480b-a35b-instruct` | `devstral-2-123b` | stack, MVP architecture, effort estimates |
| Product | `mistralai/mistral-large-3-675b-instruct-2512` | `kimi-k2-thinking` | PRD, user stories, roadmap |
| Marketing | `mistralai/mistral-nemotron` | `mistral-large-3` | positioning, go-to-market, content |
| Sales | `mistralai/mistral-large-3-675b-instruct-2512` | `kimi-k2-instruct` | outbound sequences, discovery scripts, pipeline |
| Guard | `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | `llama-guard-4-12b` | checks the final report |

Every agent result is also saved to memory: a key-value store (`memory/context.json`) and an embedding store (`memory/embeddings.json`, `llama-3.2-nemoretriever-300m-embed-v1`). In interactive mode `recall: <question>` searches the current session's results by meaning, and the planner sees a summary of earlier runs so it can skip work that is still valid.

## Running it

```bash
pip install -r requirements.txt
echo NVIDIA_API_KEY=nvapi-... > .env     # free key at build.nvidia.com

python main.py "Analyze the B2B HR automation market and draft an MVP plan"
python main.py --interactive
```

Interactive commands: any task text, `recall: <question>`, `history`, `quit`.

## Limitations

- **Guard is a safety filter, not a reviewer.** Its prompt asks for factual consistency and completeness, but both models are safety classifiers that answer "safe" / "unsafe" with a category, so only unsafe content gets flagged.
- The plan is parsed from the model's JSON answer (markdown fences are stripped), so an unusual answer format breaks the run.
- Free NVIDIA NIM endpoints are rate-limited (429 is retried with backoff), and the available models change over time.
- The run history the planner sees lives only in the running process. The memory files persist between runs, but `recall` searches only the current session.

## License

[MIT](LICENSE)

---

Author: [Fedor Makovskiy](https://fedor.alqira.ru), AI developer. Services and projects with quality metrics: [fedor.alqira.ru](https://fedor.alqira.ru).
