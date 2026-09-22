"""
Orchestrator — brain of the multi-agent startup system.
Receives a task from CEO, builds a DAG plan, runs agents, synthesizes results.
Supports iterative dialogue: each run has access to full conversation history.
"""

import json
import uuid
import asyncio
import httpx
from typing import Any
from datetime import datetime
from pathlib import Path

from agents.researcher import ResearcherAgent
from agents.cto import CTOAgent
from agents.product import ProductAgent
from agents.marketing import MarketingAgent
from agents.sales import SalesAgent
from agents.guard import GuardAgent
from memory.context_store import ContextStore
from memory.embedding_store import EmbeddingStore

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
# terminus: strict function calling + Think/Non-Think modes → reliable JSON plans
ORCHESTRATOR_MODEL = "deepseek-ai/deepseek-v3.1-terminus"
FALLBACK_MODEL = "deepseek-ai/deepseek-v3.2"

SYSTEM_PROMPT = """You are the orchestrator of a startup multi-agent startup system.
Available agents: researcher, cto, product, marketing, sales, guard.

Agent responsibilities:
- researcher: market size, competitors, ICP
- cto: tech stack, MVP architecture, effort estimates
- product: PRD, user stories, roadmap
- marketing: positioning, GTM, content strategy
- sales: outbound sequences, discovery scripts, pipeline setup
- guard: quality and safety check (always last)

When given a task:
1. Check the conversation history — avoid re-running agents whose results are still valid
2. Decompose into subtasks, assign to agents, define dependencies (DAG)
3. Return a JSON plan — nothing else

JSON plan format:
{
  "goal": "short description of the overall goal",
  "tasks": [
    {
      "id": "t1",
      "agent": "researcher",
      "instruction": "specific instruction for this agent",
      "depends_on": [],
      "context_keys": []
    }
  ]
}

Rules:
- Never solve the task yourself — only plan and synthesize
- Tasks with no dependencies can run in parallel
- Use context_keys to pass previous task results to dependent tasks
- If the CEO is refining a previous task, reuse cached results where possible (fewer tasks = faster)
- Always end with guard as the last task
"""

SYNTHESIS_PROMPT = """You are synthesizing results from multiple agents into a final report for the CEO.

Conversation history (previous runs):
{history}

Results from agents this run:
{results}

Create a structured report:
1. Executive Summary (3-5 sentences)
2. Key Findings (from each agent)
3. Recommended Next Steps
4. Decisions Required from CEO (explicit list)

If this is a follow-up/refinement, explicitly note what changed vs previous analysis.
Be concise and actionable. The CEO is busy.
Respond in Russian."""


class Orchestrator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.context = ContextStore()
        self.embeddings = EmbeddingStore(api_key)
        self._session_id: str | None = None
        # Conversation history persists across multiple run() calls in interactive mode
        self._history: list[dict] = []
        self.agents = {
            "researcher": ResearcherAgent(api_key),
            "cto": CTOAgent(api_key),
            "product": ProductAgent(api_key),
            "marketing": MarketingAgent(api_key),
            "sales": SalesAgent(api_key),
            "guard": GuardAgent(api_key),
        }
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

    async def _call_llm(self, messages: list[dict], model: str = None) -> str:
        model = model or ORCHESTRATOR_MODEL
        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{NVIDIA_API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": model, "messages": messages, "max_tokens": 2048},
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if resp.status_code != 200 and model != FALLBACK_MODEL:
                        return await self._call_llm(messages, FALLBACK_MODEL)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
                except httpx.TimeoutException:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError("Orchestrator LLM failed after 3 attempts")

    def _history_summary(self) -> str:
        if not self._history:
            return "No previous runs."
        parts = []
        for i, turn in enumerate(self._history, 1):
            parts.append(f"[Run {i}] Task: {turn['task']}\nSummary: {turn['summary'][:400]}")
        return "\n\n".join(parts)

    async def _build_plan(self, task: str) -> dict:
        print(f"\n[Orchestrator] Building plan for: {task[:80]}...")
        history_ctx = self._history_summary()
        content = await self._call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation history:\n{history_ctx}\n\nNew task: {task}"},
        ])
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        plan = json.loads(clean)
        self._log("plan", plan)
        return plan

    async def _run_task(self, task_def: dict, resolved_context: dict) -> str:
        agent_name = task_def["agent"]
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        instruction = task_def["instruction"]
        if resolved_context:
            ctx_text = "\n\n".join(f"[{k}]:\n{v}" for k, v in resolved_context.items())
            instruction = f"{instruction}\n\nКонтекст от предыдущих агентов:\n{ctx_text}"

        print(f"  -> Запуск [{agent_name}]: {task_def['instruction'][:60]}...")
        result = await agent.run(instruction)
        self.context.store(task_def["id"], result)
        try:
            await self.embeddings.add(task_def["id"], result)
        except Exception:
            pass
        return result

    async def _execute_dag(self, tasks: list[dict]) -> dict[str, str]:
        completed: dict[str, str] = {}
        pending = {t["id"]: t for t in tasks}

        while pending:
            ready = [
                t for t in pending.values()
                if all(dep in completed for dep in t.get("depends_on", []))
            ]
            if not ready:
                raise RuntimeError("Circular dependency detected in task graph")

            results = await asyncio.gather(
                *[
                    self._run_task(
                        t,
                        {k: completed[k] for k in t.get("context_keys", []) if k in completed},
                    )
                    for t in ready
                ],
                return_exceptions=True,
            )

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    print(f"  x [{task['agent']}] failed: {result}")
                    completed[task["id"]] = f"ERROR: {result}"
                else:
                    completed[task["id"]] = result
                del pending[task["id"]]

        return completed

    async def _synthesize(self, goal: str, results: dict[str, str], tasks: list[dict]) -> str:
        print("\n[Orchestrator] Синтез результатов...")
        agent_by_id = {t["id"]: t["agent"] for t in tasks}
        results_text = "\n\n---\n\n".join(
            f"Агент: {agent_by_id[tid].upper()} (задача {tid})\n{res}"
            for tid, res in results.items()
        )
        report = await self._call_llm([
            {"role": "system", "content": SYNTHESIS_PROMPT.format(
                history=self._history_summary(),
                results=results_text,
            )},
            {"role": "user", "content": f"Общая цель: {goal}"},
        ])
        return report

    async def recall(self, query: str, top_k: int = 3) -> str:
        """Semantic search over all past agent outputs in current session."""
        results = await self.embeddings.search(query, top_k=top_k)
        if not results:
            return "Пока нет сохранённых результатов. Сначала запусти задачу."
        lines = [f"Семантический поиск: '{query}'\n"]
        for r in results:
            lines.append(f"-- [{r['key']}] (релевантность: {r['score']:.0%})")
            lines.append(r["text"][:600] + ("..." if len(r["text"]) > 600 else ""))
            lines.append("")
        return "\n".join(lines)

    async def run(self, task: str) -> str:
        start = datetime.now()
        self._session_id = str(uuid.uuid4())[:8]
        self.embeddings.session_id = self._session_id
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Задача получена в {start.strftime('%H:%M:%S')} (сессия: {self._session_id})")
        if self._history:
            print(f"[Orchestrator] История: {len(self._history)} предыдущих запусков")
        print(f"{'='*60}")

        plan = await self._build_plan(task)
        print(f"\n[План] Цель: {plan['goal']}")
        print(f"[План] Задачи: {len(plan['tasks'])} — {[t['agent'] for t in plan['tasks']]}")

        results = await self._execute_dag(plan["tasks"])
        report = await self._synthesize(plan["goal"], results, plan["tasks"])

        # Guard check
        guard = self.agents["guard"]
        safety_note = await guard.check(report)
        if safety_note:
            report += f"\n\n[Guard]: {safety_note}"

        elapsed = (datetime.now() - start).seconds
        report += f"\n\n---\n_Сгенерировано за {elapsed}с, {len(plan['tasks'])} агентов_"

        # Save to conversation history for next iterative run
        self._history.append({
            "task": task,
            "summary": report[:600],
            "session_id": self._session_id,
            "ts": start.isoformat(),
        })

        self._log("report", {"task": task, "report": report})
        print(f"\n[Orchestrator] Готово за {elapsed}с")
        return report

    def _log(self, name: str, data: Any):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.logs_dir / f"{ts}_{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
