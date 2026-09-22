from agents.base import BaseAgent


class GuardAgent(BaseAgent):
    model = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"  # multilingual → handles Russian agent outputs
    fallback_model = "meta/llama-guard-4-12b"
    system_prompt = """You are a quality and safety reviewer.
Review the given content and check:
1. Factual consistency (no internal contradictions)
2. Completeness (are key sections missing?)
3. Safety (no harmful, misleading, or legally risky content)

If everything is fine, respond with an empty string.
If issues found, respond with a short bullet list of concerns only.

Respond in Russian."""

    async def check(self, content: str) -> str:
        result = await self.run(f"Review this content:\n\n{content}")
        if not result.strip() or result.strip().lower() in ("", "safe", "no issues", "ok"):
            return ""
        return result
