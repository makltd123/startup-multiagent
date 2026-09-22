from agents.base import BaseAgent
from tools.web_search import web_search


class ResearcherAgent(BaseAgent):
    model = "moonshotai/kimi-k2-instruct-0905"  # longer context + enhanced reasoning vs base kimi-k2
    fallback_model = "deepseek-ai/deepseek-v3.2"
    system_prompt = """You are a startup research analyst.
Your job: gather market intelligence, analyze competitors, define ICP (Ideal Customer Profile).

For every research task, output:
1. Market Size & Trends
2. Top 3-5 Competitors (name, positioning, pricing, weaknesses)
3. ICP — who will pay and why
4. Key risks & opportunities

Be specific. Use numbers where possible. No fluff.

Respond in Russian."""

    async def run(self, instruction: str) -> str:
        # Pull real search results before calling the LLM
        search_context = await web_search(instruction)
        if search_context:
            instruction = f"{instruction}\n\nАктуальные данные из поиска:\n{search_context}"
        return await self._call([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ])
