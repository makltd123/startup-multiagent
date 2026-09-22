"""Base class for all agents."""

import asyncio
import httpx

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
_MAX_RETRIES = 3


class BaseAgent:
    model: str = ""
    fallback_model: str = ""
    system_prompt: str = ""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def _call(self, messages: list[dict], model: str = None) -> str:
        primary = model or self.model
        async with httpx.AsyncClient(timeout=90) as client:
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = await client.post(
                        f"{NVIDIA_API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": primary, "messages": messages, "max_tokens": 4096},
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if resp.status_code != 200 and self.fallback_model:
                        resp = await client.post(
                            f"{NVIDIA_API_BASE}/chat/completions",
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json={"model": self.fallback_model, "messages": messages, "max_tokens": 4096},
                        )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"]
                except httpx.TimeoutException:
                    if attempt == _MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"Failed after {_MAX_RETRIES} attempts (model: {primary})")

    async def run(self, instruction: str) -> str:
        return await self._call([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ])
