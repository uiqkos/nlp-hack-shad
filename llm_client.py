import logging

import httpx

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def call_llm(prompt: str, system_prompt: str = None) -> str:
    """Call OpenRouter API with the given prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Debug logging
    logger.debug("=" * 60)
    logger.debug(f"LLM REQUEST to {OPENROUTER_MODEL}")
    logger.debug("=" * 60)
    if system_prompt:
        logger.debug(f"[SYSTEM PROMPT]\n{system_prompt}")
        logger.debug("-" * 40)
    logger.debug(f"[USER PROMPT]\n{prompt}")
    logger.debug("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
            },
        )
        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"]

        # Debug logging response
        logger.debug(f"[LLM RESPONSE]\n{result}")
        logger.debug("=" * 60)

        return result
