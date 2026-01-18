import asyncio
import base64
import logging

import httpx

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_VISION_MODEL

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def call_llm(prompt: str, system_prompt: str = None, max_retries: int = 3) -> str:
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
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_MODEL,
                        "messages": messages,
                        "temperature": 0.3,
                    },
                )
                response.raise_for_status()
                data = response.json()
                result = data["choices"][0]["message"]["content"]

                # Debug logging response
                logger.debug(f"[LLM RESPONSE]\n{result}")
                logger.debug("=" * 60)

                return result
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.HTTPStatusError,
            ) as exc:
                if attempt == max_retries - 1:
                    raise
                backoff = 2**attempt
                logger.warning(
                    "OpenRouter request failed (%s). Retrying in %ss...",
                    exc.__class__.__name__,
                    backoff,
                )
                await asyncio.sleep(backoff)


async def analyze_image(
    image_bytes: bytes,
    prompt: str,
    system_prompt: str | None = None,
    mime_type: str = "image/jpeg",
) -> str:
    """Call OpenRouter vision model with an image and prompt."""
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{image_b64}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENROUTER_VISION_MODEL,
                        "messages": messages,
                        "temperature": 0.3,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.HTTPStatusError,
            ) as exc:
                if attempt == 2:
                    raise
                backoff = 2**attempt
                logger.warning(
                    "OpenRouter vision request failed (%s). Retrying in %ss...",
                    exc.__class__.__name__,
                    backoff,
                )
                await asyncio.sleep(backoff)
