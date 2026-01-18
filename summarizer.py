import json

from config import CHUNK_SIZE
from llm_client import call_llm

SYSTEM_PROMPT = """Ты — ассистент для анализа обсуждений в чатах.
Твоя задача — создавать и обновлять структурированные резюме.

Отвечай ТОЛЬКО валидным JSON без markdown-разметки."""

UPDATE_SUMMARY_PROMPT = """У тебя есть текущее резюме чата и новые сообщения.
Обнови резюме, добавив информацию из новых сообщений.

Текущее резюме:
{current_summary}

Новые сообщения (формат: [msg_id] автор: текст):
{messages}

Верни обновлённое резюме в JSON формате:
{{
    "overview": "общее описание того, о чём этот чат (2-3 предложения)",
    "problems": [
        {{"problem": "описание проблемы", "solution": "решение или null", "status": "solved/unsolved"}}
    ],
    "decisions": ["принятое решение 1", "принятое решение 2"],
    "key_points": ["важный факт 1", "важный факт 2"],
    "message_labels": {{"msg_id": 0, "msg_id2": 1}}
}}

Правила для message_labels:
- Укажи для каждого релевантного сообщения индекс проблемы (0, 1, 2...)
- Помечай только сообщения, которые относятся к какой-то проблеме
- msg_id — это число в квадратных скобках перед сообщением

Важно:
- Сохраняй существующую информацию, добавляй новую
- Если проблема решена в новых сообщениях — обнови её статус
- Не дублируй информацию
- Отвечай ТОЛЬКО JSON, без markdown"""

INITIAL_SUMMARY_PROMPT = """Проанализируй сообщения и создай структурированное резюме.

Сообщения (формат: [msg_id] автор: текст):
{messages}

Верни резюме в JSON формате:
{{
    "overview": "общее описание того, о чём этот чат (2-3 предложения)",
    "problems": [
        {{"problem": "описание проблемы", "solution": "решение или null", "status": "solved/unsolved"}}
    ],
    "decisions": ["принятое решение 1", "принятое решение 2"],
    "key_points": ["важный факт 1", "важный факт 2"],
    "message_labels": {{"msg_id": 0, "msg_id2": 1}}
}}

Правила для message_labels:
- Укажи для каждого релевантного сообщения индекс проблемы (0, 1, 2...)
- Помечай только сообщения, которые относятся к какой-то проблеме
- msg_id — это число в квадратных скобках перед сообщением

Важно: Отвечай ТОЛЬКО JSON, без markdown"""

QUERY_PROMPT = """На основе резюме чата ответь на вопрос пользователя.

Резюме чата:
{summary}

Вопрос: {question}

Ответь кратко и по делу на русском языке. Если информации нет в резюме, так и скажи."""


def format_messages(messages: list[dict]) -> str:
    """Format messages for LLM input with message IDs."""
    formatted = []
    for msg in messages:
        author = msg.get("author", "Unknown")
        text = msg.get("text", "")
        msg_id = msg.get("message_id", 0)
        if text.strip():
            formatted.append(f"[{msg_id}] {author}: {text}")
    return "\n".join(formatted)


def format_summary_for_display(summary: dict) -> str:
    """Format summary dict to readable text."""
    parts = []

    if summary.get("overview"):
        parts.append(f"ОБЗОР\n{summary['overview']}")

    if summary.get("problems"):
        parts.append("\nПРОБЛЕМЫ")
        for p in summary["problems"]:
            status = "решено" if p.get("status") == "solved" else "не решено"
            parts.append(f"• {p['problem']} [{status}]")
            if p.get("solution"):
                parts.append(f"  → {p['solution']}")

    if summary.get("decisions"):
        parts.append("\nРЕШЕНИЯ")
        for d in summary["decisions"]:
            parts.append(f"• {d}")

    if summary.get("key_points"):
        parts.append("\nКЛЮЧЕВЫЕ МОМЕНТЫ")
        for k in summary["key_points"]:
            parts.append(f"• {k}")

    return "\n".join(parts) if parts else "Резюме пока пустое."


def parse_llm_json(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        lines = lines[1:]  # Remove opening ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove closing ```
        response = "\n".join(lines)
    return json.loads(response)


def chunk_messages(
    messages: list[dict], chunk_size: int, overlap: int = 5
) -> list[list[dict]]:
    """Split messages into chunks with overlap."""
    if len(messages) <= chunk_size:
        return [messages]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(messages), step):
        chunk = messages[i : i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(messages):
            break
    return chunks


async def update_summary_single(current_summary: dict, messages: list[dict]) -> dict:
    """Update summary with a single batch of messages."""
    formatted_messages = format_messages(messages)

    # Keep existing message_labels to merge later
    existing_labels = current_summary.get("message_labels", {})

    # Don't send message_labels to LLM (too much data)
    summary_for_llm = {
        k: v for k, v in current_summary.items() if k != "message_labels"
    }

    if not current_summary.get("overview"):
        prompt = INITIAL_SUMMARY_PROMPT.format(messages=formatted_messages)
    else:
        prompt = UPDATE_SUMMARY_PROMPT.format(
            current_summary=json.dumps(summary_for_llm, ensure_ascii=False, indent=2),
            messages=formatted_messages,
        )

    response = await call_llm(prompt, SYSTEM_PROMPT)
    new_summary = parse_llm_json(response)

    # Merge message_labels: existing + new
    new_labels = new_summary.get("message_labels", {})
    # Convert all keys to strings for consistency
    merged_labels = {str(k): v for k, v in existing_labels.items()}
    merged_labels.update({str(k): v for k, v in new_labels.items()})
    new_summary["message_labels"] = merged_labels

    return new_summary


async def update_summary(
    current_summary: dict,
    new_messages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    on_progress: callable = None,
) -> dict:
    """Update summary with new messages, processing in batches."""
    if not new_messages:
        return current_summary

    chunks = chunk_messages(new_messages, chunk_size)
    summary = current_summary

    for i, chunk in enumerate(chunks):
        if on_progress:
            await on_progress(i + 1, len(chunks))
        summary = await update_summary_single(summary, chunk)

    return summary


async def answer_query(summary: dict, question: str) -> str:
    """Answer a question based on the summary."""
    summary_text = format_summary_for_display(summary)
    prompt = QUERY_PROMPT.format(summary=summary_text, question=question)
    return await call_llm(prompt, SYSTEM_PROMPT)


# Legacy function for compatibility
async def summarize_thread(messages: list[dict]) -> str:
    """Create a one-time summary (legacy)."""
    empty_summary = {"overview": "", "problems": [], "decisions": [], "key_points": []}
    result = await update_summary(empty_summary, messages)
    return format_summary_for_display(result)
