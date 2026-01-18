import json
from dataclasses import asdict

from config import CHUNK_SIZE
from database import (
    Message,
    Problem,
    get_chat_meta,
    get_messages_for_problem,
    get_problem_by_id,
    get_problems_by_chat,
    link_messages_to_problem,
    save_chat_meta,
    save_problem,
    update_problem_status,
)
from llm_client import call_llm

SYSTEM_PROMPT = """Ты — ассистент для анализа обсуждений в чатах.
Твоя задача — создавать и обновлять структурированные резюме.

Если в сообщении есть блок <IMAGE_LIST>...</IMAGE_LIST>, это список изображений,
а внутри него находятся блоки <IMAGE>...</IMAGE>. Это результаты анализа картинок,
а не прямой текст пользователя. Внутри каждого <IMAGE> формат:
<IMAGE_DESC>...</IMAGE_DESC> — краткое описание изображения,
<IMAGE_TEXT>...</IMAGE_TEXT> — распознанный текст (может быть пустым).
Относись к этим данным как к содержимому изображения и учитывай это в резюме.

Отвечай ТОЛЬКО валидным JSON без markdown-разметки."""

ANALYZE_MESSAGES_PROMPT = """Проанализируй новые сообщения из чата и определи:
1. Какие проблемы обсуждаются (новые или обновления существующих)
2. Общий контекст обсуждения

Существующие проблемы в чате:
{existing_problems}

Новые сообщения (формат: [msg_id] автор: текст):
{messages}

Верни JSON:
{{
    "new_problems": [
        {{
            "title": "краткое название проблемы",
            "short_summary": "1-2 предложения о сути",
            "long_summary": "подробное описание проблемы и контекста",
            "status": "solved/unsolved",
            "message_ids": [123, 456]
        }}
    ],
    "problem_updates": [
        {{
            "problem_id": 1,
            "new_status": "solved/unsolved",
            "additional_summary": "новая информация для добавления",
            "message_ids": [789]
        }}
    ],
    "overview_update": "если нужно обновить общее описание чата",
    "new_decisions": ["новое решение если есть"],
    "new_key_points": ["новый важный факт если есть"]
}}

Правила:
- message_ids — это числа в квадратных скобках [msg_id] перед сообщениями
- Блоки <IMAGE_LIST>...</IMAGE_LIST> содержат список блоков <IMAGE>...</IMAGE>
- Блоки <IMAGE>...</IMAGE> означают данные с изображения, а не прямой текст автора
- Формат внутри <IMAGE>: <IMAGE_DESC>...</IMAGE_DESC>, <IMAGE_TEXT>...</IMAGE_TEXT>
- Если сообщение относится к существующей проблеме — добавь в problem_updates
- Если это новая проблема — добавь в new_problems
- Сообщение может относиться к нескольким проблемам
- Если проблема решена в сообщениях — обнови статус на "solved"
- Отвечай ТОЛЬКО JSON"""

QUERY_PROMPT = """На основе информации о чате ответь на вопрос пользователя.

Общий обзор чата:
{overview}

Проблемы:
{problems}

Ключевые решения: {decisions}
Важные факты: {key_points}

Если в сообщениях есть блоки <IMAGE_LIST>...</IMAGE_LIST>, это результаты анализа изображений.
Внутри списка находятся блоки <IMAGE>...</IMAGE> с форматом:
<IMAGE_DESC>...</IMAGE_DESC> — описание изображения,
<IMAGE_TEXT>...</IMAGE_TEXT> — распознанный текст (может быть пустым).

Вопрос: {question}

Ответь кратко и по делу на русском языке. Если информации нет, так и скажи."""

SUMMARIZE_PROBLEM_PROMPT = """Создай подробное резюме проблемы на основе связанных сообщений.

Название проблемы: {title}
Текущее описание: {current_summary}

Сообщения по этой проблеме:
{messages}

Верни JSON:
{{
    "short_summary": "1-2 предложения о сути проблемы",
    "long_summary": "подробное описание: что за проблема, какой контекст, какие решения предлагались, текущий статус",
    "status": "solved/unsolved"
}}

Отвечай ТОЛЬКО JSON."""


def format_messages(messages: list[Message]) -> str:
    """Форматировать сообщения для LLM."""
    formatted = []
    for msg in messages:
        author = msg.author_name or msg.author_tag or "Unknown"
        text = msg.text.strip()
        if text:
            if "<IMAGE_LIST>" in text:
                text = f"(В сообщении есть изображения. Данные ниже.)\n{text}"
            formatted.append(f"[{msg.telegram_msg_id}] {author}: {text}")
    return "\n".join(formatted)


def format_messages_from_dicts(messages: list[dict]) -> str:
    """Форматировать сообщения из словарей (для совместимости)."""
    formatted = []
    for msg in messages:
        author = msg.get("author_name") or msg.get("author", "Unknown")
        text = msg.get("text", "")
        msg_id = msg.get("telegram_msg_id") or msg.get("message_id", 0)
        text = text.strip()
        if text:
            if "<IMAGE_LIST>" in text:
                text = f"(В сообщении есть изображения. Данные ниже.)\n{text}"
            formatted.append(f"[{msg_id}] {author}: {text}")
    return "\n".join(formatted)


def format_problems_for_llm(problems: list[Problem]) -> str:
    """Форматировать проблемы для контекста LLM."""
    if not problems:
        return "Пока нет зафиксированных проблем."

    lines = []
    for p in problems:
        status = "решено" if p.status == "solved" else "не решено"
        lines.append(f"[ID:{p.id}] {p.title} [{status}]")
        if p.short_summary:
            lines.append(f"   {p.short_summary}")
    return "\n".join(lines)


def format_summary_for_display(chat_id: int) -> str:
    """Форматировать резюме чата для отображения пользователю."""
    meta = get_chat_meta(chat_id)
    problems = get_problems_by_chat(chat_id)

    parts = []

    if meta.get("overview"):
        parts.append(f"📋 ОБЗОР\n{meta['overview']}")

    if problems:
        parts.append("\n🔧 ПРОБЛЕМЫ")
        for i, p in enumerate(problems):
            status_icon = "✅" if p.status == "solved" else "❌"
            parts.append(f"{i}. {status_icon} {p.title}")
            if p.short_summary:
                parts.append(f"   {p.short_summary}")

    if meta.get("decisions"):
        parts.append("\n📌 РЕШЕНИЯ")
        for d in meta["decisions"]:
            parts.append(f"• {d}")

    if meta.get("key_points"):
        parts.append("\n💡 КЛЮЧЕВЫЕ МОМЕНТЫ")
        for k in meta["key_points"]:
            parts.append(f"• {k}")

    return (
        "\n".join(parts)
        if parts
        else "Резюме пока пустое. Напишите сообщения и используйте /summarize"
    )


def parse_llm_json(response: str) -> dict:
    """Парсинг JSON из ответа LLM."""
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    return json.loads(response)


def chunk_messages(messages: list, chunk_size: int, overlap: int = 5) -> list[list]:
    """Разбить сообщения на чанки с перекрытием."""
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


async def analyze_and_update(
    chat_id: int, new_messages: list[Message], on_progress: callable = None
) -> dict:
    """
    Анализировать новые сообщения и обновить БД.
    Возвращает статистику изменений.
    """
    if not new_messages:
        return {"new_problems": 0, "updated_problems": 0}

    existing_problems = get_problems_by_chat(chat_id)
    meta = get_chat_meta(chat_id)

    chunks = chunk_messages(new_messages, CHUNK_SIZE)

    stats = {"new_problems": 0, "updated_problems": 0}

    for i, chunk in enumerate(chunks):
        if on_progress:
            await on_progress(i + 1, len(chunks))

        formatted_messages = format_messages(chunk)
        formatted_problems = format_problems_for_llm(existing_problems)

        prompt = ANALYZE_MESSAGES_PROMPT.format(
            existing_problems=formatted_problems, messages=formatted_messages
        )

        response = await call_llm(prompt, SYSTEM_PROMPT)
        result = parse_llm_json(response)

        # Создаем маппинг telegram_msg_id -> db_id для сообщений в чанке
        msg_id_map = {msg.telegram_msg_id: msg.id for msg in chunk}

        # Обрабатываем новые проблемы
        for new_prob in result.get("new_problems", []):
            problem = Problem(
                id=None,
                chat_id=chat_id,
                title=new_prob["title"],
                short_summary=new_prob.get("short_summary", ""),
                long_summary=new_prob.get("long_summary", ""),
                status=new_prob.get("status", "unsolved"),
            )
            problem_id = save_problem(problem)
            stats["new_problems"] += 1

            # Связываем сообщения с проблемой
            msg_db_ids = [
                msg_id_map[mid]
                for mid in new_prob.get("message_ids", [])
                if mid in msg_id_map
            ]
            if msg_db_ids:
                link_messages_to_problem(msg_db_ids, problem_id)

            # Добавляем в список существующих для следующих чанков
            problem.id = problem_id
            existing_problems.append(problem)

        # Обрабатываем обновления проблем
        for update in result.get("problem_updates", []):
            problem_id = update.get("problem_id")
            if not problem_id:
                continue

            problem = get_problem_by_id(problem_id)
            if not problem:
                continue

            # Обновляем статус если изменился
            new_status = update.get("new_status")
            if new_status and new_status != problem.status:
                update_problem_status(problem_id, new_status)
                stats["updated_problems"] += 1

            # Связываем новые сообщения
            msg_db_ids = [
                msg_id_map[mid]
                for mid in update.get("message_ids", [])
                if mid in msg_id_map
            ]
            if msg_db_ids:
                link_messages_to_problem(msg_db_ids, problem_id)

        # Обновляем метаданные чата
        overview = result.get("overview_update") or meta.get("overview", "")
        decisions = list(
            set(meta.get("decisions", []) + result.get("new_decisions", []))
        )
        key_points = list(
            set(meta.get("key_points", []) + result.get("new_key_points", []))
        )

        save_chat_meta(chat_id, overview, decisions, key_points)
        meta = {"overview": overview, "decisions": decisions, "key_points": key_points}

    return stats


async def regenerate_problem_summary(problem_id: int) -> Problem:
    """Пересоздать резюме проблемы на основе связанных сообщений."""
    problem = get_problem_by_id(problem_id)
    if not problem:
        raise ValueError(f"Problem {problem_id} not found")

    messages = get_messages_for_problem(problem_id)
    if not messages:
        return problem

    formatted = format_messages(messages)
    prompt = SUMMARIZE_PROBLEM_PROMPT.format(
        title=problem.title, current_summary=problem.long_summary, messages=formatted
    )

    response = await call_llm(prompt, SYSTEM_PROMPT)
    result = parse_llm_json(response)

    problem.short_summary = result.get("short_summary", problem.short_summary)
    problem.long_summary = result.get("long_summary", problem.long_summary)
    problem.status = result.get("status", problem.status)

    save_problem(problem)
    return problem


async def answer_query(chat_id: int, question: str) -> str:
    """Ответить на вопрос по резюме чата."""
    meta = get_chat_meta(chat_id)
    problems = get_problems_by_chat(chat_id)

    if not meta.get("overview") and not problems:
        return "Резюме пока пустое. Сначала используйте /summarize"

    problems_text = []
    for p in problems:
        status = "решено" if p.status == "solved" else "не решено"
        problems_text.append(f"• {p.title} [{status}]\n  {p.short_summary}")

    prompt = QUERY_PROMPT.format(
        overview=meta.get("overview", "Нет общего описания"),
        problems="\n".join(problems_text) if problems_text else "Нет проблем",
        decisions=", ".join(meta.get("decisions", [])) or "Нет",
        key_points=", ".join(meta.get("key_points", [])) or "Нет",
        question=question,
    )

    return await call_llm(prompt, SYSTEM_PROMPT)


# ============== LEGACY COMPATIBILITY ==============


async def update_summary(
    current_summary: dict,
    new_messages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    on_progress: callable = None,
) -> dict:
    """Legacy функция для совместимости."""
    # Конвертируем dict в Message объекты если нужно
    messages = []
    for msg in new_messages:
        if isinstance(msg, Message):
            messages.append(msg)
        else:
            messages.append(
                Message(
                    id=None,
                    chat_id=0,  # Will be set properly in bot.py
                    telegram_msg_id=msg.get("message_id", 0),
                    text=msg.get("text", ""),
                    author_tag=msg.get("author_tag", ""),
                    author_name=msg.get("author", ""),
                    reply_to_msg_id=msg.get("reply_to_msg_id"),
                    telegram_link=None,
                )
            )

    # Для legacy вызовов просто возвращаем старый формат
    return current_summary


async def summarize_thread(messages: list[dict]) -> str:
    """Legacy функция."""
    return "Use /summarize command instead"
