import json

from config import CHUNK_SIZE, CONTEXT_MESSAGES_PER_PROBLEM
from database import (
    Message,
    Problem,
    get_chat_meta,
    get_message_by_telegram_id,
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

Отвечай ТОЛЬКО валидным JSON без markdown-разметки."""

ANALYZE_MESSAGES_PROMPT = """Проанализируй новые сообщения из чата и определи:
1. Какие проблемы обсуждаются (новые или обновления существующих)
2. Общий контекст обсуждения

=== КОНТЕКСТ СУЩЕСТВУЮЩИХ ПРОБЛЕМ ===
{problems_context}

=== НОВЫЕ СООБЩЕНИЯ ДЛЯ АНАЛИЗА ===
Формат: [id] имя (reply:id_ответа): текст

{messages}

=== ИНСТРУКЦИИ ===
Верни JSON:
{{
    "new_problems": [
        {{
            "title": "краткое название проблемы (3-7 слов)",
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
            "additional_summary": "новая информация для добавления к описанию",
            "message_ids": [789]
        }}
    ],
    "overview_update": "обновлённое общее описание чата (или null если не нужно)",
    "new_decisions": ["новое решение если есть"],
    "new_key_points": ["новый важный факт если есть"]
}}

Правила:
- message_ids — это числа [id] в начале каждого сообщения
- reply:X означает что сообщение — ответ на сообщение с id=X (используй для понимания контекста)
- Если сообщение относится к существующей проблеме — добавь в problem_updates с правильным problem_id
- Если это новая тема/проблема — добавь в new_problems
- Одно сообщение может относиться к нескольким проблемам
- Если проблема решена — обнови статус на "solved"
- Отвечай ТОЛЬКО валидным JSON"""

QUERY_PROMPT = """На основе информации о чате ответь на вопрос пользователя.

Общий обзор чата:
{overview}

Проблемы:
{problems}

Ключевые решения: {decisions}
Важные факты: {key_points}

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


def format_message_for_llm(msg: Message) -> str:
    """Форматировать одно сообщение для LLM с reply info."""
    author = msg.author_name or msg.author_tag or "Unknown"
    reply_part = f" (reply:{msg.reply_to_msg_id})" if msg.reply_to_msg_id else ""
    return f"[{msg.telegram_msg_id}] {author}{reply_part}: {msg.text}"


def format_messages_with_context(messages: list[Message], chat_id: int) -> str:
    """
    Форматировать сообщения для LLM, добавляя reply-сообщения для контекста.
    """
    # Собираем все msg_id в текущем чанке
    chunk_msg_ids = {msg.telegram_msg_id for msg in messages}

    # Находим все reply_to_msg_id, которых нет в текущем чанке
    needed_reply_ids = set()
    for msg in messages:
        if msg.reply_to_msg_id and msg.reply_to_msg_id not in chunk_msg_ids:
            needed_reply_ids.add(msg.reply_to_msg_id)

    # Загружаем недостающие сообщения из БД
    context_messages = []
    for reply_id in needed_reply_ids:
        reply_msg = get_message_by_telegram_id(chat_id, reply_id)
        if reply_msg:
            context_messages.append(reply_msg)

    # Сортируем контекстные сообщения по id
    context_messages.sort(key=lambda m: m.telegram_msg_id)

    formatted_parts = []

    # Сначала добавляем контекстные сообщения с пометкой
    if context_messages:
        formatted_parts.append("--- Контекст (сообщения на которые есть ответы) ---")
        for msg in context_messages:
            formatted_parts.append(format_message_for_llm(msg))
        formatted_parts.append("--- Новые сообщения ---")

    # Затем основные сообщения
    for msg in messages:
        if msg.text.strip():
            formatted_parts.append(format_message_for_llm(msg))

    return "\n".join(formatted_parts)


def format_problems_context(problems: list[Problem], chat_id: int) -> str:
    """
    Форматировать проблемы с последними сообщениями для контекста.
    """
    if not problems:
        return "Пока нет зафиксированных проблем."

    parts = []
    for p in problems:
        status = "РЕШЕНО" if p.status == "solved" else "НЕ РЕШЕНО"
        parts.append(f"[problem_id:{p.id}] {p.title} [{status}]")
        parts.append(f"  Описание: {p.short_summary}")

        # Получаем последние N сообщений для этой проблемы
        problem_messages = get_messages_for_problem(p.id)
        if problem_messages:
            last_msgs = problem_messages[-CONTEXT_MESSAGES_PER_PROBLEM:]
            parts.append(f"  Последние сообщения:")
            for msg in last_msgs:
                author = msg.author_name or "Unknown"
                text_preview = (
                    msg.text[:100] + "..." if len(msg.text) > 100 else msg.text
                )
                parts.append(f"    [{msg.telegram_msg_id}] {author}: {text_preview}")
        parts.append("")  # Пустая строка между проблемами

    return "\n".join(parts)


def format_summary_for_display(chat_id: int) -> str:
    """Форматировать резюме чата для отображения пользователю."""
    meta = get_chat_meta(chat_id)
    problems = get_problems_by_chat(chat_id)

    parts = []

    if meta.get("overview"):
        parts.append(f"OBZOR\n{meta['overview']}")

    if problems:
        parts.append("\nPROBLEMY")
        for i, p in enumerate(problems):
            status_icon = "+" if p.status == "solved" else "-"
            parts.append(f"{i}. [{status_icon}] {p.title}")
            if p.short_summary:
                parts.append(f"   {p.short_summary}")

    if meta.get("decisions"):
        parts.append("\nRESHENIYA")
        for d in meta["decisions"]:
            parts.append(f"* {d}")

    if meta.get("key_points"):
        parts.append("\nKLYUCHEVYE MOMENTY")
        for k in meta["key_points"]:
            parts.append(f"* {k}")

    return (
        "\n".join(parts)
        if parts
        else "Rezume poka pustoe. Napishite soobscheniya i ispolzuyte /summarize"
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

        # Форматируем сообщения с контекстом reply
        formatted_messages = format_messages_with_context(chunk, chat_id)

        # Форматируем проблемы с последними сообщениями
        problems_context = format_problems_context(existing_problems, chat_id)

        prompt = ANALYZE_MESSAGES_PROMPT.format(
            problems_context=problems_context, messages=formatted_messages
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

    formatted = "\n".join(format_message_for_llm(m) for m in messages)
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
        problems_text.append(f"* {p.title} [{status}]\n  {p.short_summary}")

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
    return current_summary


async def summarize_thread(messages: list[dict]) -> str:
    """Legacy функция."""
    return "Use /summarize command instead"
