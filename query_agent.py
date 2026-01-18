"""
Query Agent - LLM агент для поиска информации по проблемам в чате.

Агент получает вопрос пользователя и использует tools для:
1. Просмотра списка проблем (summary)
2. Получения деталей конкретных проблем
3. Получения сообщений, связанных с проблемой (пагинация)
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable

import httpx

from config import OPENROUTER_API_KEY
from database import (
    Message,
    Problem,
    get_chat_meta,
    get_messages_for_problem,
    get_problems_by_chat,
)

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Максимальное количество итераций агента
MAX_ITERATIONS = 10
# Количество ретраев при ошибке API
MAX_RETRIES = 3
# Сообщений на страницу по умолчанию
DEFAULT_PAGE_SIZE = 10


@dataclass
class AgentState:
    """Состояние агента для отображения пользователю."""

    status: str  # Текущий статус
    details: str | None = None  # Дополнительные детали


# Определение tools для агента
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_problems_details",
            "description": "Получить подробную информацию о конкретных проблемах по их индексам. "
            "Используй когда нужно узнать детали проблем (описание, статус, решение).",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Список индексов проблем (0, 1, 2, ...) для получения деталей",
                    }
                },
                "required": ["problem_indices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_problem_messages",
            "description": "Получить сообщения, связанные с проблемой. "
            "Используй когда описания проблемы недостаточно и нужно посмотреть исходные сообщения.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_index": {
                        "type": "integer",
                        "description": "Индекс проблемы (0, 1, 2, ...)",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Номер страницы (начиная с 1)",
                        "default": 1,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Количество сообщений на странице",
                        "default": 10,
                    },
                },
                "required": ["problem_index"],
            },
        },
    },
]

AGENT_MODEL = "openai/gpt-4.1"

SYSTEM_PROMPT = """Ты — поисковый агент. Твоя задача — найти ответ на вопрос ТОЛЬКО на основе данных из чата.

СТРОГИЕ ПРАВИЛА:
1. НИКОГДА не придумывай информацию. Отвечай ТОЛЬКО тем, что нашёл в сообщениях.
2. Если информации нет в сообщениях — честно скажи "Информация не найдена в чате".
3. НЕ используй свои знания. Только данные из чата.
4. Цитируй или пересказывай найденное, не додумывай.
5. Отвечай простым текстом без markdown (без *, **, `, # и т.д.)
6. В конце ответа ОБЯЗАТЕЛЬНО добавь ссылки на проблемы, из которых взята информация, в формате: /problem_0, /problem_3

Tools:
- get_problems_details: подробности о проблемах по индексам
- get_problem_messages: сообщения проблемы (с пагинацией)

Алгоритм:
1. Изучи список проблем
2. Выбери релевантные проблемы
3. Запроси их детали через get_problems_details
4. ОБЯЗАТЕЛЬНО запроси сообщения через get_problem_messages
5. Найди в сообщениях конкретный ответ
6. Ответь ТОЛЬКО на основе найденного
7. Добавь в конце строку "Источники: /problem_X, /problem_Y" со ссылками на использованные проблемы

Если в сообщениях нет ответа — так и скажи. Не выдумывай."""


def format_problems_list(problems: list[Problem]) -> str:
    """Форматировать краткий список проблем для агента."""
    if not problems:
        return "Проблем пока нет."

    lines = []
    for i, p in enumerate(problems):
        status_map = {
            "solved": "решено",
            "partial": "есть информация",
            "unsolved": "не решено",
        }
        status = status_map.get(p.status, p.status)
        lines.append(f"[{i}] {p.title} [{status}]")
        if p.short_summary:
            lines.append(f"    {p.short_summary}")
    return "\n".join(lines)


def format_problem_details(problems: list[Problem], indices: list[int]) -> str:
    """Форматировать подробную информацию о проблемах."""
    results = []
    for idx in indices:
        if idx < 0 or idx >= len(problems):
            results.append(f"[{idx}] Проблема не найдена")
            continue

        p = problems[idx]
        status_map = {
            "solved": "решено",
            "partial": "есть информация",
            "unsolved": "не решено",
        }
        status = status_map.get(p.status, p.status)

        parts = [
            f"[{idx}] {p.title}",
            f"Статус: {status}",
        ]
        if p.long_summary:
            parts.append(f"Описание: {p.long_summary}")
        if p.solution:
            parts.append(f"Решение: {p.solution}")

        results.append("\n".join(parts))

    return "\n\n".join(results)


def format_messages_page(messages: list[Message], page: int, page_size: int) -> str:
    """Форматировать страницу сообщений."""
    total = len(messages)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    if page < 1:
        page = 1
    if page > total_pages:
        return f"Страница {page} не существует. Всего страниц: {total_pages}"

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_messages = messages[start:end]

    lines = [f"Сообщения (страница {page}/{total_pages}, всего {total}):"]
    for m in page_messages:
        author = m.author_name or m.author_tag or "Unknown"
        text = m.text[:500] + "..." if len(m.text) > 500 else m.text
        lines.append(f"- {author}: {text}")

    return "\n".join(lines)


async def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Вызов LLM API с поддержкой tools и ретраями."""
    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Debug logging
                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                logger.debug("=" * 60)
                logger.debug("AGENT LLM RESPONSE")
                logger.debug("=" * 60)
                if msg.get("content"):
                    logger.debug(f"[CONTENT]\n{msg['content']}")
                if msg.get("tool_calls"):
                    logger.debug(f"[TOOL CALLS]")
                    for tc in msg["tool_calls"]:
                        logger.debug(
                            f"  - {tc['function']['name']}: {tc['function']['arguments']}"
                        )
                logger.debug("=" * 60)

                return data
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"LLM API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"LLM API failed after {MAX_RETRIES} attempts: {e}")
                raise

    raise last_error


async def run_query_agent(
    chat_id: int,
    question: str,
    on_status: Callable[[AgentState], None] | None = None,
) -> str:
    """
    Запустить агента для ответа на вопрос.

    Args:
        chat_id: ID чата
        question: Вопрос пользователя
        on_status: Async callback для обновления статуса (для UI)

    Returns:
        Ответ агента
    """

    async def update_status(status: str, details: str | None = None):
        if on_status:
            await on_status(AgentState(status=status, details=details))

    # Загружаем данные
    await update_status("Загружаю данные...")
    problems = get_problems_by_chat(chat_id)
    meta = get_chat_meta(chat_id)

    if not problems:
        return "В чате пока нет проблем. Используйте /summarize для анализа сообщений."

    # Формируем начальный контекст
    problems_list = format_problems_list(problems)
    overview = meta.get("overview", "")

    initial_context = f"""Вопрос пользователя: {question}

Обзор чата: {overview if overview else "Нет общего описания"}

Список проблем:
{problems_list}

Изучи список и определи, какие проблемы могут содержать ответ.
Используй tools чтобы получить детали или сообщения."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_context},
    ]

    await update_status("Анализирую проблемы...")

    # Цикл агента
    for iteration in range(MAX_ITERATIONS):
        try:
            response = await call_llm_with_tools(messages, TOOLS)
        except Exception as e:
            logger.error(f"Agent LLM call failed: {e}")
            return f"Ошибка при обращении к LLM: {str(e)}"

        choice = response["choices"][0]
        assistant_message = choice["message"]
        messages.append(assistant_message)

        # Проверяем, есть ли tool calls
        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            # Агент закончил работу и дал ответ
            await update_status("Готово")
            return assistant_message.get("content", "Не удалось найти ответ.")

        # Обрабатываем tool calls
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            result = ""

            if func_name == "get_problems_details":
                indices = args.get("problem_indices", [])
                if indices:
                    indices_str = ", ".join(str(i) for i in indices)
                    await update_status("Просматриваю проблемы", indices_str)
                result = format_problem_details(problems, indices)

            elif func_name == "get_problem_messages":
                idx = args.get("problem_index", 0)
                page = args.get("page", 1)
                page_size = args.get("page_size", DEFAULT_PAGE_SIZE)

                if idx < 0 or idx >= len(problems):
                    result = f"Проблема {idx} не найдена"
                else:
                    await update_status(
                        f"Читаю сообщения проблемы {idx}", f"страница {page}"
                    )
                    problem_messages = get_messages_for_problem(problems[idx].id)
                    result = format_messages_page(problem_messages, page, page_size)

            else:
                result = f"Неизвестный tool: {func_name}"

            # Добавляем результат в messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                }
            )

    # Достигли лимита итераций
    await update_status("Готово")
    return "Не удалось найти ответ за отведённое количество шагов. Попробуйте переформулировать вопрос."
