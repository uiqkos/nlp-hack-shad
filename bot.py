import logging
import os
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import TELEGRAM_BOT_TOKEN
from database import (
    Message,
    clear_chat_data,
    get_message_by_telegram_id,
    get_messages_count,
    get_messages_for_problem,
    get_problem_by_id,
    get_problems_by_chat,
    get_unprocessed_messages,
    save_message,
    update_problem_status,
)
from llm_client import analyze_image
from query_agent import AgentState, run_query_agent
from summarizer import (
    analyze_and_update,
    format_summary_for_display,
    regenerate_problem_summary,
)

# Set log level from env: DEBUG for verbose LLM logging, INFO for normal
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger(__name__)


def build_telegram_link(chat_id: int, message_id: int) -> str:
    """Построить ссылку на сообщение в Telegram."""
    # Для супергрупп chat_id начинается с -100
    chat_id_str = str(chat_id)
    if chat_id_str.startswith("-100"):
        chat_id_for_link = chat_id_str[4:]  # Убираем -100
    else:
        chat_id_for_link = chat_id_str.lstrip("-")
    return f"https://t.me/c/{chat_id_for_link}/{message_id}"


def get_author_tag(user) -> str:
    """Получить тег автора (username без @)."""
    if not user:
        return ""
    if user.username:
        return user.username
    return ""


def get_author_name(user) -> str:
    """Получить отображаемое имя пользователя."""
    if not user:
        return "Unknown"
    parts = []
    if user.first_name:
        parts.append(user.first_name)
    if user.last_name:
        parts.append(user.last_name)
    return " ".join(parts) if parts else "Unknown"


def build_user_link(user) -> str:
    """Построить ссылку на профиль пользователя."""
    if not user:
        return ""
    if user.username:
        return f"https://t.me/{user.username}"
    return f"tg://user?id={user.id}"


def format_author_display(name: str, tag: str) -> str:
    """Форматировать имя автора с тегом в скобках."""
    if tag:
        return f"{name} ({tag})"
    return name


HELP_TEXT = """Я бот для суммаризации чатов.

Сохраняю все сообщения (включая картинки) и создаю структурированное резюме с проблемами.

📋 Основные команды:
/summarize — обработать новые сообщения и показать резюме
/problems — показать список проблем
/stats — статистика чата

🔍 Работа с проблемами:
/problem_N — подробности о проблеме (например /problem_0)
/messages_N — ссылки на сообщения проблемы
/solve_N — переключить статус (❌→🔶→✅→❌)

❓ Прочее:
/query <вопрос> — задать вопрос по резюме
/clear — очистить все данные чата
/help — эта справка

Статусы проблем:
✅ Решено — есть конкретный ответ
🔶 Есть информация — полезные данные, но решённость под вопросом
❌ Не решено — нет ответа"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /start."""
    user = update.effective_user
    logger.info(f"/start from user {user.id} ({user.first_name})")
    await update.message.reply_text(f"Привет!\n\n{HELP_TEXT}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /help."""
    await update.message.reply_text(HELP_TEXT)


async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сохранять все сообщения (включая картинки) в БД."""
    message = update.message
    if not message:
        return

    # Игнорируем сообщения от самого бота
    if message.from_user and message.from_user.id == context.bot.id:
        return

    chat_id = message.chat_id
    text = message.text or ""
    caption = message.caption or ""

    # Если сообщение уже сохранено — не анализируем заново
    existing = get_message_by_telegram_id(chat_id, message.message_id)
    if existing and existing.text:
        return

    image_blocks: list[str] = []
    prompt = (
        "Верни строго два блока:\n"
        "<IMAGE_DESC>краткое описание изображения</IMAGE_DESC>\n"
        "<IMAGE_TEXT>извлечённый текст с изображения или пусто</IMAGE_TEXT>\n"
        "Без дополнительных пояснений."
    )
    if caption:
        prompt += f"\n\nПодпись пользователя: {caption}"

    # Обрабатываем фото (одно сообщение может содержать несколько изображений)
    if message.photo:
        # Группируем по file_unique_id, чтобы обрабатывать каждое изображение один раз
        photos_by_id = {}
        for photo in message.photo:
            existing_photo = photos_by_id.get(photo.file_unique_id)
            if not existing_photo or (photo.file_size or 0) > (
                existing_photo.file_size or 0
            ):
                photos_by_id[photo.file_unique_id] = photo

        for photo in photos_by_id.values():
            try:
                file = await photo.get_file()
                image_bytes = await file.download_as_bytearray()
                image_description = await analyze_image(image_bytes, prompt)
                image_blocks.append(f"<IMAGE>\n{image_description}\n</IMAGE>")
            except Exception as e:
                logger.error(f"Image analysis failed: {e}", exc_info=True)
                image_blocks.append(
                    "<IMAGE>\n<IMAGE_DESC>Не удалось проанализировать</IMAGE_DESC>\n"
                    "<IMAGE_TEXT></IMAGE_TEXT>\n</IMAGE>"
                )

    if image_blocks:
        image_list = "<IMAGE_LIST>\n" + "\n".join(image_blocks) + "\n</IMAGE_LIST>"
        text = (
            f"{image_list}\n\n{caption or text}".strip()
            if (caption or text)
            else image_list
        )

    if not text:
        return

    # Определяем автора: если пересланное — берём оригинального автора
    author_name = "Unknown"
    author_tag = ""
    author_link = ""

    if message.forward_origin:
        # Пересланное сообщение — берём оригинального автора
        from telegram import (
            MessageOriginChannel,
            MessageOriginChat,
            MessageOriginHiddenUser,
            MessageOriginUser,
        )

        origin = message.forward_origin
        if isinstance(origin, MessageOriginUser):
            # Переслано от пользователя
            author_name = get_author_name(origin.sender_user)
            author_tag = get_author_tag(origin.sender_user)
            author_link = build_user_link(origin.sender_user)
        elif isinstance(origin, MessageOriginHiddenUser):
            # Скрытый пользователь
            author_name = origin.sender_user_name
            author_tag = ""
            author_link = ""
        elif isinstance(origin, MessageOriginChat):
            # Переслано от имени чата/группы
            author_name = origin.sender_chat.title or "Chat"
            if origin.sender_chat.username:
                author_tag = origin.sender_chat.username
                author_link = f"https://t.me/{origin.sender_chat.username}"
        elif isinstance(origin, MessageOriginChannel):
            # Переслано из канала
            author_name = origin.chat.title or "Channel"
            if origin.chat.username:
                author_tag = origin.chat.username
                author_link = f"https://t.me/{origin.chat.username}"
    else:
        # Обычное сообщение
        user = message.from_user
        author_name = get_author_name(user)
        author_tag = get_author_tag(user)
        author_link = build_user_link(user)

    # Создаём объект Message
    msg = Message(
        id=None,
        chat_id=chat_id,
        telegram_msg_id=message.message_id,
        text=text,
        author_tag=author_tag,
        author_name=author_name,
        author_link=author_link,
        reply_to_msg_id=message.reply_to_message.message_id
        if message.reply_to_message
        else None,
        telegram_link=build_telegram_link(chat_id, message.message_id),
    )

    # Сохраняем в БД
    msg_id = save_message(msg)
    logger.info(f"Message saved: id={msg_id}, from {msg.author_name} in chat {chat_id}")


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /summarize — анализ новых сообщений."""
    message = update.message
    user = update.effective_user
    chat_id = message.chat_id
    logger.info(f"/summarize from {user.first_name} in chat {chat_id}")

    # Получаем необработанные сообщения
    new_messages = get_unprocessed_messages(chat_id)

    if not new_messages:
        # Показываем текущее резюме
        summary_text = format_summary_for_display(chat_id)
        await send_long_message(message, summary_text)
        return

    status_msg = await message.reply_text(
        f"Анализирую {len(new_messages)} новых сообщений..."
    )

    async def on_progress(current: int, total: int):
        if total > 1:
            try:
                await status_msg.edit_text(f"Обрабатываю батч {current}/{total}...")
            except Exception:
                pass

    try:
        stats = await analyze_and_update(chat_id, new_messages, on_progress)

        try:
            await status_msg.delete()
        except Exception:
            pass

        # Формируем отчёт
        report = []
        if stats["new_problems"]:
            report.append(f"Найдено новых проблем: {stats['new_problems']}")
        if stats["updated_problems"]:
            report.append(f"Обновлено проблем: {stats['updated_problems']}")

        if report:
            await message.reply_text("\n".join(report))

        # Показываем резюме
        summary_text = format_summary_for_display(chat_id)
        await send_long_message(message, summary_text)

    except Exception as e:
        logger.error(f"Error in summarize: {e}", exc_info=True)
        await message.reply_text(f"Ошибка при анализе: {str(e)}")


async def problems_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /problems — список всех проблем."""
    message = update.message
    chat_id = message.chat_id

    problems = get_problems_by_chat(chat_id)

    if not problems:
        await message.reply_text(
            "Пока нет проблем. Напишите сообщения и используйте /summarize"
        )
        return

    solved_count = sum(1 for p in problems if p.status == "solved")
    partial_count = sum(1 for p in problems if p.status == "partial")
    unsolved_count = len(problems) - solved_count - partial_count
    text = (
        f"📋 ПРОБЛЕМЫ ({solved_count}✅ / {partial_count}🔶 / {unsolved_count}❌)\n\n"
    )

    for i, p in enumerate(problems):
        if p.status == "solved":
            status_icon = "✅"
        elif p.status == "partial":
            status_icon = "🔶"
        else:
            status_icon = "❌"
        text += f"/problem_{i} {status_icon} {p.title}\n"
        if p.status in ("solved", "partial") and p.solution:
            text += f"   💡 Решение: {p.solution}\n"
        elif p.long_summary:
            text += (
                f"   {p.long_summary[:150]}...\n"
                if len(p.long_summary) > 150
                else f"   {p.long_summary}\n"
            )
        text += "\n"

    await send_long_message(message, text)


async def problem_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /problem <номер> или /problem_N — детали проблемы."""
    message = update.message
    chat_id = message.chat_id

    # Проверяем динамическую команду /problem_N
    idx = None
    if message.text:
        match = re.match(r"/problem_(\d+)", message.text)
        if match:
            idx = int(match.group(1))

    # Если не динамическая команда, проверяем аргументы
    if idx is None:
        if not context.args:
            await message.reply_text("Использование: /problem <номер> или /problem_N")
            return
        try:
            idx = int(context.args[0])
        except ValueError:
            await message.reply_text("Укажите номер проблемы (число)")
            return

    problems = get_problems_by_chat(chat_id)

    if idx >= len(problems) or idx < 0:
        await message.reply_text(
            f"Проблема {idx} не найдена. Всего проблем: {len(problems)}"
        )
        return

    p = problems[idx]
    if p.status == "solved":
        status_icon = "✅"
        status_text = "Решено"
    elif p.status == "partial":
        status_icon = "🔶"
        status_text = "Есть информация"
    else:
        status_icon = "❌"
        status_text = "Не решено"

    text = f"🔧 ПРОБЛЕМА #{idx} {status_icon}\n\n"
    text += f"📌 {p.title}\n\n"
    text += f"Статус: {status_text}\n\n"

    if p.solution:
        text += f"💡 РЕШЕНИЕ:\n{p.solution}\n\n"

    if p.long_summary:
        text += f"Описание:\n{p.long_summary}\n\n"

    # Количество связанных сообщений
    msgs = get_messages_for_problem(p.id)
    text += f"📨 Сообщений: {len(msgs)}\n\n"
    text += f"Действия:\n"
    text += f"/messages_{idx} — показать сообщения\n"
    text += f"/solve_{idx} — переключить статус"

    await send_long_message(message, text)


async def messages_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /messages <номер> или /messages_N — ссылки на сообщения проблемы."""
    message = update.message
    chat_id = message.chat_id

    # Проверяем динамическую команду /messages_N
    idx = None
    if message.text:
        match = re.match(r"/messages_(\d+)", message.text)
        if match:
            idx = int(match.group(1))

    # Если не динамическая команда, проверяем аргументы
    if idx is None:
        if not context.args:
            await message.reply_text("Использование: /messages <номер> или /messages_N")
            return
        try:
            idx = int(context.args[0])
        except ValueError:
            await message.reply_text("Укажите номер проблемы (число)")
            return

    problems = get_problems_by_chat(chat_id)

    if idx >= len(problems) or idx < 0:
        await message.reply_text(f"Проблема {idx} не найдена")
        return

    p = problems[idx]
    msgs = get_messages_for_problem(p.id)

    if not msgs:
        await message.reply_text(f"Нет сообщений для проблемы {idx}")
        return

    text = f"📨 Сообщения для проблемы #{idx}:\n{p.title}\n\n"

    for m in msgs[:30]:  # Лимит 30 ссылок
        author = format_author_display(m.author_name or "Unknown", m.author_tag)
        preview = m.text[:150] + "..." if len(m.text) > 150 else m.text
        msg_link = m.telegram_link or build_telegram_link(chat_id, m.telegram_msg_id)
        text += f"• {author}: {preview}\n"
        if m.author_link:
            text += f"  Профиль: {m.author_link}\n"
        text += f"  Сообщение: {msg_link}\n\n"

    if len(msgs) > 30:
        text += f"... и ещё {len(msgs) - 30} сообщений"

    await send_long_message(message, text)


async def solve_problem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /solve <номер> или /solve_N — переключить статус проблемы."""
    message = update.message
    chat_id = message.chat_id

    # Проверяем динамическую команду /solve_N
    idx = None
    if message.text:
        match = re.match(r"/solve_(\d+)", message.text)
        if match:
            idx = int(match.group(1))

    # Если не динамическая команда, проверяем аргументы
    if idx is None:
        if not context.args:
            await message.reply_text("Использование: /solve <номер> или /solve_N")
            return
        try:
            idx = int(context.args[0])
        except ValueError:
            await message.reply_text("Укажите номер проблемы (число)")
            return

    problems = get_problems_by_chat(chat_id)

    if idx >= len(problems) or idx < 0:
        await message.reply_text(f"Проблема {idx} не найдена")
        return

    p = problems[idx]

    # Циклическое переключение: unsolved -> partial -> solved -> unsolved
    if p.status == "unsolved":
        update_problem_status(p.id, "partial")
        await message.reply_text(
            f"🔶 Проблема #{idx} отмечена как 'есть информация'\n/problem_{idx}"
        )
    elif p.status == "partial":
        update_problem_status(p.id, "solved")
        await message.reply_text(
            f"✅ Проблема #{idx} отмечена как решённая!\n/problem_{idx}"
        )
    else:  # solved
        update_problem_status(p.id, "unsolved")
        await message.reply_text(
            f"❌ Проблема #{idx} отмечена как нерешённая\n/problem_{idx}"
        )


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /query — вопрос по резюме с использованием агента."""
    message = update.message
    chat_id = message.chat_id

    question = " ".join(context.args) if context.args else ""
    if not question:
        await message.reply_text("Использование: /query <ваш вопрос>")
        return

    logger.info(f"/query: {question}")

    # Создаём сообщение со статусом
    status_msg = await message.reply_text("Ищу ответ...")
    last_status_text = "Ищу ответ..."

    async def on_status(state: AgentState):
        """Callback для обновления статуса в сообщении."""
        nonlocal last_status_text
        if state.details:
            new_text = f"{state.status}: {state.details}"
        else:
            new_text = state.status

        # Обновляем только если текст изменился
        if new_text != last_status_text:
            last_status_text = new_text
            try:
                await status_msg.edit_text(new_text)
            except Exception:
                pass  # Игнорируем ошибки редактирования

    try:
        answer = await run_query_agent(chat_id, question, on_status)

        # Удаляем сообщение со статусом и отправляем ответ
        try:
            await status_msg.delete()
        except Exception:
            pass

        await send_long_message(message, answer)
    except Exception as e:
        logger.error(f"Error in query: {e}", exc_info=True)
        try:
            await status_msg.edit_text(f"Ошибка: {str(e)}")
        except Exception:
            await message.reply_text(f"Ошибка: {str(e)}")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /stats — статистика чата."""
    message = update.message
    chat_id = message.chat_id

    total_messages = get_messages_count(chat_id)
    unprocessed = len(get_unprocessed_messages(chat_id))
    problems = get_problems_by_chat(chat_id)

    solved = sum(1 for p in problems if p.status == "solved")
    partial = sum(1 for p in problems if p.status == "partial")
    unsolved = len(problems) - solved - partial

    text = "📊 СТАТИСТИКА ЧАТА\n\n"
    text += f"Всего сообщений: {total_messages}\n"
    text += f"Необработанных: {unprocessed}\n\n"
    text += f"Всего проблем: {len(problems)}\n"
    text += f"  ✅ Решено: {solved}\n"
    text += f"  🔶 Есть информация: {partial}\n"
    text += f"  ❌ Не решено: {unsolved}"

    await message.reply_text(text)


async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /clear — очистить все данные."""
    message = update.message
    chat_id = message.chat_id
    logger.info(f"/clear in chat {chat_id}")

    clear_chat_data(chat_id)
    await message.reply_text("Все данные чата очищены.")


async def send_long_message(
    message, text: str, max_length: int = 4096, parse_mode: str = None
) -> None:
    """Отправить длинное сообщение, разбив на части."""
    from telegram import LinkPreviewOptions
    from telegram.constants import ParseMode

    link_preview = LinkPreviewOptions(is_disabled=True)

    async def send_chunk(chunk: str):
        """Отправить один кусок текста с fallback на plain text."""
        try:
            await message.reply_text(
                chunk,
                link_preview_options=link_preview,
                parse_mode=parse_mode,
            )
        except Exception:
            # Если Markdown не парсится — отправляем как plain text
            await message.reply_text(chunk, link_preview_options=link_preview)

    if len(text) <= max_length:
        await send_chunk(text)
    else:
        for i in range(0, len(text), max_length):
            await send_chunk(text[i : i + max_length])


def main() -> None:
    """Запуск бота."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("summarize", summarize))
    application.add_handler(CommandHandler("problems", problems_list))
    application.add_handler(CommandHandler("problem", problem_detail))
    application.add_handler(CommandHandler("messages", messages_cmd))
    application.add_handler(CommandHandler("solve", solve_problem))
    application.add_handler(CommandHandler("query", query))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("clear", clear_chat))

    # Динамические команды /problem_N, /messages_N, /solve_N
    application.add_handler(
        MessageHandler(filters.Regex(r"^/problem_\d+"), problem_detail)
    )
    application.add_handler(
        MessageHandler(filters.Regex(r"^/messages_\d+"), messages_cmd)
    )
    application.add_handler(
        MessageHandler(filters.Regex(r"^/solve_\d+"), solve_problem)
    )

    # Сбор сообщений — должен быть после команд
    application.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO) & ~filters.COMMAND, collect_message
        )
    )

    logger.info("Bot started")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
