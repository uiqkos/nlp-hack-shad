import logging

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
    get_messages_count,
    get_messages_for_problem,
    get_problem_by_id,
    get_problems_by_chat,
    get_unprocessed_messages,
    save_message,
    update_problem_status,
)
from summarizer import (
    analyze_and_update,
    answer_query,
    format_summary_for_display,
    regenerate_problem_summary,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
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
    """Получить тег автора (@username или ссылку)."""
    if not user:
        return ""
    if user.username:
        return f"@{user.username}"
    # Если нет username, делаем ссылку на профиль
    return f"tg://user?id={user.id}"


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


def format_author_with_link(name: str, tag: str) -> str:
    """Форматировать имя автора со ссылкой в скобках."""
    if not tag:
        return name
    return f"{name} ({tag})"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /start."""
    user = update.effective_user
    logger.info(f"/start from user {user.id} ({user.first_name})")
    await update.message.reply_text(
        "Привет! Я бот для суммаризации чатов.\n\n"
        "Я сохраняю все сообщения и создаю структурированное резюме с проблемами.\n\n"
        "Команды:\n"
        "/summarize — обработать новые сообщения и показать резюме\n"
        "/problems — показать список проблем\n"
        "/problem <номер> — подробности о проблеме\n"
        "/messages <номер> — ссылки на сообщения проблемы\n"
        "/solve <номер> — отметить проблему решённой\n"
        "/query <вопрос> — задать вопрос по резюме\n"
        "/stats — статистика чата\n"
        "/clear — очистить всё"
    )


async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сохранять все сообщения в БД."""
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id

    # Определяем автора: если пересланное — берём оригинального автора
    author_name = "Unknown"
    author_tag = ""

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
        elif isinstance(origin, MessageOriginHiddenUser):
            # Скрытый пользователь
            author_name = origin.sender_user_name
            author_tag = ""
        elif isinstance(origin, MessageOriginChat):
            # Переслано от имени чата/группы
            author_name = origin.sender_chat.title or "Chat"
            if origin.sender_chat.username:
                author_tag = f"@{origin.sender_chat.username}"
        elif isinstance(origin, MessageOriginChannel):
            # Переслано из канала
            author_name = origin.chat.title or "Channel"
            if origin.chat.username:
                author_tag = f"@{origin.chat.username}"
    else:
        # Обычное сообщение
        user = message.from_user
        author_name = get_author_name(user)
        author_tag = get_author_tag(user)

    # Создаём объект Message
    msg = Message(
        id=None,
        chat_id=chat_id,
        telegram_msg_id=message.message_id,
        text=message.text,
        author_tag=author_tag,
        author_name=author_name,
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

    text = "📋 ПРОБЛЕМЫ:\n\n"
    for i, p in enumerate(problems):
        status_icon = "✅" if p.status == "solved" else "❌"
        text += f"{i}. {status_icon} {p.title}\n"
        if p.short_summary:
            text += (
                f"   {p.short_summary[:100]}...\n"
                if len(p.short_summary) > 100
                else f"   {p.short_summary}\n"
            )
        text += "\n"

    text += "Используйте /problem <номер> для подробностей"
    await send_long_message(message, text)


async def problem_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /problem <номер> — детали проблемы."""
    message = update.message
    chat_id = message.chat_id

    if not context.args:
        await message.reply_text("Использование: /problem <номер>")
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
    status_text = "✅ Решено" if p.status == "solved" else "❌ Не решено"

    text = f"🔧 ПРОБЛЕМА #{idx}\n\n"
    text += f"📌 {p.title}\n\n"
    text += f"Статус: {status_text}\n\n"

    if p.short_summary:
        text += f"Кратко: {p.short_summary}\n\n"

    if p.long_summary:
        text += f"Подробно:\n{p.long_summary}\n\n"

    # Количество связанных сообщений
    msgs = get_messages_for_problem(p.id)
    text += f"Связанных сообщений: {len(msgs)}\n"
    text += f"Используйте /messages {idx} для просмотра ссылок"

    await send_long_message(message, text)


async def messages_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /messages <номер> — ссылки на сообщения проблемы."""
    message = update.message
    chat_id = message.chat_id

    if not context.args:
        await message.reply_text("Использование: /messages <номер_проблемы>")
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
        author = format_author_with_link(m.author_name or "Unknown", m.author_tag)
        preview = m.text[:50] + "..." if len(m.text) > 50 else m.text
        link = m.telegram_link or build_telegram_link(chat_id, m.telegram_msg_id)
        text += f"• {author}: {preview}\n  {link}\n\n"

    if len(msgs) > 30:
        text += f"... и ещё {len(msgs) - 30} сообщений"

    await send_long_message(message, text)


async def solve_problem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /solve <номер> — отметить проблему решённой."""
    message = update.message
    chat_id = message.chat_id

    if not context.args:
        await message.reply_text("Использование: /solve <номер_проблемы>")
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

    if p.status == "solved":
        # Если уже решена — снимаем отметку
        update_problem_status(p.id, "unsolved")
        await message.reply_text(f"❌ Проблема #{idx} отмечена как нерешённая")
    else:
        update_problem_status(p.id, "solved")
        await message.reply_text(f"✅ Проблема #{idx} отмечена как решённая!")


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /query — вопрос по резюме."""
    message = update.message
    chat_id = message.chat_id

    question = " ".join(context.args) if context.args else ""
    if not question:
        await message.reply_text("Использование: /query <ваш вопрос>")
        return

    logger.info(f"/query: {question}")

    await message.reply_text("Ищу ответ...")

    try:
        answer = await answer_query(chat_id, question)
        await message.reply_text(answer)
    except Exception as e:
        logger.error(f"Error in query: {e}", exc_info=True)
        await message.reply_text(f"Ошибка: {str(e)}")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /stats — статистика чата."""
    message = update.message
    chat_id = message.chat_id

    total_messages = get_messages_count(chat_id)
    unprocessed = len(get_unprocessed_messages(chat_id))
    problems = get_problems_by_chat(chat_id)

    solved = sum(1 for p in problems if p.status == "solved")
    unsolved = len(problems) - solved

    text = "📊 СТАТИСТИКА ЧАТА\n\n"
    text += f"Всего сообщений: {total_messages}\n"
    text += f"Необработанных: {unprocessed}\n\n"
    text += f"Всего проблем: {len(problems)}\n"
    text += f"  ✅ Решено: {solved}\n"
    text += f"  ❌ Не решено: {unsolved}"

    await message.reply_text(text)


async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /clear — очистить все данные."""
    message = update.message
    chat_id = message.chat_id
    logger.info(f"/clear in chat {chat_id}")

    clear_chat_data(chat_id)
    await message.reply_text("Все данные чата очищены.")


async def send_long_message(message, text: str, max_length: int = 4096) -> None:
    """Отправить длинное сообщение, разбив на части."""
    if len(text) <= max_length:
        await message.reply_text(text)
    else:
        for i in range(0, len(text), max_length):
            await message.reply_text(text[i : i + max_length])


def main() -> None:
    """Запуск бота."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("summarize", summarize))
    application.add_handler(CommandHandler("problems", problems_list))
    application.add_handler(CommandHandler("problem", problem_detail))
    application.add_handler(CommandHandler("messages", messages_cmd))
    application.add_handler(CommandHandler("solve", solve_problem))
    application.add_handler(CommandHandler("query", query))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("clear", clear_chat))

    # Сбор сообщений — должен быть после команд
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, collect_message)
    )

    logger.info("Bot started")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
