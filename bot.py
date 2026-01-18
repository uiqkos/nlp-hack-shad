import logging
from collections import defaultdict

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import MAX_MESSAGES, TELEGRAM_BOT_TOKEN
from database import clear_chat_data, get_chat_data, save_chat_data
from summarizer import answer_query, format_summary_for_display, update_summary

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# In-memory buffer for new messages (before they're processed into summary)
# Structure: {chat_id: [messages]}
message_buffer: dict[int, list[dict]] = defaultdict(list)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    logger.info(f"/start from user {user.id} ({user.first_name})")
    await update.message.reply_text(
        "Привет! Я бот для суммаризации чатов.\n\n"
        "Я накапливаю сообщения и создаю структурированное резюме.\n\n"
        "Команды:\n"
        "/summarize — обновить и показать резюме\n"
        "/query <вопрос> — задать вопрос по резюме\n"
        "/messages — показать ссылки на сообщения по проблеме\n"
        "/clear — очистить историю"
    )


async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Collect messages from any chat."""
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    author = message.from_user.first_name if message.from_user else "Unknown"

    msg_data = {
        "author": author,
        "text": message.text,
        "message_id": message.message_id,
    }
    message_buffer[chat_id].append(msg_data)
    logger.info(
        f"Message from {author} in chat {chat_id}. Buffer: {len(message_buffer[chat_id])}"
    )

    # Limit buffer size
    if len(message_buffer[chat_id]) > MAX_MESSAGES:
        message_buffer[chat_id] = message_buffer[chat_id][-MAX_MESSAGES:]


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /summarize command — update and show summary."""
    message = update.message
    user = update.effective_user
    chat_id = message.chat_id
    logger.info(f"/summarize from {user.first_name} in chat {chat_id}")

    # Get current data from DB
    chat_data = get_chat_data(chat_id)
    current_summary = chat_data["summary"]

    # Get new messages from buffer
    new_messages = message_buffer.get(chat_id, [])

    if not new_messages and not current_summary.get("overview"):
        await message.reply_text(
            "Нет сообщений для суммаризации.\n"
            "Напишите несколько сообщений и попробуйте снова."
        )
        return

    if new_messages:
        status_msg = await message.reply_text(
            f"Обрабатываю {len(new_messages)} новых сообщений..."
        )

        async def on_progress(current: int, total: int):
            if total > 1:
                try:
                    await status_msg.edit_text(f"Обрабатываю батч {current}/{total}...")
                except Exception:
                    pass

        try:
            # Update summary with new messages in batches
            updated_summary = await update_summary(
                current_summary, new_messages, on_progress=on_progress
            )

            # Save to DB
            last_msg_id = (
                new_messages[-1]["message_id"]
                if new_messages
                else chat_data["last_message_id"]
            )
            save_chat_data(chat_id, updated_summary, last_msg_id)

            # Clear buffer
            message_buffer[chat_id] = []

            logger.info(f"Summary updated for chat {chat_id}")
            current_summary = updated_summary

            try:
                await status_msg.delete()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error updating summary: {e}", exc_info=True)
            await message.reply_text(f"Ошибка при обновлении резюме: {str(e)}")
            return

    # Show current summary
    summary_text = format_summary_for_display(current_summary)

    # Split long messages (Telegram limit is 4096)
    if len(summary_text) <= 4096:
        await message.reply_text(summary_text)
    else:
        for i in range(0, len(summary_text), 4096):
            await message.reply_text(summary_text[i : i + 4096])


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /query command — answer question based on summary."""
    message = update.message
    user = update.effective_user
    chat_id = message.chat_id

    # Get question from command args
    question = " ".join(context.args) if context.args else ""
    if not question:
        await message.reply_text("Использование: /query <ваш вопрос>")
        return

    logger.info(f"/query from {user.first_name}: {question}")

    # Get summary from DB
    chat_data = get_chat_data(chat_id)
    current_summary = chat_data["summary"]

    if not current_summary.get("overview"):
        await message.reply_text("Резюме пока пустое. Сначала используйте /summarize")
        return

    await message.reply_text("Ищу ответ...")

    try:
        answer = await answer_query(current_summary, question)
        await message.reply_text(answer)
    except Exception as e:
        logger.error(f"Error answering query: {e}", exc_info=True)
        await message.reply_text(f"Ошибка: {str(e)}")


async def messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /messages command — show message links for a problem."""
    message = update.message
    user = update.effective_user
    chat_id = message.chat_id

    # Get problem index from args
    if not context.args:
        # Show list of problems with indices
        chat_data = get_chat_data(chat_id)
        problems = chat_data["summary"].get("problems", [])
        if not problems:
            await message.reply_text("Нет проблем в резюме.")
            return

        text = "Проблемы:\n"
        for i, p in enumerate(problems):
            status = "solved" if p.get("status") == "solved" else "unsolved"
            text += f"{i}: {p['problem'][:50]}... [{status}]\n"
        text += "\nИспользуйте: /messages <номер>"
        await message.reply_text(text)
        return

    try:
        problem_idx = int(context.args[0])
    except ValueError:
        await message.reply_text("Использование: /messages <номер_проблемы>")
        return

    logger.info(f"/messages {problem_idx} from {user.first_name}")

    chat_data = get_chat_data(chat_id)
    summary = chat_data["summary"]
    problems = summary.get("problems", [])
    message_labels = summary.get("message_labels", {})

    if problem_idx >= len(problems):
        await message.reply_text(
            f"Проблема {problem_idx} не найдена. Всего проблем: {len(problems)}"
        )
        return

    # Find messages for this problem
    msg_ids = [msg_id for msg_id, idx in message_labels.items() if idx == problem_idx]

    if not msg_ids:
        await message.reply_text(f"Нет сообщений для проблемы {problem_idx}")
        return

    # Build links (works for supergroups)
    # Format: t.me/c/CHAT_ID/MSG_ID (without -100 prefix)
    chat_id_for_link = str(chat_id).replace("-100", "")

    problem_text = problems[problem_idx]["problem"]
    text = f"Проблема {problem_idx}: {problem_text}\n\nСообщения:\n"
    for msg_id in msg_ids[:20]:  # Limit to 20 links
        text += f"• https://t.me/c/{chat_id_for_link}/{msg_id}\n"

    if len(msg_ids) > 20:
        text += f"\n...и ещё {len(msg_ids) - 20} сообщений"

    await message.reply_text(text)


async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command to reset chat history."""
    message = update.message
    user = update.effective_user
    chat_id = message.chat_id
    logger.info(f"/clear from {user.first_name} in chat {chat_id}")

    # Clear DB
    clear_chat_data(chat_id)
    # Clear buffer
    message_buffer[chat_id] = []

    await message.reply_text("История и резюме очищены.")


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("summarize", summarize))
    application.add_handler(CommandHandler("query", query))
    application.add_handler(CommandHandler("messages", messages))
    application.add_handler(CommandHandler("clear", clear_chat))

    # Message collector - must be after command handlers
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, collect_message)
    )

    logger.info("Bot started")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
