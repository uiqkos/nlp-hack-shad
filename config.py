import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4.1-mini")

# Iterative summarization settings
CHUNK_SIZE = 40  # messages per chunk for iterative processing
MAX_MESSAGES = 200  # max messages to fetch from thread
CONTEXT_MESSAGES_PER_PROBLEM = 3  # last N messages to show per problem for context
