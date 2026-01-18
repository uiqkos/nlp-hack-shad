import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4.1-mini")

# Iterative summarization settings
CHUNK_SIZE = 20  # messages per chunk for iterative processing
MAX_MESSAGES = 200  # max messages to fetch from thread
