import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "chat_data.db"


@dataclass
class Message:
    id: int | None
    chat_id: int
    telegram_msg_id: int
    text: str
    author_tag: str  # username без @
    author_name: str  # Отображаемое имя
    author_link: str | None  # Ссылка на профиль автора
    reply_to_msg_id: int | None  # ID сообщения, на которое ответили
    telegram_link: str | None  # Ссылка на сообщение в Telegram


@dataclass
class Problem:
    id: int | None
    chat_id: int
    title: str
    short_summary: str
    long_summary: str
    solution: str  # Конкретное решение проблемы (если есть)
    status: str  # "solved" / "partial" / "unsolved"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()

    # Таблица сообщений
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            telegram_msg_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            author_tag TEXT,
            author_name TEXT,
            author_link TEXT,
            reply_to_msg_id INTEGER,
            telegram_link TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(chat_id, telegram_msg_id)
        )
    """)

    # Миграция: добавить author_link если его нет
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN author_link TEXT")
    except sqlite3.OperationalError:
        pass  # Колонка уже существует

    # Таблица проблем (создаём если нет)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            short_summary TEXT DEFAULT '',
            long_summary TEXT DEFAULT '',
            solution TEXT DEFAULT '',
            status TEXT DEFAULT 'unsolved',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Миграция: добавить solution если его нет
    try:
        conn.execute("ALTER TABLE problems ADD COLUMN solution TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Колонка уже существует

    # Таблица связей many-to-many между проблемами и сообщениями
    conn.execute("""
        CREATE TABLE IF NOT EXISTS problem_messages (
            problem_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            PRIMARY KEY (problem_id, message_id),
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
        )
    """)

    # Таблица метаданных чата (overview, decisions, key_points)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_meta (
            chat_id INTEGER PRIMARY KEY,
            overview TEXT DEFAULT '',
            decisions TEXT DEFAULT '[]',
            key_points TEXT DEFAULT '[]',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Индексы для быстрого поиска
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_problems_chat ON problems(chat_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_telegram ON messages(chat_id, telegram_msg_id)"
    )

    conn.commit()
    conn.close()


# ============== MESSAGE FUNCTIONS ==============


def save_message(msg: Message) -> int:
    """Сохранить сообщение в БД. Возвращает id."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO messages (chat_id, telegram_msg_id, text, author_tag, author_name, author_link, reply_to_msg_id, telegram_link)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_id, telegram_msg_id) DO UPDATE SET
            text = excluded.text,
            author_tag = excluded.author_tag,
            author_name = excluded.author_name,
            author_link = excluded.author_link,
            reply_to_msg_id = excluded.reply_to_msg_id,
            telegram_link = excluded.telegram_link
        RETURNING id
    """,
        (
            msg.chat_id,
            msg.telegram_msg_id,
            msg.text,
            msg.author_tag,
            msg.author_name,
            msg.author_link,
            msg.reply_to_msg_id,
            msg.telegram_link,
        ),
    )
    msg_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return msg_id


def get_message_by_telegram_id(chat_id: int, telegram_msg_id: int) -> Message | None:
    """Получить сообщение по telegram msg id."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM messages WHERE chat_id = ? AND telegram_msg_id = ?",
        (chat_id, telegram_msg_id),
    ).fetchone()
    conn.close()

    if row:
        return Message(
            id=row["id"],
            chat_id=row["chat_id"],
            telegram_msg_id=row["telegram_msg_id"],
            text=row["text"],
            author_tag=row["author_tag"],
            author_name=row["author_name"],
            author_link=row["author_link"],
            reply_to_msg_id=row["reply_to_msg_id"],
            telegram_link=row["telegram_link"],
        )
    return None


def get_messages_by_chat(
    chat_id: int, limit: int = None, offset: int = 0
) -> list[Message]:
    """Получить все сообщения чата."""
    conn = get_connection()
    query = "SELECT * FROM messages WHERE chat_id = ? ORDER BY telegram_msg_id ASC"
    params: list[Any] = [chat_id]

    if limit:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [
        Message(
            id=row["id"],
            chat_id=row["chat_id"],
            telegram_msg_id=row["telegram_msg_id"],
            text=row["text"],
            author_tag=row["author_tag"],
            author_name=row["author_name"],
            author_link=row["author_link"],
            reply_to_msg_id=row["reply_to_msg_id"],
            telegram_link=row["telegram_link"],
        )
        for row in rows
    ]


def get_unprocessed_messages(chat_id: int) -> list[Message]:
    """Получить сообщения, которые не привязаны ни к одной проблеме."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT m.* FROM messages m
        LEFT JOIN problem_messages pm ON m.id = pm.message_id
        WHERE m.chat_id = ? AND pm.message_id IS NULL
        ORDER BY m.telegram_msg_id ASC
    """,
        (chat_id,),
    ).fetchall()
    conn.close()

    return [
        Message(
            id=row["id"],
            chat_id=row["chat_id"],
            telegram_msg_id=row["telegram_msg_id"],
            text=row["text"],
            author_tag=row["author_tag"],
            author_name=row["author_name"],
            author_link=row["author_link"],
            reply_to_msg_id=row["reply_to_msg_id"],
            telegram_link=row["telegram_link"],
        )
        for row in rows
    ]


def get_messages_count(chat_id: int) -> int:
    """Получить количество сообщений в чате."""
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)
    ).fetchone()[0]
    conn.close()
    return count


# ============== PROBLEM FUNCTIONS ==============


def save_problem(problem: Problem) -> int:
    """Сохранить или обновить проблему. Возвращает id."""
    conn = get_connection()

    if problem.id:
        conn.execute(
            """
            UPDATE problems SET
                title = ?, short_summary = ?, long_summary = ?, solution = ?, status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (
                problem.title,
                problem.short_summary,
                problem.long_summary,
                problem.solution,
                problem.status,
                problem.id,
            ),
        )
        problem_id = problem.id
    else:
        cursor = conn.execute(
            """
            INSERT INTO problems (chat_id, title, short_summary, long_summary, solution, status)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
        """,
            (
                problem.chat_id,
                problem.title,
                problem.short_summary,
                problem.long_summary,
                problem.solution,
                problem.status,
            ),
        )
        problem_id = cursor.fetchone()[0]

    conn.commit()
    conn.close()
    return problem_id


def get_problem_by_id(problem_id: int) -> Problem | None:
    """Получить проблему по id."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM problems WHERE id = ?", (problem_id,)).fetchone()
    conn.close()

    if row:
        return Problem(
            id=row["id"],
            chat_id=row["chat_id"],
            title=row["title"],
            short_summary=row["short_summary"],
            long_summary=row["long_summary"],
            solution=row["solution"] or "",
            status=row["status"],
        )
    return None


def get_problems_by_chat(chat_id: int) -> list[Problem]:
    """Получить все проблемы чата."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM problems WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
    ).fetchall()
    conn.close()

    return [
        Problem(
            id=row["id"],
            chat_id=row["chat_id"],
            title=row["title"],
            short_summary=row["short_summary"],
            long_summary=row["long_summary"],
            solution=row["solution"] or "",
            status=row["status"],
        )
        for row in rows
    ]


def update_problem_status(problem_id: int, status: str):
    """Обновить статус проблемы."""
    conn = get_connection()
    conn.execute(
        "UPDATE problems SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, problem_id),
    )
    conn.commit()
    conn.close()


def delete_problem(problem_id: int):
    """Удалить проблему (связи удалятся каскадно)."""
    conn = get_connection()
    conn.execute("DELETE FROM problems WHERE id = ?", (problem_id,))
    conn.commit()
    conn.close()


# ============== PROBLEM-MESSAGE LINK FUNCTIONS ==============


def link_message_to_problem(message_id: int, problem_id: int):
    """Связать сообщение с проблемой."""
    conn = get_connection()
    conn.execute(
        """
        INSERT OR IGNORE INTO problem_messages (problem_id, message_id)
        VALUES (?, ?)
    """,
        (problem_id, message_id),
    )
    conn.commit()
    conn.close()


def link_messages_to_problem(message_ids: list[int], problem_id: int):
    """Связать несколько сообщений с проблемой."""
    conn = get_connection()
    conn.executemany(
        """
        INSERT OR IGNORE INTO problem_messages (problem_id, message_id)
        VALUES (?, ?)
    """,
        [(problem_id, msg_id) for msg_id in message_ids],
    )
    conn.commit()
    conn.close()


def get_messages_for_problem(problem_id: int) -> list[Message]:
    """Получить все сообщения, связанные с проблемой."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT m.* FROM messages m
        JOIN problem_messages pm ON m.id = pm.message_id
        WHERE pm.problem_id = ?
        ORDER BY m.telegram_msg_id ASC
    """,
        (problem_id,),
    ).fetchall()
    conn.close()

    return [
        Message(
            id=row["id"],
            chat_id=row["chat_id"],
            telegram_msg_id=row["telegram_msg_id"],
            text=row["text"],
            author_tag=row["author_tag"],
            author_name=row["author_name"],
            author_link=row["author_link"],
            reply_to_msg_id=row["reply_to_msg_id"],
            telegram_link=row["telegram_link"],
        )
        for row in rows
    ]


def get_problems_for_message(message_id: int) -> list[Problem]:
    """Получить все проблемы, к которым относится сообщение."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT p.* FROM problems p
        JOIN problem_messages pm ON p.id = pm.problem_id
        WHERE pm.message_id = ?
    """,
        (message_id,),
    ).fetchall()
    conn.close()

    return [
        Problem(
            id=row["id"],
            chat_id=row["chat_id"],
            title=row["title"],
            short_summary=row["short_summary"],
            long_summary=row["long_summary"],
            solution=row["solution"] or "",
            status=row["status"],
        )
        for row in rows
    ]


# ============== CHAT META FUNCTIONS ==============


def get_chat_meta(chat_id: int) -> dict[str, Any]:
    """Получить метаданные чата (overview, decisions, key_points)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM chat_meta WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    conn.close()

    if row:
        import json

        return {
            "overview": row["overview"],
            "decisions": json.loads(row["decisions"]),
            "key_points": json.loads(row["key_points"]),
        }
    return {"overview": "", "decisions": [], "key_points": []}


def save_chat_meta(
    chat_id: int, overview: str, decisions: list[str], key_points: list[str]
):
    """Сохранить метаданные чата."""
    import json

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO chat_meta (chat_id, overview, decisions, key_points, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chat_id) DO UPDATE SET
            overview = excluded.overview,
            decisions = excluded.decisions,
            key_points = excluded.key_points,
            updated_at = CURRENT_TIMESTAMP
    """,
        (
            chat_id,
            overview,
            json.dumps(decisions, ensure_ascii=False),
            json.dumps(key_points, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


# ============== CLEAR FUNCTIONS ==============


def clear_chat_data(chat_id: int):
    """Полностью очистить данные чата."""
    conn = get_connection()
    # Сначала удаляем связи (хотя CASCADE должен сработать)
    conn.execute(
        """
        DELETE FROM problem_messages WHERE problem_id IN
        (SELECT id FROM problems WHERE chat_id = ?)
    """,
        (chat_id,),
    )
    conn.execute("DELETE FROM problems WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chat_meta WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


# ============== LEGACY COMPATIBILITY ==============


def get_chat_data(chat_id: int) -> dict[str, Any]:
    """Legacy функция для совместимости. Возвращает данные в старом формате."""
    meta = get_chat_meta(chat_id)
    problems = get_problems_by_chat(chat_id)

    problems_list = []
    message_labels = {}

    for i, p in enumerate(problems):
        problems_list.append(
            {
                "problem": p.title,
                "solution": p.long_summary if p.status == "solved" else None,
                "status": p.status,
            }
        )
        # Получаем сообщения для этой проблемы
        msgs = get_messages_for_problem(p.id)
        for m in msgs:
            message_labels[str(m.telegram_msg_id)] = i

    return {
        "summary": {
            "overview": meta["overview"],
            "problems": problems_list,
            "decisions": meta["decisions"],
            "key_points": meta["key_points"],
            "message_labels": message_labels,
        },
        "last_message_id": 0,
    }


def save_chat_data(chat_id: int, summary: dict, last_message_id: int):
    """Legacy функция для совместимости. Это НЕ рекомендуется использовать напрямую."""
    # Сохраняем мета-данные
    save_chat_meta(
        chat_id,
        summary.get("overview", ""),
        summary.get("decisions", []),
        summary.get("key_points", []),
    )
    # Проблемы обрабатываются отдельно через новые функции


# Initialize DB on module import
init_db()
