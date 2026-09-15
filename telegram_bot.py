import asyncio
import logging
import os
import shutil
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from time import monotonic

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from book_qa import BookKnowledgeBase
from file_utils import normalize_book_filename

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

MAX_BOOK_SIZE_BYTES = 20 * 1024 * 1024
MAX_QUESTION_LENGTH = 1_000
MAX_SOURCE_EXCERPT_CHARACTERS = 600
MAX_TELEGRAM_MESSAGE_CHARACTERS = 4_000
MAX_ACTIVE_USERS = 32
MAX_PERSISTED_CANDIDATES = 32
MAX_TRACKED_SESSIONS = 1_024
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_UPLOADS_PER_WINDOW = 5
MAX_QUESTIONS_PER_WINDOW = 30
MAX_CONCURRENT_UPDATES = 16
BOOKS_DIR = Path(os.getenv("BOOKS_DIR", "books"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "embeddings"))

SessionKey = tuple[int, int]

# Each chat/user session gets a separate in-memory knowledge base and cache.
# The uploaded source file is removed after processing, while the cache is retained
# only for the current active book.
user_books: dict[SessionKey, BookKnowledgeBase] = {}
_user_storage_dirs: dict[SessionKey, Path] = {}
_user_locks: dict[SessionKey, asyncio.Lock] = {}
_user_lock_refs: dict[SessionKey, int] = {}
_active_user_order: OrderedDict[SessionKey, None] = OrderedDict()
_request_history: dict[SessionKey, dict[str, deque[float]]] = {}


def _validate_user_id(user_id: object) -> int:
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise ValueError("invalid Telegram user id")  # noqa: TRY004 - caller reports bad update
    return user_id


def _validate_chat_id(chat_id: object) -> int:
    if not isinstance(chat_id, int) or isinstance(chat_id, bool):
        raise ValueError("invalid Telegram chat id")  # noqa: TRY004 - caller reports bad update
    return chat_id


def _session_key(user_id: object, chat_id: object) -> SessionKey:
    return (_validate_chat_id(chat_id), _validate_user_id(user_id))


def _session_key_for_update(update: Update) -> SessionKey:
    user = getattr(update, "effective_user", None)
    user_id = _validate_user_id(getattr(user, "id", None))
    chat = getattr(update, "effective_chat", None)
    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        message = getattr(update, "message", None)
        chat_id = getattr(getattr(message, "chat", None), "id", None)
    # Real Telegram messages always carry a chat. The private-chat fallback
    # preserves compatibility with small local update fakes and old caches.
    return _session_key(user_id, user_id if chat_id is None else chat_id)


def _user_lock(session_key: SessionKey) -> asyncio.Lock:
    lock = _user_locks.get(session_key)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[session_key] = lock
    return lock


@asynccontextmanager
async def _hold_user_lock(session_key: SessionKey):
    """Acquire a session's lock and pin it against eviction.

    A lock must never be evicted while a task holds a reference to it: with
    concurrent updates enabled, a queued task that survives eviction would
    serialize on an orphaned lock while new tasks get a fresh one, letting
    two handlers for the same session run their upload critical section
    concurrently. The reference count tells eviction which locks are safe
    to discard.
    """
    lock = _user_lock(session_key)
    _user_lock_refs[session_key] = _user_lock_refs.get(session_key, 0) + 1
    try:
        async with lock:
            yield lock
    finally:
        remaining = _user_lock_refs.get(session_key, 0) - 1
        if remaining > 0:
            _user_lock_refs[session_key] = remaining
        else:
            _user_lock_refs.pop(session_key, None)


def _touch_active_user(session_key: SessionKey) -> None:
    _active_user_order.pop(session_key, None)
    _active_user_order[session_key] = None


def _evict_active_user(exempt_session_key: SessionKey) -> None:
    while len(user_books) >= MAX_ACTIVE_USERS:
        candidate = next(
            (
                session_key
                for session_key in _active_user_order
                if session_key != exempt_session_key and session_key in user_books
                # A session whose lock is pinned by an in-flight handler must
                # survive: evicting it would orphan that handler's lock.
                and not _user_lock_refs.get(session_key)
            ),
            None,
        )
        if candidate is None:
            candidate = next(
                (
                    session_key
                    for session_key in user_books
                    if session_key != exempt_session_key
                    and not _user_lock_refs.get(session_key)
                ),
                None,
            )
        if candidate is None:
            return

        user_books.pop(candidate, None)
        _active_user_order.pop(candidate, None)
        previous_storage = _user_storage_dirs.pop(candidate, None)
        if previous_storage is not None:
            _remove_directory(previous_storage)
        _request_history.pop(candidate, None)
        _user_locks.pop(candidate, None)


def _session_storage_root(base_dir: Path, session_key: SessionKey) -> Path:
    chat_id, user_id = session_key
    if chat_id == user_id:
        # Private-chat caches from the previous user-scoped format remain
        # restorable. Group chats always use a distinct namespace.
        return base_dir / str(user_id)
    return base_dir / f"chat-{chat_id}" / str(user_id)


def _restore_latest_book(session_key: SessionKey) -> BookKnowledgeBase | None:
    user_storage_root = _session_storage_root(EMBEDDINGS_DIR, session_key)
    try:
        candidates = [
            path
            for path in user_storage_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        ]
    except OSError:
        return None

    def modification_time(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return -1

    candidates.sort(key=modification_time, reverse=True)
    for storage_dir in candidates[:MAX_PERSISTED_CANDIDATES]:
        knowledge_base = BookKnowledgeBase(storage_dir=storage_dir)
        if knowledge_base.load_embeddings():
            _user_storage_dirs[session_key] = storage_dir
            return knowledge_base
    return None


def _get_kb(session_key: SessionKey) -> BookKnowledgeBase:
    if session_key not in user_books:
        _evict_active_user(session_key)
        user_books[session_key] = _restore_latest_book(
            session_key
        ) or BookKnowledgeBase(
            storage_dir=_session_storage_root(EMBEDDINGS_DIR, session_key)
        )
    _touch_active_user(session_key)
    return user_books[session_key]


def get_kb(user_id: int, chat_id: int | None = None) -> BookKnowledgeBase:
    user_id = _validate_user_id(user_id)
    session_key = _session_key(user_id, user_id if chat_id is None else chat_id)
    return _get_kb(session_key)


def _new_book_paths(
    user_id: int, filename: str, chat_id: int | None = None
) -> tuple[Path, Path]:
    session_key = _session_key(user_id, user_id if chat_id is None else chat_id)
    filename = normalize_book_filename(filename)
    upload_id = uuid.uuid4().hex
    book_dir = _session_storage_root(BOOKS_DIR, session_key)
    embedding_dir = _session_storage_root(EMBEDDINGS_DIR, session_key) / upload_id
    return book_dir / f"{upload_id}_{filename}", embedding_dir


def _validate_document_size(document: object) -> bool:
    declared_size = getattr(document, "file_size", None)
    if declared_size is None:
        return True
    if not isinstance(declared_size, int) or isinstance(declared_size, bool):
        return False
    return 0 <= declared_size <= MAX_BOOK_SIZE_BYTES


def _allow_request(session_key: SessionKey, bucket: str, limit: int) -> bool:
    now = monotonic()
    session_history = _request_history.get(session_key)
    if session_history is None:
        if len(_request_history) >= MAX_TRACKED_SESSIONS:
            oldest_session = next(iter(_request_history), None)
            if oldest_session is not None:
                _request_history.pop(oldest_session, None)
        session_history = {}
        _request_history[session_key] = session_history
    history = session_history.setdefault(bucket, deque())
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while history and history[0] <= cutoff:
        history.popleft()
    if len(history) >= limit:
        return False
    history.append(now)
    return True


def _log_failure(operation: str, error: Exception) -> None:
    # Provider and filesystem exceptions can contain paths, URLs, or credentials.
    logger.error("%s failed: %s", operation, type(error).__name__)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _log_failure("temporary file cleanup", exc)


def _remove_directory(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        _log_failure("temporary directory cleanup", exc)


def _process_uploaded_book(
    file_path: Path, embedding_dir: Path, book_name: str
) -> BookKnowledgeBase | None:
    success = False
    try:
        knowledge_base = BookKnowledgeBase(storage_dir=embedding_dir)
        success = knowledge_base.load_book(file_path, book_name)
        return knowledge_base if success else None
    finally:
        # Keep cleanup in the worker so cancelling the async handler cannot
        # remove files while extraction or embedding is still in progress.
        _remove_file(file_path)
        if not success:
            _remove_directory(embedding_dir)


def _discard_cancelled_result(
    task: asyncio.Future[object], embedding_dir: Path
) -> None:
    if task.cancelled():
        _remove_directory(embedding_dir)
        return
    with suppress(Exception):
        if task.result() is not None:
            _remove_directory(embedding_dir)


async def _wait_for_processing(
    task: asyncio.Task[BookKnowledgeBase | None], embedding_dir: Path
) -> BookKnowledgeBase | None:
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Let the worker finish its filesystem cleanup before propagating
        # cancellation. A second cancellation is handled by the callback.
        task.add_done_callback(
            lambda completed: _discard_cancelled_result(completed, embedding_dir)
        )
        processed_book: BookKnowledgeBase | None = None
        with suppress(asyncio.CancelledError, Exception):
            processed_book = await asyncio.shield(task)
        if processed_book is not None:
            _remove_directory(embedding_dir)
        raise


def _bounded_message(text: object) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= MAX_TELEGRAM_MESSAGE_CHARACTERS:
        return text
    return text[: MAX_TELEGRAM_MESSAGE_CHARACTERS - 3].rstrip() + "..."


async def _safe_edit(message: object, text: str) -> bool:
    try:
        await message.edit_text(_bounded_message(text))  # type: ignore[attr-defined]
        return True
    except Exception as exc:  # noqa: BLE001 - Telegram status updates are best effort
        _log_failure("status update", exc)
        return False


async def _safe_reply(message: object, text: str) -> bool:
    try:
        await message.reply_text(_bounded_message(text))  # type: ignore[attr-defined]
        return True
    except Exception as exc:  # noqa: BLE001 - Telegram replies are best effort
        _log_failure("reply", exc)
        return False


def _format_source_excerpts(chunks: list[str]) -> str:
    """Format bounded, numbered retrieval excerpts for user verification."""
    excerpts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, str):
            continue
        excerpt = " ".join(chunk.split())
        if len(excerpt) > MAX_SOURCE_EXCERPT_CHARACTERS:
            excerpt = excerpt[: MAX_SOURCE_EXCERPT_CHARACTERS - 3].rstrip() + "..."
        if excerpt:
            excerpts.append(f"[{index}] {excerpt}")
    return "\n".join(excerpts) or "No source excerpt available"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    msg = (
        "Welcome to Book Q&A Bot\n\n"
        "Send a book file then ask questions about it.\n\n"
        "Commands: /load_book /summary /help"
    )
    await update.message.reply_text(msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    msg = (
        "Send a book file (PDF/TXT)\n"
        "Wait for processing\n"
        "Ask questions\n\n"
        "Commands: /start /load_book /summary"
    )
    await update.message.reply_text(msg)


async def load_book(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    await update.message.reply_text("Send a book file (PDF or TXT)")


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    session_key = _session_key_for_update(update)
    text = _get_kb(session_key).get_book_summary()
    await update.message.reply_text(text)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    session_key = _session_key_for_update(update)
    chat_id, user_id = session_key

    if not _allow_request(session_key, "upload", MAX_UPLOADS_PER_WINDOW):
        await update.message.reply_text(
            "Too many uploads. Please wait a little before trying again."
        )
        return

    try:
        filename = normalize_book_filename(document.file_name)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    if not _validate_document_size(document):
        await update.message.reply_text(
            "That book is too large. The maximum supported size is 20 MB."
        )
        return

    async with _hold_user_lock(session_key):
        msg = await update.message.reply_text("Downloading...")
        file_path, embedding_dir = _new_book_paths(user_id, filename, chat_id)
        processing_task: asyncio.Task[BookKnowledgeBase | None] | None = None

        try:
            _ensure_private_directory(file_path.parent)
            _ensure_private_directory(embedding_dir)
            file = await context.bot.get_file(document.file_id)
            await file.download_to_drive(file_path)

            if (
                not file_path.is_file()
                or file_path.stat().st_size > MAX_BOOK_SIZE_BYTES
            ):
                response = (
                    "That book is too large. The maximum supported size is 20 MB."
                )
                if not await _safe_edit(msg, response):
                    await _safe_reply(update.message, response)
                return
            file_path.chmod(0o600)

            await _safe_edit(msg, "Processing book...")
            processing_task = asyncio.create_task(
                asyncio.to_thread(
                    _process_uploaded_book,
                    file_path,
                    embedding_dir,
                    Path(filename).stem,
                )
            )
            kb_ref = await _wait_for_processing(processing_task, embedding_dir)

            if kb_ref is None:
                response = "I could not process that book. Please upload a text-based PDF or TXT file."
                if not await _safe_edit(msg, response):
                    await _safe_reply(update.message, response)
                return

            if session_key not in user_books:
                _evict_active_user(session_key)
            previous_storage = _user_storage_dirs.get(session_key)
            user_books[session_key] = kb_ref
            _user_storage_dirs[session_key] = embedding_dir
            _touch_active_user(session_key)
            if previous_storage and previous_storage != embedding_dir:
                _remove_directory(previous_storage)

            response = (
                f"Book loaded: {kb_ref.book_name}\n"
                f"Chunks: {len(kb_ref.documents)}\n\n"
                "Ready to answer questions!"
            )
            if not await _safe_edit(msg, response):
                await _safe_reply(update.message, response)
        except Exception as exc:  # noqa: BLE001 - upload failures are user-safe
            _log_failure("book upload", exc)
            response = "I could not download or process that book. Please try again."
            if not await _safe_edit(msg, response):
                await _safe_reply(update.message, response)
        finally:
            if processing_task is None:
                _remove_file(file_path)
                _remove_directory(embedding_dir)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    question = update.message.text or ""
    session_key = _session_key_for_update(update)

    if not _allow_request(session_key, "question", MAX_QUESTIONS_PER_WINDOW):
        await update.message.reply_text(
            "Too many questions. Please wait a little before trying again."
        )
        return

    if len(question) > MAX_QUESTION_LENGTH:
        await update.message.reply_text(
            "Please keep questions to 1,000 characters or fewer."
        )
        return

    kb_ref = _get_kb(session_key)
    if kb_ref is None or not kb_ref.documents:
        await update.message.reply_text("No book loaded. Use /load_book")
        return
    _touch_active_user(session_key)

    msg = await update.message.reply_text("Searching...")

    try:
        answer, chunks = await asyncio.to_thread(
            kb_ref.answer_question, question, 2, True
        )
        response = _bounded_message(
            f"Q: {question}\n\n"
            f"A: [1] {answer}\n\n"
            "Sources (retrieved excerpts):\n"
            f"{_format_source_excerpts(chunks)}"
        )
        if not await _safe_edit(msg, response):
            await _safe_reply(update.message, response)
    except Exception as exc:  # noqa: BLE001 - model failures are user-safe
        _log_failure("question answering", exc)
        response = "I could not answer that question right now. Please try again."
        if not await _safe_edit(msg, response):
            await _safe_reply(update.message, response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    del update
    if context.error is not None:
        _log_failure("unhandled bot error", context.error)


def _build_application(token: str) -> Application:
    return (
        Application.builder()
        .token(token)
        .concurrent_updates(MAX_CONCURRENT_UPDATES)
        .build()
    )


def _install_handlers(application: Application) -> None:
    message_only = filters.UpdateType.MESSAGE
    application.add_handler(
        CommandHandler("start", start, filters=message_only)
    )
    application.add_handler(
        CommandHandler("help", help_command, filters=message_only)
    )
    application.add_handler(
        CommandHandler("load_book", load_book, filters=message_only)
    )
    application.add_handler(
        CommandHandler("summary", summary, filters=message_only)
    )
    application.add_handler(
        MessageHandler(filters.Document.ALL & message_only, handle_document)
    )
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & message_only, handle_question
        )
    )
    application.add_error_handler(error_handler)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. Set it in .env before starting the bot."
        )

    application = _build_application(token)
    _install_handlers(application)

    print("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
