import asyncio
import logging
import os
import shutil
import uuid
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

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
MAX_ACTIVE_USERS = 32
MAX_PERSISTED_CANDIDATES = 32
BOOKS_DIR = Path(os.getenv("BOOKS_DIR", "books"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "embeddings"))

# Each user gets a separate in-memory knowledge base and a separate cache directory.
# The uploaded source file is removed after processing, while the cache is retained
# only for the current active book.
user_books: dict[int, BookKnowledgeBase] = {}
_user_storage_dirs: dict[int, Path] = {}
_user_locks: dict[int, asyncio.Lock] = {}
_active_user_order: OrderedDict[int, None] = OrderedDict()


def _validate_user_id(user_id: object) -> int:
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise ValueError("invalid Telegram user id")  # noqa: TRY004 - caller reports bad update
    return user_id


def _user_lock(user_id: int) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


def _touch_active_user(user_id: int) -> None:
    _active_user_order.pop(user_id, None)
    _active_user_order[user_id] = None


def _evict_active_user(exempt_user_id: int) -> None:
    while len(user_books) >= MAX_ACTIVE_USERS:
        candidate = next(
            (
                user_id
                for user_id in _active_user_order
                if user_id != exempt_user_id and user_id in user_books
            ),
            None,
        )
        if candidate is None:
            candidate = next(
                (user_id for user_id in user_books if user_id != exempt_user_id),
                None,
            )
        if candidate is None:
            return

        user_books.pop(candidate, None)
        _active_user_order.pop(candidate, None)
        previous_storage = _user_storage_dirs.pop(candidate, None)
        if previous_storage is not None:
            _remove_directory(previous_storage)


def _restore_latest_book(user_id: int) -> BookKnowledgeBase | None:
    user_storage_root = EMBEDDINGS_DIR / str(user_id)
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
            _user_storage_dirs[user_id] = storage_dir
            return knowledge_base
    return None


def get_kb(user_id: int) -> BookKnowledgeBase:
    user_id = _validate_user_id(user_id)
    if user_id not in user_books:
        _evict_active_user(user_id)
        user_books[user_id] = _restore_latest_book(user_id) or BookKnowledgeBase(
            storage_dir=EMBEDDINGS_DIR / str(user_id)
        )
    _touch_active_user(user_id)
    return user_books[user_id]


def _new_book_paths(user_id: int, filename: str) -> tuple[Path, Path]:
    user_id = _validate_user_id(user_id)
    filename = normalize_book_filename(filename)
    upload_id = uuid.uuid4().hex
    book_dir = BOOKS_DIR / str(user_id)
    embedding_dir = EMBEDDINGS_DIR / str(user_id) / upload_id
    return book_dir / f"{upload_id}_{filename}", embedding_dir


def _validate_document_size(document: object) -> bool:
    declared_size = getattr(document, "file_size", None)
    if declared_size is None:
        return True
    if not isinstance(declared_size, int) or isinstance(declared_size, bool):
        return False
    return 0 <= declared_size <= MAX_BOOK_SIZE_BYTES


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


async def _safe_edit(message: object, text: str) -> None:
    try:
        await message.edit_text(text)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - Telegram status updates are best effort
        _log_failure("status update", exc)


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
    user_id = update.effective_user.id
    text = get_kb(user_id).get_book_summary()
    await update.message.reply_text(text)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    user_id = _validate_user_id(update.effective_user.id)

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

    async with _user_lock(user_id):
        msg = await update.message.reply_text("Downloading...")
        file_path, embedding_dir = _new_book_paths(user_id, filename)
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
                await _safe_edit(
                    msg, "That book is too large. The maximum supported size is 20 MB."
                )
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
                await _safe_edit(
                    msg,
                    "I could not process that book. Please upload a text-based PDF or TXT file.",
                )
                return

            if user_id not in user_books:
                _evict_active_user(user_id)
            previous_storage = _user_storage_dirs.get(user_id)
            user_books[user_id] = kb_ref
            _user_storage_dirs[user_id] = embedding_dir
            _touch_active_user(user_id)
            if previous_storage and previous_storage != embedding_dir:
                _remove_directory(previous_storage)

            await _safe_edit(
                msg,
                f"Book loaded: {kb_ref.book_name}\n"
                f"Chunks: {len(kb_ref.documents)}\n\n"
                "Ready to answer questions!",
            )
        except Exception as exc:  # noqa: BLE001 - upload failures are user-safe
            _log_failure("book upload", exc)
            await _safe_edit(
                msg, "I could not download or process that book. Please try again."
            )
        finally:
            if processing_task is None:
                _remove_file(file_path)
                _remove_directory(embedding_dir)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    question = update.message.text or ""
    user_id = _validate_user_id(update.effective_user.id)

    if len(question) > MAX_QUESTION_LENGTH:
        await update.message.reply_text(
            "Please keep questions to 1,000 characters or fewer."
        )
        return

    kb_ref = get_kb(user_id)
    if kb_ref is None or not kb_ref.documents:
        await update.message.reply_text("No book loaded. Use /load_book")
        return
    _touch_active_user(user_id)

    msg = await update.message.reply_text("Searching...")

    try:
        answer, _chunks = await asyncio.to_thread(
            kb_ref.answer_question, question, 2, True
        )
        response = f"Q: {question}\n\nA: {answer}"
        await _safe_edit(msg, response)
    except Exception as exc:  # noqa: BLE001 - model failures are user-safe
        _log_failure("question answering", exc)
        await _safe_edit(
            msg, "I could not answer that question right now. Please try again."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    del update
    if context.error is not None:
        _log_failure("unhandled bot error", context.error)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. Set it in .env before starting the bot."
        )

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("load_book", load_book))
    application.add_handler(CommandHandler("summary", summary))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)
    )
    application.add_error_handler(error_handler)

    print("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
