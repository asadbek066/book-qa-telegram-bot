import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import telegram_bot


class FakeStatus:
    def __init__(self, messages):
        self.messages = messages

    async def edit_text(self, text):
        self.messages.append(text)


class FakeMessage:
    def __init__(self, messages, document=None, text=None):
        self.messages = messages
        self.document = document
        self.text = text

    async def reply_text(self, text):
        self.messages.append(text)
        return FakeStatus(self.messages)


class FakeFile:
    def __init__(self, content):
        self.content = content

    async def download_to_drive(self, path):
        Path(path).write_bytes(self.content)


class FakeBot:
    def __init__(self, content):
        self.content = content

    async def get_file(self, file_id):
        del file_id
        return FakeFile(self.content)


class FakeKnowledgeBase:
    def __init__(self, storage_dir=None):
        self.storage_dir = Path(storage_dir)
        self.documents = []
        self.embeddings = None
        self.book_name = None

    def load_book(self, path, book_name=None):
        del path
        self.book_name = book_name
        self.documents = ["private document"]
        self.embeddings = object()
        return True

    def answer_question(self, question, top_k, short):
        del top_k, short
        return f"answer for {question}", ["private document"]


class BookDocument:
    def __init__(self, filename="book.txt", size=10):
        self.file_name = filename
        self.file_id = "file-id"
        self.file_size = size


class TelegramBotBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.messages = []
        self.patches = [
            patch.object(telegram_bot, "BookKnowledgeBase", FakeKnowledgeBase),
            patch.object(telegram_bot, "BOOKS_DIR", root / "books"),
            patch.object(telegram_bot, "EMBEDDINGS_DIR", root / "embeddings"),
        ]
        for active_patch in self.patches:
            active_patch.start()
        telegram_bot.user_books.clear()
        telegram_bot._user_storage_dirs.clear()
        telegram_bot._user_locks.clear()

    def tearDown(self):
        telegram_bot.user_books.clear()
        telegram_bot._user_storage_dirs.clear()
        telegram_bot._user_locks.clear()
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def update_for(user_id, document=None, text=None):
        message = FakeMessage([], document=document, text=text)
        return SimpleNamespace(
            message=message, effective_user=SimpleNamespace(id=user_id)
        )

    def test_same_filename_is_isolated_between_users(self):
        async def run():
            for user_id in (101, 202):
                update = self.update_for(user_id, BookDocument("same.txt"))
                update.message.messages = self.messages
                context = SimpleNamespace(bot=FakeBot(b"book text"))
                await telegram_bot.handle_document(update, context)

        asyncio.run(run())

        self.assertEqual(set(telegram_bot.user_books), {101, 202})
        self.assertIsNot(telegram_bot.user_books[101], telegram_bot.user_books[202])
        self.assertNotEqual(
            telegram_bot._user_storage_dirs[101], telegram_bot._user_storage_dirs[202]
        )
        self.assertTrue(
            any(message.startswith("Book loaded: same") for message in self.messages)
        )

    def test_same_user_uploads_are_serialized(self):
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        class SerialKnowledgeBase(FakeKnowledgeBase):
            def load_book(self, path, book_name=None):
                nonlocal active, maximum_active
                del path
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.05)
                with state_lock:
                    active -= 1
                self.book_name = book_name
                self.documents = ["private document"]
                self.embeddings = object()
                return True

        async def run():
            with patch.object(telegram_bot, "BookKnowledgeBase", SerialKnowledgeBase):
                context = SimpleNamespace(bot=FakeBot(b"book text"))
                await asyncio.gather(
                    telegram_bot.handle_document(
                        self.update_for(101, BookDocument("first.txt")), context
                    ),
                    telegram_bot.handle_document(
                        self.update_for(101, BookDocument("second.txt")), context
                    ),
                )

        asyncio.run(run())

        self.assertEqual(maximum_active, 1)
        self.assertEqual(set(telegram_bot.user_books), {101})
        self.assertTrue(telegram_bot._user_storage_dirs[101].is_dir())

    def test_active_book_memory_is_bounded_without_cross_user_access(self):
        async def run():
            with patch.object(telegram_bot, "MAX_ACTIVE_USERS", 1):
                await telegram_bot.handle_document(
                    self.update_for(101, BookDocument("first.txt")),
                    SimpleNamespace(bot=FakeBot(b"book text")),
                )
                await telegram_bot.handle_document(
                    self.update_for(202, BookDocument("second.txt")),
                    SimpleNamespace(bot=FakeBot(b"book text")),
                )

        asyncio.run(run())

        self.assertEqual(set(telegram_bot.user_books), {202})
        update = self.update_for(101, text="What was in my book?")
        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))
        self.assertEqual(update.message.messages, ["No book loaded. Use /load_book"])

    def test_latest_valid_persisted_cache_is_restored_for_the_user(self):
        class RestoringKnowledgeBase(FakeKnowledgeBase):
            def load_embeddings(self):
                self.book_name = "restored"
                self.documents = ["restored document"]
                self.embeddings = object()
                return True

            def get_book_summary(self):
                return f"Book: {self.book_name}\nChunks: {len(self.documents)}"

        storage_dir = Path(self.temp_dir.name) / "embeddings" / "303" / "upload-id"
        storage_dir.mkdir(parents=True)

        with patch.object(telegram_bot, "BookKnowledgeBase", RestoringKnowledgeBase):
            knowledge_base = telegram_bot.get_kb(303)

        self.assertEqual(knowledge_base.get_book_summary(), "Book: restored\nChunks: 1")
        self.assertEqual(telegram_bot._user_storage_dirs[303], storage_dir)

    def test_download_failure_is_generic_and_cleans_temporary_paths(self):
        class FailingBot:
            async def get_file(self, file_id):
                del file_id
                raise RuntimeError("token=private-secret")

        update = self.update_for(101, BookDocument("book.txt"))

        asyncio.run(
            telegram_bot.handle_document(update, SimpleNamespace(bot=FailingBot()))
        )

        self.assertEqual(update.message.messages[0], "Downloading...")
        self.assertIn("try again", update.message.messages[-1])
        self.assertNotIn("private-secret", " ".join(update.message.messages))
        self.assertFalse(
            any(
                path.is_file()
                for path in (Path(self.temp_dir.name) / "books").rglob("*")
            )
        )
        self.assertFalse(
            any(
                path.is_file()
                for path in (Path(self.temp_dir.name) / "embeddings").rglob("*")
            )
        )

    def test_handler_cancellation_waits_for_processing_cleanup(self):
        started = threading.Event()
        finished = threading.Event()

        class SlowKnowledgeBase(FakeKnowledgeBase):
            def load_book(self, path, book_name=None):
                del path, book_name
                started.set()
                time.sleep(0.05)
                finished.set()
                self.documents = ["private document"]
                self.embeddings = object()
                return True

        async def run():
            with patch.object(telegram_bot, "BookKnowledgeBase", SlowKnowledgeBase):
                task = asyncio.create_task(
                    telegram_bot.handle_document(
                        self.update_for(101, BookDocument("book.txt")),
                        SimpleNamespace(bot=FakeBot(b"book text")),
                    )
                )
                await asyncio.to_thread(started.wait)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(run())

        self.assertTrue(finished.is_set())
        self.assertEqual(telegram_bot.user_books, {})
        self.assertFalse(
            any(
                path.is_file()
                for path in (Path(self.temp_dir.name) / "books").rglob("*")
            )
        )
        self.assertFalse(
            any(
                path.is_file()
                for path in (Path(self.temp_dir.name) / "embeddings").rglob("*")
            )
        )

    def test_oversized_document_is_rejected_before_download(self):
        update = self.update_for(
            101, BookDocument("book.txt", telegram_bot.MAX_BOOK_SIZE_BYTES + 1)
        )

        asyncio.run(
            telegram_bot.handle_document(update, SimpleNamespace(bot=FakeBot(b"")))
        )

        self.assertEqual(len(update.message.messages), 1)
        self.assertIn("too large", update.message.messages[0])
        self.assertEqual(telegram_bot.user_books, {})

    def test_question_uses_only_requesting_users_book(self):
        telegram_bot.user_books[101] = FakeKnowledgeBase("one")
        telegram_bot.user_books[101].documents = ["user one"]
        update = self.update_for(101, text="What is private?")
        context = SimpleNamespace()

        asyncio.run(telegram_bot.handle_question(update, context))

        self.assertTrue(
            any(
                "answer for What is private?" in message
                for message in update.message.messages
            )
        )

    def test_question_without_a_book_does_not_use_another_user_book(self):
        telegram_bot.user_books[101] = FakeKnowledgeBase("one")
        telegram_bot.user_books[101].documents = ["user one"]
        update = self.update_for(202, text="What is private?")

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        self.assertEqual(update.message.messages, ["No book loaded. Use /load_book"])

    def test_question_length_is_bounded(self):
        telegram_bot.user_books[101] = FakeKnowledgeBase("one")
        telegram_bot.user_books[101].documents = ["user one"]
        update = self.update_for(101, text="x" * (telegram_bot.MAX_QUESTION_LENGTH + 1))

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        self.assertIn("1,000 characters", update.message.messages[0])


if __name__ == "__main__":
    unittest.main()
