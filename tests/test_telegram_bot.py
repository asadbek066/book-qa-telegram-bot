import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import telegram_bot
from telegram import User
from telegram.ext import CommandHandler, MessageHandler


class FakeStatus:
    def __init__(self, messages):
        self.messages = messages

    async def edit_text(self, text):
        self.messages.append(text)


class FailingStatus(FakeStatus):
    async def edit_text(self, text):
        del text
        raise RuntimeError("status message disappeared")


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
        telegram_bot._active_user_order.clear()
        telegram_bot._request_history.clear()

    def tearDown(self):
        telegram_bot.user_books.clear()
        telegram_bot._user_storage_dirs.clear()
        telegram_bot._user_locks.clear()
        telegram_bot._active_user_order.clear()
        telegram_bot._request_history.clear()
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def update_for(user_id, document=None, text=None, chat_id=None):
        message = FakeMessage([], document=document, text=text)
        return SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(id=user_id),
            effective_chat=SimpleNamespace(id=user_id if chat_id is None else chat_id),
        )

    def test_same_filename_is_isolated_between_users(self):
        async def run():
            for user_id in (101, 202):
                update = self.update_for(user_id, BookDocument("same.txt"))
                update.message.messages = self.messages
                context = SimpleNamespace(bot=FakeBot(b"book text"))
                await telegram_bot.handle_document(update, context)

        asyncio.run(run())

        self.assertEqual(set(telegram_bot.user_books), {(101, 101), (202, 202)})
        self.assertIsNot(
            telegram_bot.user_books[(101, 101)], telegram_bot.user_books[(202, 202)]
        )
        self.assertNotEqual(
            telegram_bot._user_storage_dirs[(101, 101)],
            telegram_bot._user_storage_dirs[(202, 202)],
        )
        self.assertTrue(
            any(message.startswith("Book loaded: same") for message in self.messages)
        )

    def test_same_user_is_isolated_between_chat_sessions(self):
        async def run():
            context = SimpleNamespace(bot=FakeBot(b"book text"))
            await telegram_bot.handle_document(
                self.update_for(101, BookDocument("group-one.txt"), chat_id=-1001),
                context,
            )
            await telegram_bot.handle_document(
                self.update_for(101, BookDocument("group-two.txt"), chat_id=-1002),
                context,
            )

        asyncio.run(run())

        self.assertEqual(len(telegram_bot.user_books), 2)
        self.assertNotEqual(
            telegram_bot._user_storage_dirs[(-1001, 101)],
            telegram_bot._user_storage_dirs[(-1002, 101)],
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
        self.assertEqual(set(telegram_bot.user_books), {(101, 101)})
        self.assertTrue(telegram_bot._user_storage_dirs[(101, 101)].is_dir())

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

        self.assertEqual(set(telegram_bot.user_books), {(202, 202)})
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
        self.assertEqual(telegram_bot._user_storage_dirs[(303, 303)], storage_dir)

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
        telegram_bot.user_books[(101, 101)] = FakeKnowledgeBase("one")
        telegram_bot.user_books[(101, 101)].documents = ["user one"]
        update = self.update_for(101, text="What is private?")
        context = SimpleNamespace()

        asyncio.run(telegram_bot.handle_question(update, context))

        self.assertTrue(
            any(
                "answer for What is private?" in message
                for message in update.message.messages
            )
        )

    def test_question_response_includes_bounded_retrieval_sources(self):
        telegram_bot.user_books[(101, 101)] = FakeKnowledgeBase("one")
        telegram_bot.user_books[(101, 101)].documents = ["user one"]
        update = self.update_for(101, text="What is private?")

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        response = update.message.messages[-1]
        self.assertIn("A: [1] answer for What is private?", response)
        self.assertIn("Sources (retrieved excerpts):", response)
        self.assertIn("[1] private document", response)

    def test_question_response_is_capped_to_telegram_message_limit(self):
        class HugeAnswerKnowledgeBase(FakeKnowledgeBase):
            def answer_question(self, question, top_k, short):
                del question, top_k, short
                return "x" * 10_000, ["source"]

        key = (101, 101)
        telegram_bot.user_books[key] = HugeAnswerKnowledgeBase("one")
        telegram_bot.user_books[key].documents = ["user one"]
        update = self.update_for(101, text="What is private?")

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        response = update.message.messages[-1]
        self.assertLessEqual(
            len(response), telegram_bot.MAX_TELEGRAM_MESSAGE_CHARACTERS
        )
        self.assertTrue(response.endswith("..."))

    def test_question_falls_back_to_new_reply_when_status_edit_fails(self):
        class EditFailingMessage(FakeMessage):
            async def reply_text(self, text):
                self.messages.append(text)
                return FailingStatus(self.messages)

        telegram_bot.user_books[(101, 101)] = FakeKnowledgeBase("one")
        telegram_bot.user_books[(101, 101)].documents = ["user one"]
        messages = []
        update = SimpleNamespace(
            message=EditFailingMessage(messages, text="What is private?"),
            effective_user=SimpleNamespace(id=101),
            effective_chat=SimpleNamespace(id=101),
        )

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        self.assertEqual(messages[0], "Searching...")
        self.assertIn("answer for What is private?", messages[-1])

    def test_upload_rate_limit_rejects_excess_work(self):
        async def run():
            with patch.object(telegram_bot, "MAX_UPLOADS_PER_WINDOW", 1):
                context = SimpleNamespace(bot=FakeBot(b"book text"))
                await telegram_bot.handle_document(
                    self.update_for(101, BookDocument("first.txt")), context
                )
                second_update = self.update_for(101, BookDocument("second.txt"))
                await telegram_bot.handle_document(second_update, context)
                return second_update

        second_update = asyncio.run(run())

        self.assertIn("too many uploads", second_update.message.messages[-1].lower())

    def test_request_tracking_memory_is_bounded(self):
        with patch.object(telegram_bot, "MAX_TRACKED_SESSIONS", 2):
            for session_key in ((1, 1), (2, 2), (3, 3)):
                self.assertTrue(telegram_bot._allow_request(session_key, "question", 1))

        self.assertEqual(len(telegram_bot._request_history), 2)
        self.assertNotIn((1, 1), telegram_bot._request_history)

    def test_question_without_a_book_does_not_use_another_user_book(self):
        telegram_bot.user_books[(101, 101)] = FakeKnowledgeBase("one")
        telegram_bot.user_books[(101, 101)].documents = ["user one"]
        update = self.update_for(202, text="What is private?")

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        self.assertEqual(update.message.messages, ["No book loaded. Use /load_book"])

    def test_question_length_is_bounded(self):
        telegram_bot.user_books[(101, 101)] = FakeKnowledgeBase("one")
        telegram_bot.user_books[(101, 101)].documents = ["user one"]
        update = self.update_for(101, text="x" * (telegram_bot.MAX_QUESTION_LENGTH + 1))

        asyncio.run(telegram_bot.handle_question(update, SimpleNamespace()))

        self.assertIn("1,000 characters", update.message.messages[0])

    def test_evicted_session_releases_its_lock(self):
        for session_key in ((1, 1), (2, 2)):
            telegram_bot.user_books[session_key] = FakeKnowledgeBase("book")
            telegram_bot._active_user_order[session_key] = None
            telegram_bot._user_locks[session_key] = asyncio.Lock()

        with patch.object(telegram_bot, "MAX_ACTIVE_USERS", 2):
            asyncio.run(
                telegram_bot.handle_document(
                    self.update_for(3, BookDocument("third.txt")),
                    SimpleNamespace(bot=FakeBot(b"book text")),
                )
            )

        self.assertNotIn((1, 1), telegram_bot._user_locks)
        self.assertIn((2, 2), telegram_bot._user_locks)
        self.assertIn((3, 3), telegram_bot._user_locks)


class TelegramBotApplicationBuilderTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import AsyncMock

        bot_user = User(
            id=123, is_bot=True, first_name="Fake", username="fakebot"
        )
        self.patches = [
            patch("telegram.Bot.get_me", new=AsyncMock(return_value=bot_user))
        ]
        for active_patch in self.patches:
            active_patch.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        asyncio.run(self._setup_application())

    async def _setup_application(self):
        self.application = telegram_bot._build_application("123:FAKETOKEN")
        telegram_bot._install_handlers(self.application)
        await self.application.initialize()
        self.application.bot._bot_user = User(
            id=123, is_bot=True, first_name="Fake", username="fakebot"
        )

    def tearDown(self):
        asyncio.run(self.application.shutdown())

    def test_updates_are_processed_with_a_bounded_concurrency(self):
        self.assertEqual(
            self.application.concurrent_updates,
            telegram_bot.MAX_CONCURRENT_UPDATES,
        )

    def test_message_edits_and_channel_posts_never_reach_handlers(self):
        from datetime import datetime, timezone

        from telegram import Chat, Message, Update

        now = datetime.now(timezone.utc)
        document_edit = Update(
            update_id=1,
            edited_message=Message(
                message_id=2,
                date=now,
                chat=Chat(id=1, type=Chat.PRIVATE),
                document=SimpleNamespace(file_name="book.txt", file_size=10),
            ),
        )
        text_edit = Update(
            update_id=2,
            edited_message=Message(
                message_id=3, date=now, chat=Chat(id=1, type=Chat.PRIVATE), text="hello"
            ),
        )
        command_edit = Update(
            update_id=3,
            edited_message=Message(
                message_id=4, date=now, chat=Chat(id=1, type=Chat.PRIVATE), text="/start"
            ),
        )
        for message in (
            document_edit.edited_message,
            text_edit.edited_message,
            command_edit.edited_message,
        ):
            message.set_bot(self.application.bot)

        handlers = self.application.handlers[0]
        document_handler, question_handler = (
            handler for handler in handlers if isinstance(handler, MessageHandler)
        )
        self.assertFalse(document_handler.check_update(document_edit))
        self.assertFalse(question_handler.check_update(text_edit))
        command_handlers = [h for h in handlers if isinstance(h, CommandHandler)]
        self.assertGreaterEqual(len(command_handlers), 4)
        for handler in command_handlers:
            self.assertFalse(handler.check_update(command_edit))

    def test_real_messages_still_reach_every_handler(self):
        from datetime import datetime, timezone

        from telegram import Chat, Message, MessageEntity, Update

        now = datetime.now(timezone.utc)
        chat = Chat(id=1, type=Chat.PRIVATE)
        document_update = Update(
            update_id=1,
            message=Message(
                message_id=2,
                date=now,
                chat=chat,
                document=SimpleNamespace(file_name="book.txt", file_size=10),
            ),
        )
        text_update = Update(
            update_id=2,
            message=Message(
                message_id=3, date=now, chat=chat, text="plain question"
            ),
        )
        start_update = Update(
            update_id=3,
            message=Message(
                message_id=4,
                date=now,
                chat=chat,
                text="/start",
                entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=6)],
            ),
        )
        for message in (
            document_update.message,
            text_update.message,
            start_update.message,
        ):
            message.set_bot(self.application.bot)

        handlers = self.application.handlers[0]
        command_handlers = [h for h in handlers if isinstance(h, CommandHandler)]
        self.assertTrue(
            any(h.check_update(start_update) for h in command_handlers)
        )
        message_handlers = [
            h for h in handlers if isinstance(h, MessageHandler)
        ]
        self.assertEqual(len(message_handlers), 2)
        self.assertTrue(message_handlers[0].check_update(document_update))
        self.assertTrue(message_handlers[1].check_update(text_update))


if __name__ == "__main__":
    unittest.main()
