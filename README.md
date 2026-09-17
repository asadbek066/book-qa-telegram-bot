# Book Q&A Telegram Bot

Telegram bot for document-grounded Q&A over uploaded PDF and TXT books.

## Overview

This project lets a user upload a book and ask questions about its content in chat.
The bot extracts text, chunks it, embeds the chunks, retrieves the most relevant parts,
and returns a concise answer.

## Features

- Upload PDF or TXT files directly in Telegram
- Extract and chunk book content for retrieval
- Embedding-based similarity search with `sentence-transformers`
- Short, context-based answers from top relevant chunks with numbered source excerpts
- Basic commands for loading, summary, and help
- Per-chat/user book isolation with collision-resistant temporary storage
- Bounded uploads, extraction, Telegram responses, and per-session request rates
  to protect the bot from resource exhaustion

## Tech Stack

- Python
- `python-telegram-bot`
- `sentence-transformers`
- `pypdf`
- `torch`

## Project Structure

- `telegram_bot.py`: Telegram bot handlers and command flow
- `book_qa.py`: document processing, chunking, embeddings, and retrieval
- `.env.example`: environment template

## Setup

1. Create and activate a virtual environment with Python 3.11 or 3.12.
2. Install dependencies:

```bash
# The CPU Torch wheel only exists on the PyTorch index, so install it first;
# every other locked dependency resolves from PyPI without index confusion.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --no-deps "torch==2.14.0+cpu"
python -m pip install -r requirements.lock
```

`requirements.txt` is the human-maintained source constraint. Regenerate
`requirements.lock` with the command recorded at its top when dependencies
are intentionally upgraded:

```bash
uv pip compile requirements.txt --python-version 3.11 --universal \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --output-file requirements.lock
```

3. Create `.env` from `.env.example` and set:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
```

4. Run:

```bash
python telegram_bot.py
```

## Usage

1. Start the bot and run `/load_book`
2. Upload a PDF or TXT file
3. Ask questions in chat

Available commands:

- `/start`
- `/help`
- `/load_book`
- `/summary`

## Notes

- Answers are retrieval-based and limited by extracted text quality. Telegram responses include
  bounded numbered retrieval excerpts so users can verify what text supported the answer; these
  are excerpt citations rather than PDF page numbers.
- Each response is capped below Telegram's message-size limit. If Telegram cannot edit the
  temporary status message, the bot sends the completed response as a new reply.
- Scanned PDFs without selectable text may not work well.
- Uploads are limited to 20 MB, PDFs to 500 pages, and extracted text to 2 million characters.
- Book processing is time-bounded to 5 minutes and each question to 2 minutes; when a
  bound is exceeded the user gets an error and the late result's cache is discarded.
- The default embedding model is loaded from a pinned Hub revision; set
  `EMBEDDING_MODEL_REVISION` to a different revision, or empty it to follow the Hub
  default (for example when the pinned revision is not available offline).
- Uploads and questions for the same chat session are serialized so a read cannot
  observe a half-replaced book.
- Each private chat and group-chat/user pair has an independent active book. Source uploads are
  removed after processing; private-chat caches remain under `embeddings/<telegram-user-id>/`,
  while group-chat caches are namespaced under `embeddings/chat-<telegram-chat-id>/`.
- Uploads are limited to 5 per session per minute and questions to 30 per session per minute.
- The process keeps at most 32 active chat sessions' books in memory; evicted sessions can upload
  their book again.
- After a restart, the newest valid cache for that chat session is restored on its first interaction.
- Embeddings are stored as NumPy arrays and documents as JSON. Cache writes are atomic and cache
  contents are validated before use; legacy pickle caches are not loaded.
- The bot has no owner allowlist. Keep the bot token private and treat the local `books/` and
  `embeddings/` directories as sensitive. Restrict filesystem access to the bot process.
