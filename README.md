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
- Chunks are sized to the embedding model: the usable token budget is derived from the loaded
  model's `max_seq_length` and tokenizer (254 tokens for `all-MiniLM-L6-v2`) with 10% overlap,
  falling back to 150 words when the model does not expose them. Words are tokenised in batches
  and no single "word" may exceed 100 characters. Longer chunks would be
  silently truncated by the model. The chunking parameters are stored in the cache and a
  cache built with different parameters is rebuilt, so books uploaded before this change
  must be uploaded again.
- Abstaining on unrelated questions is opt-in. `MIN_SIMILARITY_SCORE` (read from the
  environment on each question) defaults to `0.0`, which only rejects negative or NaN scores,
  so in practice every question gets the best-matching text. Set it higher to make the bot
  reply "I could not find this in the book." (with no sources) when the best chunk's cosine
  similarity is below it. `all-MiniLM-L6-v2` is English-centric: for books or questions in
  other languages, and for terse questions, genuine matches can score only 0.05-0.2, so a high
  value produces false "not found" replies. Calibrate on your own books before enabling it:
  enable debug logging (`logging.getLogger("book_qa").setLevel(logging.DEBUG)`) to log each
  question's top similarity score (never the question text), collect scores for questions the
  book answers and for clearly off-topic ones, and set the value between the two groups. A value
  of 1 or more makes every question abstain (a warning is logged once).
- On a hit the short answer is the sentence of the top chunk that best matches the question
  (followed by the sentences after it, up to 30 words). This costs one extra batched embedding
  call over at most 64 sentences, so at most two `encode` calls per question.
- Scanned PDFs without selectable text may not work well.
- Uploads are limited to 20 MB, PDFs to 500 pages, each decoded PDF stream to 2 MB, and
  extracted text to 2 million characters.
- Book processing is time-bounded to 5 minutes and each question to 2 minutes; when a
  bound is exceeded the user gets an error and the late result's cache is discarded. At
  most 4 book-processing jobs and 8 question-answering jobs run at once; a timed-out job
  keeps its slot until its worker finishes, and new work is declined while all slots are full.
- The default embedding model is loaded from a pinned Hub revision; set
  `EMBEDDING_MODEL_REVISION` to a different revision, or empty it to follow the Hub
  default (for example when the pinned revision is not available offline).
- Uploads and questions for the same chat session are serialized so a read cannot
  observe a half-replaced book.
- Each private chat and group-chat/user pair has an independent active book. Source uploads are
  removed after processing; private-chat caches remain under `embeddings/<telegram-user-id>/`,
  while group-chat caches are namespaced under `embeddings/chat-<telegram-chat-id>/`.
- Uploads are limited to 5 per session per minute and questions to 30 per session per minute.
- The process keeps at most 32 active chat sessions' books in memory. Eviction frees memory only:
  the persisted cache stays on disk and is restored on the session's next interaction.
- Disk usage is bounded: a successful upload removes that session's older caches, and then the
  least recently used cached books (by directory mtime, refreshed on restore and on every
  question) are deleted until at most 256 books and 2 GiB remain (`MAX_CACHED_BOOKS`,
  `MAX_CACHE_BYTES` in `telegram_bot.py`). Caches of sessions held in memory are never pruned.
  There is no delete-my-book command; uploading a new book replaces the old one.
- After a restart, the newest valid cache for that chat session is restored on its first interaction.
  Restoring does not load the embedding model; the cache's chunking parameters are compared with
  the model's when it is first loaded for a question, and a mismatch asks the user to re-upload.
- Embeddings are stored as NumPy arrays and documents as JSON. Cache writes are atomic and cache
  contents are validated against the embedding model revision before use; legacy pickle caches are
  not loaded. Caches created before revision metadata was added must be rebuilt by re-uploading.
- The bot has no owner allowlist. Keep the bot token private and treat the local `books/` and
  `embeddings/` directories as sensitive. Restrict filesystem access to the bot process.

## Updating dependencies

`requirements.txt` holds the source ranges; `requirements.lock` is compiled from it with
[uv](https://docs.astral.sh/uv/).

- **Relock locally:** `scripts/relock.sh` runs the exact `uv pip compile` command (including
  the PyTorch CPU index) and then the same agreement check CI runs. That check uses
  `pip --dry-run --no-index`, so install the new lock first if versions changed (see Setup),
  then run `scripts/relock.sh --check-only`.
- **Weekly job:** `.github/workflows/relock.yml` runs every Monday (or on manual dispatch),
  and if the lock changed it runs the tests and agreement check and opens or updates a single
  PR on `automation/relock`. It never merges. Because that PR is created with `GITHUB_TOKEN`,
  the normal CI workflow does not start on it automatically; close and reopen the PR to run it.
- **Dependabot** handles GitHub Actions and pip bumps that fall outside the declared range
  in `requirements.txt` (`increase-if-necessary`). Those PRs fail the lock-agreement check
  until you run `scripts/relock.sh` and push the new lock to the PR branch.
