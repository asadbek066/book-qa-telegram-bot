# Book Q&A Telegram Bot

Telegram bot that answers questions about an uploaded PDF or TXT book. It extracts the
text, chunks it, embeds the chunks with `sentence-transformers` (`all-MiniLM-L6-v2`),
and for each question replies with the best-matching sentence plus numbered source
excerpts. Answers are extractive: there is no generative model.

## Layout

- `telegram_bot.py` – handlers, per-session state, rate limits, cache pruning
- `book_qa.py` – text extraction, chunking, embeddings, retrieval
- `file_utils.py` – atomic writes and safe file handling
- `tests/` – unit tests (`unittest`, no network or model download)
- `scripts/relock.sh` – regenerates `requirements.lock`

## Setup

Python 3.11 or 3.12.

```bash
# The CPU Torch wheel only exists on the PyTorch index, so install it first.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --no-deps "torch==2.14.0+cpu"
python -m pip install -r requirements.lock

cp .env.example .env   # set TELEGRAM_BOT_TOKEN
python telegram_bot.py
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | required | Bot token from BotFather |
| `BOOKS_DIR` | `books` | Staging directory for uploads (removed after processing) |
| `EMBEDDINGS_DIR` | `embeddings` | Embedding caches, one directory per session |
| `EMBEDDING_MODEL_REVISION` | pinned | Hub revision of the model; empty follows the Hub default |
| `MIN_SIMILARITY_SCORE` | `0.0` | Abstain threshold, see below |

## Usage

Run `/load_book`, upload a PDF or TXT file, then ask questions in chat. Commands:
`/start`, `/help`, `/load_book`, `/summary`.

Each private chat, and each user within a group chat, has its own active book.
Uploading a new book replaces the old one; there is no delete command.

## Development

```bash
python -m unittest discover -s tests -v
ruff check . && ruff format --check .
```

## Behaviour

**Chunking.** Chunks are sized to the embedding model: the token budget comes from the
model's `max_seq_length` and tokenizer (254 tokens for `all-MiniLM-L6-v2`) with 10%
overlap, falling back to 150 words if the model does not expose them. The chunking
parameters are stored with the cache, and a cache built with different parameters
is rejected, so the book has to be uploaded again.

**Answers.** The reply is the sentence of the top chunk that best matches the question,
followed by the next sentences up to 30 words, then numbered excerpts of the retrieved
chunks. Excerpts are not PDF page numbers. Scanned PDFs without selectable text will not
work.

**Abstaining.** Off by default. With `MIN_SIMILARITY_SCORE=0.0` only negative or NaN
scores are rejected, so every question gets the best-matching text. A higher value makes
the bot reply "I could not find this in the book." when the best chunk's cosine
similarity is below it. The model is English-centric: for other languages and for terse
questions real matches can score only 0.05–0.2, so calibrate first. Set the `book_qa`
logger to DEBUG to log each question's top score (never the question text), compare
scores for answerable and off-topic questions, and pick a value between the two groups.
A value of 1 or more makes every question abstain.

**Caches.** Embeddings are stored as NumPy arrays and documents as JSON under
`embeddings/<telegram-user-id>/` (private chats) or `embeddings/chat-<chat-id>/`
(groups). Writes are atomic, and a cache is validated against the model revision before
use. After a restart the newest valid cache for a session is restored on its first
interaction without loading the model.

## Limits

| What | Limit |
| --- | --- |
| Upload size | 20 MB |
| PDF pages / decoded stream / extracted text | 500 / 2 MB / 2 million characters |
| Book processing / question time | 5 min / 2 min |
| Concurrent jobs | 4 processing, 8 answering |
| Rate per session | 5 uploads and 30 questions per minute |
| Books held in memory | 32 sessions (eviction keeps the disk cache) |
| Disk cache | 256 books and 2 GiB, least recently used removed first |

A job that times out keeps its slot until its worker thread finishes; new work is
declined while all slots are busy. The disk bounds are `MAX_CACHED_BOOKS` and
`MAX_CACHE_BYTES` in `telegram_bot.py`.

## Security

The bot has no owner allowlist: anyone who can message it can upload a book. Keep the
token private and restrict filesystem access to `books/` and `embeddings/`.

## Updating dependencies

`requirements.txt` holds the source ranges; `requirements.lock` is compiled from it
with [uv](https://docs.astral.sh/uv/).

- `scripts/relock.sh` runs the `uv pip compile` command (including the PyTorch CPU
  index) and then the agreement check CI runs. The check uses
  `pip --dry-run --no-index`, so if versions changed, install the new lock first and
  run `scripts/relock.sh --check-only`.
- `.github/workflows/relock.yml` runs weekly or on manual dispatch. If the lock
  changed it runs the tests and opens a PR from a new `automation/relock-<run id>`
  branch. It never merges or force-pushes, and it skips while an earlier relock PR is
  open. PRs created with `GITHUB_TOKEN` do not trigger CI; close and reopen to run it.
- Dependabot version-update PRs are switched off (`open-pull-requests-limit: 0`);
  bump ranges in `requirements.txt` by hand and relock.
