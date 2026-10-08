# book-qa-telegram-bot

Telegram bot (python-telegram-bot 21.5, long polling) that accepts PDF and TXT uploads, splits
the text into overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and
replies with the first sentences of the best-matching chunk. It does not call an LLM. Per-user
state is keyed by `(chat_id, user_id)`, and embeddings are cached on disk as `.npy` and JSON
files with SHA256 digests.

## Commands

CI runs these on Python 3.11 and 3.12. Torch is installed first from the CPU index, then the lock:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu --no-deps "torch==2.14.0+cpu"
python -m pip install -r requirements.lock
python -m unittest discover -s tests -v
python -m pip install --dry-run --no-index -r requirements.txt -r requirements.lock
python -m pip check
python -m py_compile file_utils.py telegram_bot.py book_qa.py
```

The dry-run step checks that `requirements.txt` agrees with `requirements.lock`.

## Layout

- `telegram_bot.py`: handlers, per-session state, per-session locks, the global processing slot and per-user rate limiting.
- `book_qa.py`: text chunking, embedding, similarity search and the on-disk embedding cache.
- `file_utils.py`: file helpers; CI compiles it with the other two modules.
- `tests/test_book_qa.py`: unit tests, including `FakeEmbeddingModel`.
- `requirements.txt`: direct dependencies. `requirements.lock`: full lock that CI installs.
- `.env.example`: `TELEGRAM_BOT_TOKEN` and the optional `EMBEDDING_MODEL_REVISION`.

## Rules for changes

- Regenerate the lock with the command recorded in its header. The header also records the indexes, which must be used too:

  ```bash
  uv pip compile requirements.txt --python-version 3.11 --universal --index-strategy unsafe-best-match --output-file requirements.lock
  ```

  Index URLs used for the lock: `--index-url https://download.pytorch.org/whl/cpu` and `--extra-index-url https://pypi.org/simple`.
- After changing `requirements.txt`, the lock must still pass the CI dry-run check.
- Unit tests use `FakeEmbeddingModel` from `tests/test_book_qa.py` for embedding.
- The default embedding model is pinned to a revision. `EMBEDDING_MODEL_REVISION` overrides it; an empty value follows the Hub default, as `.env.example` describes.
- Keep `.npy` loading with `allow_pickle=False` and the SHA256 digest check. Keep the 0700/0600 modes on cache and upload files.
- Keep upload cleanup in a `finally` block, and keep session keys isolated by `(chat_id, user_id)`.
- The unit test suite imports torch and sentence-transformers, so low-memory hosts may not be able to run it.
