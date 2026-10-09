import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pypdf
import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

MAX_PDF_PAGES = 500
MAX_PDF_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_STREAM_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_PDF_RECOVERY_INPUT_BYTES = 1 * 1024 * 1024
MAX_PDF_XFORM_INVOCATIONS = 32
MAX_TEXT_CHARACTERS = 2_000_000
MAX_CHUNKS = 10_000
MAX_EMBEDDING_DIMENSIONS = 4_096
# Version 3 records the chunking parameters in the cache, so caches built with
# the older 500-word chunks (versions 1-2) are rebuilt instead of mixed in.
CACHE_FORMAT_VERSION = 3
# all-MiniLM-L6-v2 truncates its input at 256 word pieces, so chunks must be
# sized to the model. With a model that exposes max_seq_length and a tokenizer
# the chunk budget is counted in tokens (max_seq_length minus special tokens);
# otherwise a conservative word count is used: 150 English words are roughly
# 200 word pieces, below the 254 usable by the default model.
FALLBACK_CHUNK_WORDS = 150
CHUNK_OVERLAP_RATIO = 0.1
MIN_TOKEN_CHUNK_BUDGET = 8
# Words are tokenised in batches (one tokenizer call per batch, not per word).
TOKENIZE_BATCH_WORDS = 2_048
# No piece of text counted as one "word" may exceed this many characters, even
# if the tokenizer collapses it to a single unknown token; this keeps a long
# base64-like blob from producing a multi-megabyte chunk.
MAX_PIECE_CHARACTERS = 100
# Opt-in abstention: when the best chunk's cosine similarity is not at least
# MIN_SIMILARITY_SCORE (environment variable, read per question) the bot answers
# NOT_FOUND_MESSAGE instead of book text. The default 0.0 effectively disables it
# (only negative scores, and NaN, abstain): all-MiniLM-L6-v2 is English-centric,
# so for other languages and terse questions genuine matches can score only
# 0.05-0.2, and a false "not found" is worse than a weak answer. Calibrate on
# your own books (see README) before raising it.
DEFAULT_MIN_SIMILARITY_SCORE = 0.0
NOT_FOUND_MESSAGE = "I could not find this in the book."
STALE_CACHE_MESSAGE = (
    "This saved book was prepared with different chunking settings and cannot "
    "be used. Please upload it again."
)
# Upper bound on sentences re-embedded to pick the best one in the top chunk,
# so a question costs at most two encode calls (question + one sentence batch).
MAX_SENTENCE_CANDIDATES = 64
MAX_ANSWER_WORDS = 30
MAX_BOOK_NAME_LENGTH = 128
MAX_DOCUMENT_CACHE_BYTES = MAX_TEXT_CHARACTERS * 4 + 64 * 1024
MAX_EMBEDDINGS_CACHE_BYTES = 64 * 1024 * 1024
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Pin the Hub revision of the shared default embedding model so a compromised
# or re-tagged upstream repository cannot swap the weights this bot loads.
# Read lazily so the EMBEDDING_MODEL_REVISION override from .env applies; an
# empty value follows the Hub default (for example when the pinned revision is
# unavailable offline).
DEFAULT_EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class BookChunk(str):
    """A retrieved chunk of text that also knows its PDF page range.

    ``pages`` is ``(first, last)`` using 1-based PDF page positions (not printed
    page labels), or ``None`` for books without pages (TXT) and for caches
    written before page data was stored. Being a ``str`` keeps every consumer
    that treats chunks as plain text working unchanged.
    """

    pages: tuple[int, int] | None

    def __new__(cls, text: str, pages: tuple[int, int] | None = None):
        chunk = super().__new__(cls, text)
        chunk.pages = pages
        return chunk


def format_page_reference(pages: tuple[int, int] | None) -> str:
    """Return "p. 12" or "pp. 12-13", or "" when there is no page reference."""
    if pages is None:
        return ""
    first, last = pages
    return f"p. {first}" if first == last else f"pp. {first}-{last}"


_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _embedding_model_revision(model_name: str) -> str | None:
    if model_name != DEFAULT_EMBEDDING_MODEL:
        return None
    return (
        os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_EMBEDDING_MODEL_REVISION).strip()
        or None
    )


_WARNED_SETTINGS: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED_SETTINGS:
        _WARNED_SETTINGS.add(key)
        logger.warning(message)


def _min_similarity_score() -> float:
    raw = os.getenv("MIN_SIMILARITY_SCORE", "").strip()
    if not raw:
        return DEFAULT_MIN_SIMILARITY_SCORE
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        _warn_once(
            "invalid", "Ignoring invalid MIN_SIMILARITY_SCORE; using the default"
        )
        return DEFAULT_MIN_SIMILARITY_SCORE
    if value >= 1.0:
        _warn_once(
            "high",
            "MIN_SIMILARITY_SCORE is >= 1, so every question will get the not-found reply",
        )
    return max(-1.0, min(1.0, value))


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load one shared, read-only embedding model per configured model name."""
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(
                model_name,
                revision=_embedding_model_revision(model_name),
                trust_remote_code=False,
            )
            _MODEL_CACHE[model_name] = model
        return model


def _atomic_write(path: Path, mode: str, writer: Callable[[Any], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        encoding = "utf-8" if "b" not in mode else None
        with os.fdopen(file_descriptor, mode, encoding=encoding) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class BookKnowledgeBase:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model: Any | None = None,
        storage_dir: str | Path = "embeddings",
    ):
        self.model_name = model_name
        self._model = model
        self.documents: list[str] = []
        # Parallel to ``documents``: (first, last) 1-based PDF page per chunk,
        # or None when the book has no page data (TXT, or an older cache).
        self.pages: list[tuple[int, int]] | None = None
        self.embeddings: Any | None = None
        self.book_name: str | None = None
        # Chunking parameters of a restored cache that could not yet be compared
        # with the model's (see load_embeddings); checked on the first question.
        self._unverified_chunking: dict[str, Any] | None = None
        self.storage_dir = Path(storage_dir)
        self.embeddings_file = self.storage_dir / "book_embeddings.npy"
        self.documents_file = self.storage_dir / "book_documents.json"

    def _loaded_model(self) -> Any | None:
        """Return the embedding model only if it is already in memory."""
        if self._model is not None:
            return self._model
        with _MODEL_CACHE_LOCK:
            return _MODEL_CACHE.get(self.model_name)

    @staticmethod
    def _chunking_is_valid(chunking: Any) -> bool:
        return (
            isinstance(chunking, dict)
            and chunking.get("unit") in ("tokens", "words")
            and all(
                isinstance(chunking.get(key), int)
                and not isinstance(chunking.get(key), bool)
                for key in ("chunk_size", "overlap")
            )
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _get_embedding_model(self.model_name)
        return self._model

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        pages = self.extract_pdf_pages(pdf_path)
        return "\n".join(pages) if pages else ""

    def extract_pdf_pages(self, pdf_path: str | Path) -> list[str] | None:
        """Return the text of every PDF page in order (index 0 is page 1).

        Returns None when the PDF is rejected or unreadable.
        """
        try:
            path = Path(pdf_path)
            if path.stat().st_size > MAX_PDF_FILE_BYTES:
                logger.warning("PDF rejected: file size limit exceeded")
                return None

            with (
                pypdf.apply_configuration(
                    maximum_declared_stream_length=MAX_PDF_FILE_BYTES,
                    array_based_stream_maximum_output_length=MAX_PDF_STREAM_OUTPUT_BYTES,
                    jbig2_maximum_output_length=MAX_PDF_STREAM_OUTPUT_BYTES,
                    lzw_maximum_output_length=MAX_PDF_STREAM_OUTPUT_BYTES,
                    run_length_maximum_output_length=MAX_PDF_STREAM_OUTPUT_BYTES,
                    zlib_maximum_output_length=MAX_PDF_STREAM_OUTPUT_BYTES,
                    zlib_maximum_recovery_input_length=MAX_PDF_RECOVERY_INPUT_BYTES,
                    flate_maximum_row_length=MAX_PDF_STREAM_OUTPUT_BYTES,
                    image_maximum_buffer_size=MAX_PDF_STREAM_OUTPUT_BYTES,
                    xmp_maximum_input_length=MAX_PDF_RECOVERY_INPUT_BYTES,
                    xform_maximum_invocations_per_extraction=MAX_PDF_XFORM_INVOCATIONS,
                ),
                path.open("rb") as file,
            ):
                pdf_reader = pypdf.PdfReader(file)
                page_count = len(pdf_reader.pages)
                if page_count > MAX_PDF_PAGES:
                    logger.warning("PDF rejected: page limit exceeded")
                    return None

                text_parts: list[str] = []
                character_count = 0
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    character_count += len(page_text)
                    if character_count > MAX_TEXT_CHARACTERS:
                        logger.warning("PDF rejected: extracted text limit exceeded")
                        return None
                    text_parts.append(page_text)
                return text_parts
        except Exception as exc:  # noqa: BLE001 - malformed PDFs are user input
            logger.error("PDF extraction failed: %s", type(exc).__name__)
            return None

    def extract_book(self, file_path: str | Path) -> tuple[str, list[str] | None]:
        """Return the book text and, for PDFs, the per-page texts (else None).

        This is the single extraction seam: ``load_book`` and
        ``extract_text_from_file`` both go through it, and PDFs are read only by
        ``extract_pdf_pages``. Any failure is logged and yields ``("", None)``.
        """
        path = Path(file_path)
        try:
            if not path.is_file():
                return "", None
            if path.suffix.lower() == ".pdf":
                pages = self.extract_pdf_pages(path)
                if not pages:
                    return "", None
                return "\n".join(pages), pages
            if path.suffix.lower() != ".txt":
                return "", None
            with path.open("r", encoding="utf-8", errors="replace") as file:
                text = file.read(MAX_TEXT_CHARACTERS + 1)
            if len(text) > MAX_TEXT_CHARACTERS:
                logger.warning("Text file rejected: extracted text limit exceeded")
                return "", None
            return text, None
        except Exception as exc:  # noqa: BLE001 - local files are user input
            logger.error("Book extraction failed: %s", type(exc).__name__)
            return "", None

    def extract_text_from_file(self, file_path: str | Path) -> str:
        """Thin wrapper over ``extract_book`` that returns only the text."""
        return self.extract_book(file_path)[0]

    @staticmethod
    def _token_budget(model: Any) -> tuple[int, Any] | None:
        """Return (usable tokens, tokenizer) when the model exposes its limit."""
        limit = getattr(model, "max_seq_length", None)
        tokenizer = getattr(model, "tokenizer", None)
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
            or not (
                callable(tokenizer) or callable(getattr(tokenizer, "tokenize", None))
            )
        ):
            return None
        special = 2
        add_special = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(add_special):
            try:
                special = int(add_special(False))
            except (TypeError, ValueError):
                special = 2
        budget = limit - special
        if budget < MIN_TOKEN_CHUNK_BUDGET:
            return None
        return budget, tokenizer

    def chunking_parameters(self, model: Any | None = None) -> dict[str, Any]:
        """Describe how this knowledge base chunks text; stored with the cache."""
        token_budget = self._token_budget(
            model if model is not None else self._get_model()
        )
        if token_budget is not None:
            budget = token_budget[0]
            return {
                "unit": "tokens",
                "chunk_size": budget,
                "overlap": max(1, int(budget * CHUNK_OVERLAP_RATIO)),
            }
        return {
            "unit": "words",
            "chunk_size": FALLBACK_CHUNK_WORDS,
            "overlap": int(FALLBACK_CHUNK_WORDS * CHUNK_OVERLAP_RATIO),
        }

    def _resolve_chunking(
        self, chunk_size: int | None, overlap: int | None
    ) -> dict[str, Any]:
        if chunk_size is not None or overlap is not None:
            params = {
                "unit": "words",
                "chunk_size": FALLBACK_CHUNK_WORDS
                if chunk_size is None
                else chunk_size,
                "overlap": (
                    int(FALLBACK_CHUNK_WORDS * CHUNK_OVERLAP_RATIO)
                    if overlap is None
                    else overlap
                ),
            }
        else:
            params = self.chunking_parameters()
        if (
            params["chunk_size"] <= 0
            or params["overlap"] < 0
            or params["overlap"] >= params["chunk_size"]
        ):
            raise ValueError("overlap must be smaller than chunk_size")
        return params

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Split text into overlapping chunks that fit the embedding model.

        Explicit ``chunk_size``/``overlap`` are word counts. Without them the
        size is derived from the model (see ``chunking_parameters``).
        """
        params = self._resolve_chunking(chunk_size, overlap)
        return [chunk for chunk, _ in self._chunk_words(text.split(), None, params)]

    def chunk_pages(
        self,
        pages: list[str],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Chunk per-page texts; also return each chunk's (first, last) page.

        Page numbers are 1-based positions in the page list (empty pages keep
        their position). The chunks are identical to ``chunk_text`` on the pages
        joined with newlines. Runs in linear time.
        """
        params = self._resolve_chunking(chunk_size, overlap)
        words: list[str] = []
        word_pages: list[int] = []
        for page_number, page_text in enumerate(pages, start=1):
            page_words = page_text.split()
            words.extend(page_words)
            word_pages.extend([page_number] * len(page_words))
        result = self._chunk_words(words, word_pages, params)
        return (
            [chunk for chunk, _ in result],
            [span for _, span in result if span is not None],
        )

    def _chunk_words(
        self,
        words: list[str],
        word_pages: list[int] | None,
        params: dict[str, Any],
    ) -> list[tuple[str, tuple[int, int] | None]]:
        size, overlap_size = params["chunk_size"], params["overlap"]

        def span(first_word: int, last_word: int) -> tuple[int, int] | None:
            if word_pages is None:
                return None
            return word_pages[first_word], word_pages[last_word]

        if params["unit"] == "words":
            chunks: list[tuple[str, tuple[int, int] | None]] = []
            for i in range(0, len(words), size - overlap_size):
                end = min(i + size, len(words))
                chunk = " ".join(words[i:end])
                if chunk.strip():
                    chunks.append((chunk, span(i, end - 1)))
            return chunks

        tokenizer = self._token_budget(self._get_model())[1]  # type: ignore[index]
        pieces, counts, origins = self._pieces_with_counts(words, size, tokenizer)

        token_chunks: list[tuple[str, tuple[int, int] | None]] = []
        start = 0
        while start < len(pieces):
            end, used = start, 0
            while end < len(pieces) and (end == start or used + counts[end] <= size):
                used += counts[end]
                end += 1
            token_chunks.append(
                (" ".join(pieces[start:end]), span(origins[start], origins[end - 1]))
            )
            if end >= len(pieces):
                break
            # Step back to share about `overlap_size` tokens with the next
            # chunk, always advancing by at least one piece.
            next_start, shared = end, 0
            while (
                next_start - 1 > start
                and shared + counts[next_start - 1] <= overlap_size
            ):
                next_start -= 1
                shared += counts[next_start]
            start = next_start
        return token_chunks

    @staticmethod
    def _token_counts(tokenizer: Any, texts: list[str]) -> list[int]:
        """Count word pieces per text with one batched tokenizer call."""
        if callable(tokenizer):
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            return [len(ids) for ids in encoded]
        return [len(tokenizer.tokenize(text)) for text in texts]

    @classmethod
    def _pieces_with_counts(
        cls, words: list[str], budget: int, tokenizer: Any
    ) -> tuple[list[str], list[int], list[int]]:
        """Split words so each piece fits the budget.

        Returns the pieces, their token counts and the index of the source word
        each piece came from.
        """
        flat: list[str] = []
        flat_origins: list[int] = []
        for word_index, word in enumerate(words):
            if len(word) > MAX_PIECE_CHARACTERS:
                parts = [
                    word[i : i + MAX_PIECE_CHARACTERS]
                    for i in range(0, len(word), MAX_PIECE_CHARACTERS)
                ]
                flat.extend(parts)
                flat_origins.extend([word_index] * len(parts))
            else:
                flat.append(word)
                flat_origins.append(word_index)

        pieces: list[str] = []
        counts: list[int] = []
        origins: list[int] = []
        for offset in range(0, len(flat), TOKENIZE_BATCH_WORDS):
            batch = flat[offset : offset + TOKENIZE_BATCH_WORDS]
            for piece, count, origin in zip(
                batch,
                cls._token_counts(tokenizer, batch),
                flat_origins[offset : offset + TOKENIZE_BATCH_WORDS],
                strict=True,
            ):
                if count > budget:
                    # Rare: re-count only the pieces that are too large.
                    for part, part_count in cls._split_to_budget(
                        piece, budget, tokenizer, count
                    ):
                        pieces.append(part)
                        counts.append(part_count)
                        origins.append(origin)
                else:
                    pieces.append(piece)
                    counts.append(max(1, count))
                    origins.append(origin)
        return pieces, counts, origins

    @classmethod
    def _split_to_budget(
        cls, word: str, budget: int, tokenizer: Any, count: int | None = None
    ) -> list[tuple[str, int]]:
        """Halve a single over-long word until each piece fits the budget."""
        if count is None:
            count = cls._token_counts(tokenizer, [word])[0]
        if count <= budget or len(word) < 2:
            return [(word, max(1, count))]
        middle = len(word) // 2
        return cls._split_to_budget(
            word[:middle], budget, tokenizer
        ) + cls._split_to_budget(word[middle:], budget, tokenizer)

    def load_book(self, file_path: str | Path, book_name: str | None = None) -> bool:
        text, page_texts = self.extract_book(file_path)
        if not text:
            logger.warning("Book contains no extractable text")
            return False

        try:
            if page_texts is None:
                documents, pages = self.chunk_text(text), None
            else:
                documents, pages = self.chunk_pages(page_texts)
            if not documents or len(documents) > MAX_CHUNKS:
                logger.warning("Book rejected: chunk limit exceeded or no chunks")
                return False

            embeddings = self._get_model().encode(documents, convert_to_tensor=True)
            self.documents = documents
            self.pages = pages
            self.embeddings = embeddings
            self._unverified_chunking = None
            self.book_name = book_name or Path(file_path).stem
            if not self.save_embeddings():
                self.documents = []
                self.pages = None
                self.embeddings = None
                self.book_name = None
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - model and storage failures are recoverable
            logger.error("Book processing failed: %s", type(exc).__name__)
            self.documents = []
            self.pages = None
            self.embeddings = None
            self.book_name = None
            return False

    @staticmethod
    def _embedding_array(embeddings: Any) -> np.ndarray:
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        array = np.asarray(embeddings, dtype=np.float32)
        if (
            array.ndim != 2
            or array.shape[0] == 0
            or array.shape[1] == 0
            or array.shape[1] > MAX_EMBEDDING_DIMENSIONS
            or not np.isfinite(array).all()
        ):
            raise ValueError("embeddings must be a finite two-dimensional array")
        return array

    @staticmethod
    def _embedding_digest(embeddings: np.ndarray) -> str:
        return hashlib.sha256(embeddings.tobytes(order="C")).hexdigest()

    @staticmethod
    def _documents_are_valid(documents: Any) -> bool:
        if (
            not isinstance(documents, list)
            or not documents
            or len(documents) > MAX_CHUNKS
            or not all(isinstance(document, str) for document in documents)
            or not all(document.strip() for document in documents)
        ):
            return False
        return sum(len(document) for document in documents) <= MAX_TEXT_CHARACTERS

    @staticmethod
    def _pages_are_valid(pages: Any, document_count: int) -> bool:
        """Page data is optional (None); when present it must match the chunks."""
        if pages is None:
            return True
        if not isinstance(pages, list) or len(pages) != document_count:
            return False
        for entry in pages:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not all(
                    isinstance(number, int) and not isinstance(number, bool)
                    for number in entry
                )
                or not 1 <= entry[0] <= entry[1] <= MAX_PDF_PAGES
            ):
                return False
        return True

    @staticmethod
    def _book_name_is_valid(book_name: Any) -> bool:
        return book_name is None or (
            isinstance(book_name, str)
            and bool(book_name)
            and len(book_name) <= MAX_BOOK_NAME_LENGTH
        )

    def save_embeddings(self) -> bool:
        try:
            if not self._documents_are_valid(self.documents):
                raise ValueError("documents are invalid")
            if not self._book_name_is_valid(self.book_name):
                raise ValueError("book name is invalid")

            if not self._pages_are_valid(self.pages, len(self.documents)):
                raise ValueError("page data is invalid")

            embedding_array = self._embedding_array(self.embeddings)
            if embedding_array.shape[0] != len(self.documents):
                raise ValueError("embedding/document count mismatch")

            _atomic_write(
                self.embeddings_file,
                "wb",
                lambda handle: np.save(handle, embedding_array, allow_pickle=False),
            )
            _atomic_write(
                self.documents_file,
                "w",
                lambda handle: json.dump(
                    {
                        "version": CACHE_FORMAT_VERSION,
                        "model_name": self.model_name,
                        "model_revision": _embedding_model_revision(self.model_name),
                        "book_name": self.book_name,
                        "chunking": self.chunking_parameters(),
                        "embedding_sha256": self._embedding_digest(embedding_array),
                        "documents": self.documents,
                        "pages": (
                            None
                            if self.pages is None
                            else [list(entry) for entry in self.pages]
                        ),
                    },
                    handle,
                    ensure_ascii=False,
                ),
            )
            return True
        except Exception as exc:  # noqa: BLE001 - cache writes must fail closed
            logger.error("Could not save embeddings: %s", type(exc).__name__)
            return False

    def load_embeddings(self) -> bool:
        try:
            if not self.embeddings_file.is_file() or not self.documents_file.is_file():
                return False
            if self.documents_file.stat().st_size > MAX_DOCUMENT_CACHE_BYTES:
                return False
            if self.embeddings_file.stat().st_size > MAX_EMBEDDINGS_CACHE_BYTES:
                return False

            with self.documents_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                return False
            if payload.get("version") != CACHE_FORMAT_VERSION:
                return False
            if payload.get("model_name") != self.model_name:
                return False
            if payload.get("model_revision") != _embedding_model_revision(
                self.model_name
            ):
                return False

            # A cache built with different chunk sizes (for example another
            # model limit or changed defaults) must be rebuilt, not mixed in.
            # Restoring must not load (or download) the embedding model: when it
            # is not in memory yet the comparison is deferred to the first
            # question, which loads it under the question timeout.
            cached_chunking = payload.get("chunking")
            if not self._chunking_is_valid(cached_chunking):
                return False
            loaded_model = self._loaded_model()
            if loaded_model is not None and cached_chunking != self.chunking_parameters(
                loaded_model
            ):
                return False

            documents = payload.get("documents")
            book_name = payload.get("book_name")
            if not self._documents_are_valid(documents):
                return False
            if not self._book_name_is_valid(book_name):
                return False
            # Caches written before page citations have no "pages" key; they
            # stay valid and simply show no page references.
            raw_pages = payload.get("pages")
            if not self._pages_are_valid(raw_pages, len(documents)):
                return False

            embedding_array = np.load(self.embeddings_file, allow_pickle=False)
            if (
                not isinstance(embedding_array, np.ndarray)
                or embedding_array.ndim != 2
                or embedding_array.shape[0] != len(documents)
                or embedding_array.shape[1] == 0
                or embedding_array.shape[1] > MAX_EMBEDDING_DIMENSIONS
                or not np.isfinite(embedding_array).all()
            ):
                return False

            with np.errstate(over="ignore", invalid="ignore"):
                # No copy in the common case: a cache round-tripped through
                # np.save/np.load is already float32 and C-contiguous.
                embedding_values = np.asarray(embedding_array).astype(
                    np.float32, order="C", copy=False
                )
            # A finite float64 cache value can overflow while being converted to
            # float32. Reject the converted representation before similarity
            # search sees it.
            if not np.isfinite(embedding_values).all():
                return False
            if payload.get("embedding_sha256") != self._embedding_digest(
                embedding_values
            ):
                return False
            self.documents = documents
            self.pages = (
                None if raw_pages is None else [(int(a), int(b)) for a, b in raw_pages]
            )
            self.embeddings = torch.as_tensor(embedding_values)
            self.book_name = book_name
            self._unverified_chunking = (
                cached_chunking if loaded_model is None else None
            )
            logger.info("Loaded cached embeddings")
            return True
        except Exception as exc:  # noqa: BLE001 - malformed local cache is untrusted
            logger.error("Could not load embeddings: %s", type(exc).__name__)
            return False

    def retrieve(
        self, question: str, top_k: int = 3
    ) -> tuple[list[BookChunk], float, Any]:
        """Return the top_k chunks, the best cosine score and the question embedding.

        Callers must have checked that a book is loaded and the question is text.
        """
        try:
            result_count = max(1, min(int(top_k), len(self.documents)))
        except (TypeError, ValueError):
            result_count = min(3, len(self.documents))

        question_embedding = self._get_model().encode(question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(question_embedding, self.embeddings)[0]
        score_array = (
            cos_scores.detach().cpu().numpy()
            if hasattr(cos_scores, "detach")
            else np.asarray(cos_scores)
        )
        top_results = np.argsort(-score_array)[:result_count]
        top_score = float(score_array[int(top_results[0])])
        # Log the score only, never the question text.
        logger.debug("Top retrieval similarity: %.4f", top_score)
        has_pages = self.pages is not None and len(self.pages) == len(self.documents)
        chunks = [
            BookChunk(
                self.documents[int(index)],
                self.pages[int(index)] if has_pages else None,  # type: ignore[index]
            )
            for index in top_results
        ]
        return chunks, top_score, question_embedding

    def _best_sentences(self, chunk: str, question_embedding: Any) -> str:
        """Pick the sentence(s) of a chunk that best match the question.

        Costs one extra batched encode call over at most MAX_SENTENCE_CANDIDATES
        sentences, so a question needs at most two encode calls in total.
        """
        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", chunk)
            if sentence.strip()
        ]
        if not sentences:
            return ""

        start = 0
        candidates = sentences[:MAX_SENTENCE_CANDIDATES]
        if len(candidates) > 1:
            sentence_scores = util.pytorch_cos_sim(
                question_embedding,
                self._get_model().encode(candidates, convert_to_tensor=True),
            )[0]
            start = int(np.argmax(np.asarray(sentence_scores.detach().cpu())))

        short_answer = ""
        word_count = 0
        for sentence in sentences[start:]:
            words = sentence.split()
            if word_count + len(words) > MAX_ANSWER_WORDS:
                break
            short_answer = f"{short_answer} {sentence}".strip()
            word_count += len(words)

        if not short_answer:
            short_answer = " ".join(sentences[start].split()[:15]) + "..."
        return short_answer

    def answer_question(
        self, question: str, top_k: int = 3, short: bool = True
    ) -> tuple[str, list[str]]:
        """Answer from the book, or abstain with no chunks when nothing matches."""
        if not self.documents or self.embeddings is None:
            return "No book loaded", []
        if not isinstance(question, str) or not question.strip():
            return "Please ask a question", []

        # Loads the model if needed (inside the caller's question timeout) and
        # only then can a restored cache's chunking be compared with it.
        model = self._get_model()
        if self._unverified_chunking is not None:
            if self._unverified_chunking != self.chunking_parameters(model):
                self.documents = []
                self.pages = None
                self.embeddings = None
                self.book_name = None
                self._unverified_chunking = None
                return STALE_CACHE_MESSAGE, []
            self._unverified_chunking = None

        relevant_chunks, top_score, question_embedding = self.retrieve(question, top_k)
        # `not >=` so a NaN score abstains instead of counting as a hit.
        if not top_score >= _min_similarity_score():
            return NOT_FOUND_MESSAGE, []

        if short:
            answer = self._best_sentences(relevant_chunks[0], question_embedding)
            if not answer:
                return "No readable answer found", relevant_chunks
        else:
            context = "\n\n".join(relevant_chunks)
            answer = f"Based on the book:\n\n{context}"

        return answer, relevant_chunks

    def get_book_summary(self) -> str:
        if not self.documents:
            return "No book loaded"

        return f"Book: {self.book_name}\nChunks: {len(self.documents)}"
