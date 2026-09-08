import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

MAX_PDF_PAGES = 500
MAX_TEXT_CHARACTERS = 2_000_000
MAX_CHUNKS = 10_000
MAX_EMBEDDING_DIMENSIONS = 4_096
CACHE_FORMAT_VERSION = 1
MAX_BOOK_NAME_LENGTH = 128
MAX_DOCUMENT_CACHE_BYTES = MAX_TEXT_CHARACTERS * 4 + 64 * 1024
MAX_EMBEDDINGS_CACHE_BYTES = 64 * 1024 * 1024

_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load one shared, read-only embedding model per configured model name."""
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
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
        model_name: str = "all-MiniLM-L6-v2",
        model: Any | None = None,
        storage_dir: str | Path = "embeddings",
    ):
        self.model_name = model_name
        self._model = model
        self.documents: list[str] = []
        self.embeddings: Any | None = None
        self.book_name: str | None = None
        self.storage_dir = Path(storage_dir)
        self.embeddings_file = self.storage_dir / "book_embeddings.npy"
        self.documents_file = self.storage_dir / "book_documents.json"

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _get_embedding_model(self.model_name)
        return self._model

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        try:
            with Path(pdf_path).open("rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                if page_count > MAX_PDF_PAGES:
                    logger.warning("PDF rejected: page limit exceeded")
                    return ""

                text_parts: list[str] = []
                character_count = 0
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    character_count += len(page_text)
                    if character_count > MAX_TEXT_CHARACTERS:
                        logger.warning("PDF rejected: extracted text limit exceeded")
                        return ""
                    text_parts.append(page_text)
                return "\n".join(text_parts)
        except Exception as exc:  # noqa: BLE001 - malformed PDFs are user input
            logger.error("PDF extraction failed: %s", type(exc).__name__)
            return ""

    def extract_text_from_file(self, file_path: str | Path) -> str:
        path = Path(file_path)
        try:
            if not path.is_file():
                return ""
            if path.suffix.lower() == ".pdf":
                return self.extract_text_from_pdf(path)
            if path.suffix.lower() != ".txt":
                return ""
            with path.open("r", encoding="utf-8", errors="replace") as file:
                text = file.read(MAX_TEXT_CHARACTERS + 1)
            if len(text) > MAX_TEXT_CHARACTERS:
                logger.warning("Text file rejected: extracted text limit exceeded")
                return ""
            return text
        except Exception as exc:  # noqa: BLE001 - local files are user input
            logger.error("Book extraction failed: %s", type(exc).__name__)
            return ""

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks = []
        words = text.split()
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def load_book(self, file_path: str | Path, book_name: str | None = None) -> bool:
        text = self.extract_text_from_file(file_path)
        if not text:
            logger.warning("Book contains no extractable text")
            return False

        try:
            documents = self.chunk_text(text)
            if not documents or len(documents) > MAX_CHUNKS:
                logger.warning("Book rejected: chunk limit exceeded or no chunks")
                return False

            embeddings = self._get_model().encode(documents, convert_to_tensor=True)
            self.documents = documents
            self.embeddings = embeddings
            self.book_name = book_name or Path(file_path).stem
            if not self.save_embeddings():
                self.documents = []
                self.embeddings = None
                self.book_name = None
                return False
            return True
        except Exception as exc:  # noqa: BLE001 - model and storage failures are recoverable
            logger.error("Book processing failed: %s", type(exc).__name__)
            self.documents = []
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
                        "book_name": self.book_name,
                        "embedding_sha256": self._embedding_digest(embedding_array),
                        "documents": self.documents,
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

            documents = payload.get("documents")
            book_name = payload.get("book_name")
            if not self._documents_are_valid(documents):
                return False
            if not self._book_name_is_valid(book_name):
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
                embedding_values = np.array(
                    embedding_array, dtype=np.float32, copy=True, order="C"
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
            self.embeddings = torch.as_tensor(embedding_values)
            self.book_name = book_name
            logger.info("Loaded cached embeddings")
            return True
        except Exception as exc:  # noqa: BLE001 - malformed local cache is untrusted
            logger.error("Could not load embeddings: %s", type(exc).__name__)
            return False

    def answer_question(
        self, question: str, top_k: int = 3, short: bool = True
    ) -> tuple[str, list[str]]:
        if not self.documents or self.embeddings is None:
            return "No book loaded", []
        if not isinstance(question, str) or not question.strip():
            return "Please ask a question", []

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
        relevant_chunks = [self.documents[int(index)] for index in top_results]

        if short:
            best_chunk = relevant_chunks[0]
            import re

            sentences = re.split(r"[.!?]+", best_chunk)
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

            if not sentences:
                return "No readable answer found", relevant_chunks

            short_answer = ""
            word_count = 0
            for sentence in sentences:
                words = sentence.split()
                if word_count + len(words) <= 30:
                    short_answer = f"{short_answer} {sentence}".strip()
                    word_count += len(words)
                else:
                    break

            if not short_answer:
                short_answer = " ".join(sentences[0].split()[:15]) + "..."

            answer = short_answer
        else:
            context = "\n\n".join(relevant_chunks)
            answer = f"Based on the book:\n\n{context}"

        return answer, relevant_chunks

    def get_book_summary(self) -> str:
        if not self.documents:
            return "No book loaded"

        return f"Book: {self.book_name}\nChunks: {len(self.documents)}"
