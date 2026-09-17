import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import book_qa
from book_qa import (
    MAX_DOCUMENT_CACHE_BYTES,
    MAX_EMBEDDINGS_CACHE_BYTES,
    MAX_TEXT_CHARACTERS,
    BookKnowledgeBase,
)


def _write_simple_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a minimal valid PDF with one Type1 text page per entry."""
    bodies: list[tuple[int, str]] = []
    page_numbers = []
    content_numbers = []
    number = 3
    for _ in page_texts:
        page_numbers.append(number)
        content_numbers.append(number + 1)
        number += 2
    font_number = number
    bodies.append((1, "<< /Type /Catalog /Pages 2 0 R >>"))
    kids = " ".join(f"{object_number} 0 R" for object_number in page_numbers)
    bodies.append(
        (2, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>")
    )
    for page_number, content_number, text in zip(
        page_numbers, content_numbers, page_texts
    ):
        bodies.append(
            (
                page_number,
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
                    f"/Contents {content_number} 0 R "
                    f"/Resources << /Font << /F1 {font_number} 0 R >> >> >>"
                ),
            )
        )
        stream = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET"
        bodies.append(
            (
                content_number,
                f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
            )
        )
    bodies.append(
        (font_number, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    )

    document = "%PDF-1.4\n"
    offsets: dict[int, int] = {}
    for object_number, body in bodies:
        offsets[object_number] = len(document)
        document += f"{object_number} 0 obj\n{body}\nendobj\n"
    xref_offset = len(document)
    document += f"xref\n0 {font_number + 1}\n"
    document += "0000000000 65535 f \n"
    for object_number in range(1, font_number + 1):
        document += f"{offsets[object_number]:010d} 00000 n \n"
    document += (
        f"trailer\n<< /Root 1 0 R /Size {font_number + 1} >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    path.write_bytes(document.encode("latin-1"))


class FakeEmbeddingModel:
    def encode(self, values, convert_to_tensor=True):
        del convert_to_tensor
        if isinstance(values, str):
            return torch.tensor([1.0, 0.0])
        return torch.tensor([[1.0, 0.0] for _ in values])


class BookKnowledgeBaseTests(unittest.TestCase):
    def test_uppercase_pdf_extension_uses_pdf_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.PDF"
            path.write_bytes(b"not a real pdf")
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
            knowledge_base.extract_text_from_pdf = lambda _: "extracted pdf text"

            self.assertEqual(
                knowledge_base.extract_text_from_file(path), "extracted pdf text"
            )

    def test_load_and_reload_use_safe_non_pickle_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")

            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )
            self.assertTrue(knowledge_base.load_book(source, "notes"))
            self.assertTrue(knowledge_base.embeddings_file.name.endswith(".npy"))
            self.assertTrue(knowledge_base.documents_file.name.endswith(".json"))

            restored = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )
            self.assertTrue(restored.load_embeddings())
            self.assertEqual(restored.get_book_summary(), "Book: notes\nChunks: 1")
            answer, chunks = restored.answer_question("alpha")
            self.assertIn("alpha", answer)
            self.assertEqual(len(chunks), 1)

    def test_cache_model_mismatch_does_not_replace_active_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")

            writer = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )
            self.assertTrue(writer.load_book(source, "notes"))

            restored = BookKnowledgeBase(
                model_name="different-model",
                model=FakeEmbeddingModel(),
                storage_dir=root / "cache",
            )
            restored.documents = ["existing active book"]
            restored.embeddings = torch.tensor([[1.0, 0.0]])
            restored.book_name = "existing"

            self.assertFalse(restored.load_embeddings())
            self.assertEqual(restored.documents, ["existing active book"])
            self.assertEqual(restored.book_name, "existing")

    def test_cache_pair_with_mismatched_document_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")
            cache = root / "cache"

            writer = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(writer.load_book(source, "notes"))
            payload = json.loads((cache / "book_documents.json").read_text())
            payload["documents"].append("another document")
            (cache / "book_documents.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertFalse(restored.load_embeddings())
            self.assertEqual(restored.documents, [])
            self.assertIsNone(restored.embeddings)

    def test_cache_digest_rejects_same_shape_but_different_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")
            cache = root / "cache"

            writer = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(writer.load_book(source, "notes"))
            payload = json.loads((cache / "book_documents.json").read_text())
            payload["embedding_sha256"] = "0" * 64
            (cache / "book_documents.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertFalse(restored.load_embeddings())

    def test_cache_rejects_values_that_overflow_during_float32_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            infinite_embedding = torch.tensor([[float("inf")]], dtype=torch.float32)
            (cache / "book_documents.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model_name": "all-MiniLM-L6-v2",
                        "book_name": "notes",
                        "embedding_sha256": hashlib.sha256(
                            infinite_embedding.numpy().tobytes()
                        ).hexdigest(),
                        "documents": ["document"],
                    }
                ),
                encoding="utf-8",
            )
            with (cache / "book_embeddings.npy").open("wb") as file:
                np.save(
                    file,
                    np.array([[np.finfo(np.float64).max]], dtype=np.float64),
                    allow_pickle=False,
                )

            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertFalse(restored.load_embeddings())
            self.assertEqual(restored.documents, [])
            self.assertIsNone(restored.embeddings)

    def test_oversized_cache_files_are_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            (cache / "book_documents.json").write_bytes(
                b"{" + b"x" * MAX_DOCUMENT_CACHE_BYTES + b"}"
            )
            (cache / "book_embeddings.npy").write_bytes(b"not loaded")
            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertFalse(restored.load_embeddings())

            (cache / "book_documents.json").write_text("{}", encoding="utf-8")
            with (cache / "book_embeddings.npy").open("wb") as file:
                file.truncate(MAX_EMBEDDINGS_CACHE_BYTES + 1)
            self.assertFalse(restored.load_embeddings())

    def test_punctuation_only_chunk_has_a_bounded_answer(self):
        knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
        knowledge_base.documents = ["!!!"]
        knowledge_base.embeddings = torch.tensor([[1.0, 0.0]])

        answer, chunks = knowledge_base.answer_question("meaning")

        self.assertEqual(answer, "No readable answer found")
        self.assertEqual(chunks, ["!!!"])

    def test_corrupt_cache_is_rejected_without_pickle_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            (cache / "book_documents.json").write_text(
                json.dumps(["document"]), encoding="utf-8"
            )
            (cache / "book_embeddings.npy").write_bytes(b"not a numpy file")

            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=cache
            )
            self.assertFalse(knowledge_base.load_embeddings())
            self.assertEqual(knowledge_base.documents, [])
            self.assertIsNone(knowledge_base.embeddings)

    def test_text_extraction_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            path.write_text("x" * (MAX_TEXT_CHARACTERS + 1), encoding="utf-8")
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())

            self.assertEqual(knowledge_base.extract_text_from_file(path), "")

    def test_chunk_parameters_must_make_progress(self):
        knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
        with self.assertRaises(ValueError):
            knowledge_base.chunk_text("one two", chunk_size=10, overlap=10)

    def test_real_pdf_text_is_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.pdf"
            _write_simple_pdf(path, ["Hello PDF world"])
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())

            self.assertIn("Hello PDF world", knowledge_base.extract_text_from_pdf(path))

    def test_real_pdf_page_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.pdf"
            _write_simple_pdf(path, ["First page", "Second page"])
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())

            with patch.object(book_qa, "MAX_PDF_PAGES", 1):
                self.assertEqual(knowledge_base.extract_text_from_pdf(path), "")

    def test_malformed_pdf_returns_no_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.pdf"
            path.write_bytes(b"not a real pdf")
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())

            self.assertEqual(knowledge_base.extract_text_from_pdf(path), "")

    def test_load_book_persists_working_embeddings_from_a_real_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.pdf"
            _write_simple_pdf(source, ["Alpha beta gamma delta"])
            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )

            self.assertTrue(knowledge_base.load_book(source, "pdf book"))
            restored = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )
            self.assertTrue(restored.load_embeddings())
            self.assertIn("Alpha beta gamma", restored.documents[0])

    def test_only_the_default_model_uses_the_pinned_revision(self):
        calls: list[tuple[str, str | None, bool]] = []

        class RecordingModel:
            def __init__(self, model_name, *, revision=None, trust_remote_code=None):
                calls.append((model_name, revision, trust_remote_code))

        with (
            patch.object(book_qa, "SentenceTransformer", RecordingModel),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("EMBEDDING_MODEL_REVISION", None)
            book_qa._MODEL_CACHE.clear()
            self.addCleanup(book_qa._MODEL_CACHE.clear)
            book_qa._get_embedding_model(book_qa.DEFAULT_EMBEDDING_MODEL)
            book_qa._get_embedding_model("another-model")

        self.assertEqual(
            calls[0],
            (
                book_qa.DEFAULT_EMBEDDING_MODEL,
                book_qa.DEFAULT_EMBEDDING_MODEL_REVISION,
                False,
            ),
        )
        self.assertEqual(calls[1], ("another-model", None, False))

    def test_embedding_revision_can_be_overridden_from_the_environment(self):
        calls: list[tuple[str, str | None, bool]] = []

        class RecordingModel:
            def __init__(self, model_name, *, revision=None, trust_remote_code=None):
                calls.append((model_name, revision, trust_remote_code))

        with (
            patch.object(book_qa, "SentenceTransformer", RecordingModel),
            patch.dict(os.environ, {"EMBEDDING_MODEL_REVISION": "custom-sha"}),
        ):
            book_qa._MODEL_CACHE.clear()
            self.addCleanup(book_qa._MODEL_CACHE.clear)
            book_qa._get_embedding_model(book_qa.DEFAULT_EMBEDDING_MODEL)

        self.assertEqual(calls[0][1], "custom-sha")

    def test_embedding_revision_can_be_disabled_with_an_empty_override(self):
        calls: list[tuple[str, str | None, bool]] = []

        class RecordingModel:
            def __init__(self, model_name, *, revision=None, trust_remote_code=None):
                calls.append((model_name, revision, trust_remote_code))

        with (
            patch.object(book_qa, "SentenceTransformer", RecordingModel),
            patch.dict(os.environ, {"EMBEDDING_MODEL_REVISION": "   "}),
        ):
            book_qa._MODEL_CACHE.clear()
            self.addCleanup(book_qa._MODEL_CACHE.clear)
            book_qa._get_embedding_model(book_qa.DEFAULT_EMBEDDING_MODEL)

        self.assertIsNone(calls[0][1])


if __name__ == "__main__":
    unittest.main()
