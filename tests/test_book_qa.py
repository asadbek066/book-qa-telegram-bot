import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from book_qa import (
    MAX_DOCUMENT_CACHE_BYTES,
    MAX_EMBEDDINGS_CACHE_BYTES,
    MAX_TEXT_CHARACTERS,
    BookKnowledgeBase,
)


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


if __name__ == "__main__":
    unittest.main()
