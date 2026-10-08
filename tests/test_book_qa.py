import hashlib
import json
import os
import re
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import book_qa
from book_qa import (
    MAX_DOCUMENT_CACHE_BYTES,
    MAX_EMBEDDINGS_CACHE_BYTES,
    MAX_PDF_FILE_BYTES,
    MAX_PDF_STREAM_OUTPUT_BYTES,
    MAX_PDF_XFORM_INVOCATIONS,
    MAX_TEXT_CHARACTERS,
    BookKnowledgeBase,
)


def _write_simple_pdf(
    path: Path, page_texts: list[str], *, compress_streams: bool = False
) -> None:
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
        stream_data = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("latin-1")
        filter_entry = ""
        if compress_streams:
            stream_data = zlib.compress(stream_data)
            filter_entry = " /Filter /FlateDecode"
        stream = stream_data.decode("latin-1")
        bodies.append(
            (
                content_number,
                (
                    f"<< /Length {len(stream_data)}{filter_entry} >>\n"
                    f"stream\n{stream}\nendstream"
                ),
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


class HashingEmbeddingModel:
    """Deterministic bag-of-words embedder: no randomness, no network.

    Each lower-cased word is hashed (md5, not Python's salted hash) into one of
    ``dimensions`` buckets; the count vector is L2-normalised, so the cosine
    similarity of two texts reflects their shared vocabulary.
    """

    def __init__(self, dimensions=512):
        self.dimensions = dimensions
        self.encode_calls = 0

    def _vector(self, text):
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def encode(self, values, convert_to_tensor=True):
        del convert_to_tensor
        self.encode_calls += 1
        if isinstance(values, str):
            return torch.tensor(self._vector(values))
        return torch.tensor([self._vector(value) for value in values])


BOOK_CHUNKS = [
    "The lighthouse keeper polished the great brass lamp every evening before "
    "the storm rolled in from the northern sea.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose "
    "inside the chloroplasts of green plants.",
    "The treaty of Westphalia ended the thirty years war and established the "
    "principle of state sovereignty in Europe.",
    "To bake sourdough bread, feed the starter, mix flour with water and salt, "
    "then let the dough ferment overnight.",
]


def _knowledge_base(model=None, chunks=None):
    model = model or HashingEmbeddingModel()
    knowledge_base = BookKnowledgeBase(model=model)
    knowledge_base.documents = list(chunks or BOOK_CHUNKS)
    knowledge_base.embeddings = model.encode(knowledge_base.documents)
    return knowledge_base


class FakeTokenizer:
    """Splits every word into 3-character pieces, like a tiny word-piece model."""

    def tokenize(self, text):
        return [text[i : i + 3] for i in range(0, len(text), 3)]

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 2


class LimitedEmbeddingModel(FakeEmbeddingModel):
    def __init__(self, max_seq_length=42):
        self.max_seq_length = max_seq_length
        self.tokenizer = FakeTokenizer()


class ChunkSizingTests(unittest.TestCase):
    @staticmethod
    def words(count):
        return [f"w{index:04d}" for index in range(count)]  # 5 chars, 2 tokens

    def test_chunks_never_exceed_the_model_limit(self):
        model = LimitedEmbeddingModel(max_seq_length=42)
        knowledge_base = BookKnowledgeBase(model=model)
        text = " ".join(self.words(300)) + " " + "x" * 500

        chunks = knowledge_base.chunk_text(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            tokens = sum(len(model.tokenizer.tokenize(w)) for w in chunk.split())
            self.assertLessEqual(tokens + 2, model.max_seq_length)

    def test_every_source_word_is_in_some_chunk(self):
        model = LimitedEmbeddingModel()
        knowledge_base = BookKnowledgeBase(model=model)
        source = self.words(250)

        chunks = knowledge_base.chunk_text(" ".join(source))

        covered = {word for chunk in chunks for word in chunk.split()}
        self.assertEqual(covered, set(source))

    def test_overlap_follows_the_configured_ratio(self):
        model = LimitedEmbeddingModel(max_seq_length=42)  # 40 usable, 4 overlap
        knowledge_base = BookKnowledgeBase(model=model)
        self.assertEqual(
            knowledge_base.chunking_parameters(),
            {"unit": "tokens", "chunk_size": 40, "overlap": 4},
        )

        chunks = knowledge_base.chunk_text(" ".join(self.words(120)))

        for previous, following in zip(chunks, chunks[1:]):
            # 2 words x 2 tokens = 4 shared tokens.
            self.assertEqual(previous.split()[-2:], following.split()[:2])

    def test_over_long_word_is_split_to_fit(self):
        model = LimitedEmbeddingModel(max_seq_length=42)
        knowledge_base = BookKnowledgeBase(model=model)

        chunks = knowledge_base.chunk_text("start " + "y" * 400 + " end")

        for chunk in chunks:
            tokens = sum(len(model.tokenizer.tokenize(w)) for w in chunk.split())
            self.assertLessEqual(tokens, 40)
        self.assertEqual("".join(chunks).count("y"), 400)

    def test_word_fallback_when_model_does_not_expose_a_limit(self):
        knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
        self.assertEqual(
            knowledge_base.chunking_parameters(),
            {
                "unit": "words",
                "chunk_size": book_qa.FALLBACK_CHUNK_WORDS,
                "overlap": int(
                    book_qa.FALLBACK_CHUNK_WORDS * book_qa.CHUNK_OVERLAP_RATIO
                ),
            },
        )
        chunks = knowledge_base.chunk_text(" ".join(self.words(400)))
        self.assertTrue(
            all(len(c.split()) <= book_qa.FALLBACK_CHUNK_WORDS for c in chunks)
        )

    def test_cache_written_with_other_chunking_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text(" ".join(self.words(100)), encoding="utf-8")
            cache = root / "cache"

            writer = BookKnowledgeBase(model=LimitedEmbeddingModel(42), storage_dir=cache)
            self.assertTrue(writer.load_book(source, "notes"))
            payload = json.loads((cache / "book_documents.json").read_text())
            self.assertEqual(payload["chunking"]["chunk_size"], 40)

            same = BookKnowledgeBase(model=LimitedEmbeddingModel(42), storage_dir=cache)
            self.assertTrue(same.load_embeddings())

            other_limit = BookKnowledgeBase(
                model=LimitedEmbeddingModel(130), storage_dir=cache
            )
            self.assertFalse(other_limit.load_embeddings())
            word_based = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertFalse(word_based.load_embeddings())

    def test_cache_from_the_previous_format_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")
            cache = root / "cache"
            writer = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(writer.load_book(source, "notes"))
            payload = json.loads((cache / "book_documents.json").read_text())
            payload["version"] = 2
            payload.pop("chunking")
            (cache / "book_documents.json").write_text(json.dumps(payload))

            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertFalse(restored.load_embeddings())


class RetrievalRankingTests(unittest.TestCase):
    QUESTIONS = [
        ("Who polished the brass lamp in the lighthouse?", 0),
        ("How do plants turn sunlight and water into glucose?", 1),
        ("Which treaty ended the thirty years war?", 2),
        ("How long should the sourdough dough ferment?", 3),
    ]

    def test_chunk_with_the_answer_ranks_first(self):
        knowledge_base = _knowledge_base()
        for question, expected in self.QUESTIONS:
            with self.subTest(question=question):
                _, chunks = knowledge_base.answer_question(question, top_k=4)
                self.assertEqual(chunks[0], BOOK_CHUNKS[expected])

    def test_results_are_ordered_by_descending_score(self):
        model = HashingEmbeddingModel()
        knowledge_base = _knowledge_base(model)
        question = "Which treaty ended the thirty years war in Europe?"
        question_vector = model.encode(question)

        _, chunks = knowledge_base.answer_question(question, top_k=4)

        scores = [
            float(torch.dot(question_vector, model.encode(chunk))) for chunk in chunks
        ]
        self.assertEqual(len(chunks), 4)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], scores[1])

    def test_source_excerpts_are_the_returned_chunks(self):
        knowledge_base = _knowledge_base()
        _, chunks = knowledge_base.answer_question(
            "What does the lighthouse keeper do?", top_k=2
        )

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk in BOOK_CHUNKS for chunk in chunks))
        self.assertEqual(len(set(chunks)), 2)
        self.assertEqual(chunks[0], BOOK_CHUNKS[0])

    def test_short_answer_comes_from_the_best_chunk(self):
        knowledge_base = _knowledge_base()
        answer, chunks = knowledge_base.answer_question(
            "What is photosynthesis in green plants?"
        )
        self.assertIn("Photosynthesis", answer)
        self.assertEqual(chunks[0], BOOK_CHUNKS[1])


class NotFoundAndSentenceTests(unittest.TestCase):
    OFF_TOPIC = "Describe quantum entanglement experiments"

    def test_off_topic_question_abstains_without_sources(self):
        knowledge_base = _knowledge_base()

        answer, chunks = knowledge_base.answer_question(self.OFF_TOPIC, top_k=3)

        self.assertEqual(answer, book_qa.NOT_FOUND_MESSAGE)
        self.assertEqual(chunks, [])
        _, top_score, _ = knowledge_base.retrieve(self.OFF_TOPIC)
        self.assertLess(top_score, book_qa.DEFAULT_MIN_SIMILARITY_SCORE)

    def test_off_topic_long_answer_mode_also_abstains(self):
        answer, chunks = _knowledge_base().answer_question(
            self.OFF_TOPIC, top_k=3, short=False
        )
        self.assertEqual((answer, chunks), (book_qa.NOT_FOUND_MESSAGE, []))

    def test_on_topic_question_is_answered(self):
        answer, chunks = _knowledge_base().answer_question(
            "Which treaty ended the thirty years war?"
        )
        self.assertNotEqual(answer, book_qa.NOT_FOUND_MESSAGE)
        self.assertIn("Westphalia", answer)
        self.assertEqual(chunks[0], BOOK_CHUNKS[2])

    def test_threshold_boundary_answers_at_equal_and_abstains_above(self):
        knowledge_base = _knowledge_base()
        question = "Which treaty ended the thirty years war?"
        _, top_score, _ = knowledge_base.retrieve(question)

        with patch.dict(os.environ, {"MIN_SIMILARITY_SCORE": repr(top_score)}):
            _, chunks = knowledge_base.answer_question(question)
            self.assertTrue(chunks)
        with patch.dict(os.environ, {"MIN_SIMILARITY_SCORE": repr(top_score + 1e-6)}):
            answer, chunks = knowledge_base.answer_question(question)
            self.assertEqual((answer, chunks), (book_qa.NOT_FOUND_MESSAGE, []))

    def test_threshold_setting_defaults_and_rejects_bad_values(self):
        for raw in ("", "abc", "nan", "inf"):
            with patch.dict(os.environ, {"MIN_SIMILARITY_SCORE": raw}):
                self.assertEqual(
                    book_qa._min_similarity_score(),
                    book_qa.DEFAULT_MIN_SIMILARITY_SCORE,
                )
        with patch.dict(os.environ, {"MIN_SIMILARITY_SCORE": "0.4"}):
            self.assertEqual(book_qa._min_similarity_score(), 0.4)
        with patch.dict(os.environ, {"MIN_SIMILARITY_SCORE": "7"}):
            self.assertEqual(book_qa._min_similarity_score(), 1.0)

    def test_top_score_is_logged_without_the_question_text(self):
        knowledge_base = _knowledge_base()
        with self.assertLogs("book_qa", level="DEBUG") as logs:
            knowledge_base.answer_question(self.OFF_TOPIC)
        output = "\n".join(logs.output)
        self.assertIn("Top retrieval similarity", output)
        self.assertNotIn("entanglement", output)

    def test_best_matching_sentence_is_returned_not_the_first(self):
        chunk = (
            "The village sat beside a quiet river. Farmers grew barley and oats "
            "there. The old bridge was rebuilt in stone after the great flood. "
            "Children often fished near the mill."
        )
        model = HashingEmbeddingModel()
        knowledge_base = _knowledge_base(model, [chunk, BOOK_CHUNKS[1]])

        answer, chunks = knowledge_base.answer_question(
            "When was the old bridge rebuilt after the flood?"
        )

        self.assertTrue(answer.startswith("The old bridge was rebuilt in stone"))
        self.assertNotIn("Farmers", answer)
        self.assertEqual(chunks[0], chunk)

    def test_encode_calls_per_question_are_bounded(self):
        model = HashingEmbeddingModel()
        sentences = ". ".join(f"Sentence number {n} about topic{n}" for n in range(200))
        knowledge_base = _knowledge_base(model, [sentences + "."])
        model.encode_calls = 0

        knowledge_base.answer_question("sentence number 7 about topic7")

        self.assertLessEqual(model.encode_calls, 2)


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

    def test_pdf_extraction_uses_bounded_stream_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.pdf"
            _write_simple_pdf(path, ["Hello PDF world"])
            original_apply = book_qa.pypdf.apply_configuration

            with patch.object(
                book_qa.pypdf,
                "apply_configuration",
                wraps=original_apply,
            ) as apply_configuration:
                knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
                self.assertIn(
                    "Hello PDF world", knowledge_base.extract_text_from_pdf(path)
                )

            limits = apply_configuration.call_args.kwargs
            self.assertEqual(
                limits["zlib_maximum_output_length"], MAX_PDF_STREAM_OUTPUT_BYTES
            )
            self.assertEqual(
                limits["image_maximum_buffer_size"], MAX_PDF_STREAM_OUTPUT_BYTES
            )
            self.assertEqual(
                limits["xform_maximum_invocations_per_extraction"],
                MAX_PDF_XFORM_INVOCATIONS,
            )

    def test_pdf_stream_expansion_over_the_configured_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.pdf"
            _write_simple_pdf(
                path,
                ["A" * (MAX_PDF_STREAM_OUTPUT_BYTES + 1)],
                compress_streams=True,
            )
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())

            self.assertEqual(knowledge_base.extract_text_from_pdf(path), "")

    def test_oversized_pdf_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.pdf"
            with path.open("wb") as file:
                file.truncate(MAX_PDF_FILE_BYTES + 1)
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())

            with patch.object(book_qa.pypdf, "PdfReader") as pdf_reader:
                self.assertEqual(knowledge_base.extract_text_from_pdf(path), "")

            pdf_reader.assert_not_called()

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

    def test_cache_model_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")

            with patch.dict(os.environ, {"EMBEDDING_MODEL_REVISION": "revision-one"}):
                writer = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=root / "cache"
                )
                self.assertTrue(writer.load_book(source, "notes"))
                payload = json.loads(writer.documents_file.read_text(encoding="utf-8"))
                self.assertEqual(payload["model_revision"], "revision-one")

            with patch.dict(os.environ, {"EMBEDDING_MODEL_REVISION": "revision-two"}):
                restored = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=root / "cache"
                )
                self.assertFalse(restored.load_embeddings())

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
