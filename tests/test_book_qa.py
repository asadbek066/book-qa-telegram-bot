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
    bodies.append((2, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>"))
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


class CountingTokenizer:
    """Batch-callable tokenizer that counts calls; 3-character word pieces."""

    def __init__(self, collapse=False):
        self.calls = 0
        self.tokenize_calls = 0
        self.collapse = collapse

    def tokenize(self, text):
        self.tokenize_calls += 1
        return [text[i : i + 3] for i in range(0, len(text), 3)]

    def __call__(self, texts, add_special_tokens=False):
        del add_special_tokens
        self.calls += 1
        if self.collapse:  # like an unknown-token blob: one token per text
            return {"input_ids": [[0] for _ in texts]}
        return {"input_ids": [[0] * -(-len(t) // 3) for t in texts]}

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 2


class LimitedEmbeddingModel(FakeEmbeddingModel):
    def __init__(self, max_seq_length=42, tokenizer=None):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer or FakeTokenizer()


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

            writer = BookKnowledgeBase(
                model=LimitedEmbeddingModel(42), storage_dir=cache
            )
            self.assertTrue(writer.load_book(source, "notes"))
            payload = json.loads((cache / "book_documents.json").read_text())
            self.assertEqual(payload["chunking"]["chunk_size"], 40)

            same = BookKnowledgeBase(model=LimitedEmbeddingModel(42), storage_dir=cache)
            self.assertTrue(same.load_embeddings())

            other_limit = BookKnowledgeBase(
                model=LimitedEmbeddingModel(130), storage_dir=cache
            )
            self.assertFalse(other_limit.load_embeddings())
            word_based = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=cache
            )
            self.assertFalse(word_based.load_embeddings())

    def test_tokenizer_is_called_per_batch_not_per_word(self):
        tokenizer = CountingTokenizer()
        model = LimitedEmbeddingModel(42, tokenizer)
        knowledge_base = BookKnowledgeBase(model=model)
        words = self.words(20_000)

        chunks = knowledge_base.chunk_text(" ".join(words))

        self.assertEqual(tokenizer.tokenize_calls, 0)
        self.assertLessEqual(
            tokenizer.calls, len(words) // book_qa.TOKENIZE_BATCH_WORDS + 1
        )
        for chunk in chunks:
            self.assertLessEqual(sum(-(-len(w) // 3) for w in chunk.split()), 40)

    def test_blob_the_tokenizer_collapses_to_one_token_stays_bounded(self):
        tokenizer = CountingTokenizer(collapse=True)
        model = LimitedEmbeddingModel(42, tokenizer)
        knowledge_base = BookKnowledgeBase(model=model)
        blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo" * 150  # ~5 KB, no spaces

        chunks = knowledge_base.chunk_text(f"intro {blob} outro")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(
                all(len(w) <= book_qa.MAX_PIECE_CHARACTERS for w in chunk.split())
            )
            self.assertLessEqual(len(chunk), 40 * (book_qa.MAX_PIECE_CHARACTERS + 1))
        self.assertGreaterEqual(
            "".join(w for c in chunks for w in c.split()).count("QUJD"), 150
        )

    def test_restore_does_not_load_the_model_and_defers_the_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text(" ".join(self.words(100)), encoding="utf-8")
            cache = root / "cache"
            writer = BookKnowledgeBase(
                model=LimitedEmbeddingModel(42), storage_dir=cache
            )
            self.assertTrue(writer.load_book(source, "notes"))

            def never_load(name):
                raise AssertionError("restore must not load the model")

            book_qa._MODEL_CACHE.clear()
            with patch.object(book_qa, "_get_embedding_model", never_load):
                restored = BookKnowledgeBase(storage_dir=cache)
                self.assertTrue(restored.load_embeddings())
            self.assertTrue(restored.documents)

            # First question loads the model; the same parameters pass.
            restored._model = LimitedEmbeddingModel(42)
            answer, chunks = restored.answer_question("w0001")
            self.assertNotEqual(answer, book_qa.STALE_CACHE_MESSAGE)
            self.assertTrue(chunks)

    def test_stale_parameters_found_on_first_question_invalidate_the_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text(" ".join(self.words(100)), encoding="utf-8")
            cache = root / "cache"
            writer = BookKnowledgeBase(
                model=LimitedEmbeddingModel(42), storage_dir=cache
            )
            self.assertTrue(writer.load_book(source, "notes"))

            book_qa._MODEL_CACHE.clear()
            restored = BookKnowledgeBase(storage_dir=cache)
            self.assertTrue(restored.load_embeddings())
            restored._model = LimitedEmbeddingModel(130)  # different limit

            answer, chunks = restored.answer_question("w0001")

            self.assertEqual((answer, chunks), (book_qa.STALE_CACHE_MESSAGE, []))
            self.assertEqual(restored.documents, [])
            self.assertEqual(restored.answer_question("w0001")[0], "No book loaded")

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

    def setUp(self):
        book_qa._WARNED_SETTINGS.clear()
        self.addCleanup(book_qa._WARNED_SETTINGS.clear)

    @staticmethod
    def enabled(value="0.15"):
        return patch.dict(os.environ, {"MIN_SIMILARITY_SCORE": value})

    def test_default_threshold_does_not_abstain_on_a_weak_positive_score(self):
        self.assertEqual(book_qa.DEFAULT_MIN_SIMILARITY_SCORE, 0.0)
        knowledge_base = _knowledge_base()
        weak = "treaty alpha beta gamma delta epsilon zeta eta theta iota"
        with patch.dict(os.environ):
            os.environ.pop("MIN_SIMILARITY_SCORE", None)
            _, top_score, _ = knowledge_base.retrieve(weak)
            self.assertTrue(0.0 < top_score < 0.15)

            answer, chunks = knowledge_base.answer_question(weak)

        self.assertNotEqual(answer, book_qa.NOT_FOUND_MESSAGE)
        self.assertTrue(chunks)

    def test_default_threshold_abstains_on_a_negative_score(self):
        knowledge_base = _knowledge_base()
        negative = (["x"], -0.2, torch.tensor([1.0]))
        with patch.object(knowledge_base, "retrieve", return_value=negative):
            answer, chunks = knowledge_base.answer_question("anything")
        self.assertEqual((answer, chunks), (book_qa.NOT_FOUND_MESSAGE, []))

    def test_nan_score_abstains(self):
        knowledge_base = _knowledge_base()
        nan = (["x"], float("nan"), torch.tensor([1.0]))
        for setting in ("0.0", "-1"):
            with (
                self.enabled(setting),
                patch.object(knowledge_base, "retrieve", return_value=nan),
            ):
                answer, chunks = knowledge_base.answer_question("anything")
            self.assertEqual((answer, chunks), (book_qa.NOT_FOUND_MESSAGE, []))

    def test_explicit_threshold_abstains_off_topic_without_sources(self):
        knowledge_base = _knowledge_base()
        with self.enabled():
            answer, chunks = knowledge_base.answer_question(self.OFF_TOPIC, top_k=3)
        self.assertEqual(answer, book_qa.NOT_FOUND_MESSAGE)
        self.assertEqual(chunks, [])
        _, top_score, _ = knowledge_base.retrieve(self.OFF_TOPIC)
        self.assertLess(top_score, 0.15)

    def test_off_topic_long_answer_mode_also_abstains(self):
        with self.enabled():
            answer, chunks = _knowledge_base().answer_question(
                self.OFF_TOPIC, top_k=3, short=False
            )
        self.assertEqual((answer, chunks), (book_qa.NOT_FOUND_MESSAGE, []))

    def test_invalid_setting_is_logged_once_per_process(self):
        with self.enabled("abc"), self.assertLogs("book_qa", level="WARNING") as logs:
            for _ in range(3):
                book_qa._min_similarity_score()
        self.assertEqual(len(logs.output), 1)

    def test_setting_of_one_or_more_warns_that_everything_abstains(self):
        with self.enabled("1.5"), self.assertLogs("book_qa", level="WARNING") as logs:
            self.assertEqual(book_qa._min_similarity_score(), 1.0)
            book_qa._min_similarity_score()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("every question", logs.output[0])

    def test_on_topic_question_is_answered(self):
        with self.enabled():
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


def _numbered_pages(page_count, words_per_page, prefix="p"):
    return [
        " ".join(f"{prefix}{page}x{n:03d}" for n in range(words_per_page))
        for page in range(1, page_count + 1)
    ]


class PageCitationTests(unittest.TestCase):
    def test_chunk_pages_map_chunks_and_overlap_to_page_ranges(self):
        knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
        pages = _numbered_pages(3, 5)  # words 0-4, 5-9, 10-14

        chunks, ranges = knowledge_base.chunk_pages(pages, chunk_size=6, overlap=2)

        self.assertEqual(len(chunks), len(ranges))
        # Chunk 1 is words 0-5 (spans the page 1/2 break), chunk 2 words 4-9
        # (overlaps chunk 1 by two words), chunk 3 words 8-13, chunk 4 12-14.
        self.assertEqual(ranges, [(1, 2), (1, 2), (2, 3), (3, 3)])
        self.assertIn("p1x004", chunks[0].split())
        self.assertIn("p1x004", chunks[1].split())
        self.assertEqual(chunks, knowledge_base.chunk_text("\n".join(pages), 6, 2))

    def test_token_unit_chunks_keep_page_ranges_for_split_words(self):
        model = LimitedEmbeddingModel(max_seq_length=42)
        knowledge_base = BookKnowledgeBase(model=model)
        pages = ["a" * 250 + " first", "", "second third"]

        chunks, ranges = knowledge_base.chunk_pages(pages)

        self.assertEqual(len(chunks), len(ranges))
        self.assertEqual(chunks, knowledge_base.chunk_text("\n".join(pages)))
        self.assertEqual(ranges[0][0], 1)
        self.assertEqual(ranges[-1][1], 3)  # empty page 2 still counts as a position
        for first, last in ranges:
            self.assertLessEqual(first, last)

    def test_pdf_chunks_carry_one_based_pdf_page_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "book.pdf"
            _write_simple_pdf(pdf, _numbered_pages(3, 100))
            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )

            self.assertTrue(knowledge_base.load_book(pdf, "book"))

            # 300 words, 150-word chunks, 15 words overlap.
            self.assertEqual(knowledge_base.pages, [(1, 2), (2, 3), (3, 3)])
            _, chunks = knowledge_base.answer_question("anything", top_k=3)
            self.assertEqual(
                [chunk.pages for chunk in chunks], [(1, 2), (2, 3), (3, 3)]
            )

    def test_txt_book_has_no_page_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text(" ".join(_numbered_pages(1, 400)), encoding="utf-8")
            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )

            self.assertTrue(knowledge_base.load_book(source, "notes"))

            self.assertIsNone(knowledge_base.pages)
            _, chunks = knowledge_base.answer_question("anything", top_k=2)
            self.assertTrue(chunks)
            for chunk in chunks:
                self.assertIsNone(getattr(chunk, "pages", None))

    def test_cache_round_trip_preserves_pages_in_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "book.pdf"
            _write_simple_pdf(pdf, _numbered_pages(3, 100))
            cache = root / "cache"
            writer = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(writer.load_book(pdf, "book"))

            payload = json.loads((cache / "book_documents.json").read_text())
            self.assertEqual(payload["pages"], [[1, 2], [2, 3], [3, 3]])

            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(restored.load_embeddings())
            self.assertEqual(restored.pages, [(1, 2), (2, 3), (3, 3)])
            _, chunks = restored.answer_question("anything", top_k=1)
            self.assertEqual(chunks[0].pages, (1, 2))

    def test_old_cache_without_page_data_still_loads_without_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")
            cache = root / "cache"
            writer = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(writer.load_book(source, "notes"))
            payload = json.loads((cache / "book_documents.json").read_text())
            payload.pop("pages", None)
            (cache / "book_documents.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            restored = BookKnowledgeBase(model=FakeEmbeddingModel(), storage_dir=cache)
            self.assertTrue(restored.load_embeddings())
            self.assertIsNone(restored.pages)
            answer, chunks = restored.answer_question("alpha")
            self.assertEqual(chunks, ["alpha beta gamma"])
            self.assertIsNone(getattr(chunks[0], "pages", None))

    def test_malformed_page_data_is_rejected_not_crashed_on(self):
        for bad in ([[1, 2]] * 5, "x", [[0, 1]], [[3, 2]], [[True, 1]], [[1]]):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "notes.txt"
                source.write_text("alpha beta gamma", encoding="utf-8")
                cache = root / "cache"
                writer = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=cache
                )
                self.assertTrue(writer.load_book(source, "notes"))
                payload = json.loads((cache / "book_documents.json").read_text())
                payload["pages"] = bad
                (cache / "book_documents.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                restored = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=cache
                )
                self.assertFalse(restored.load_embeddings())

    def test_pdf_limits_still_apply_when_collecting_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "big.pdf"
            _write_simple_pdf(pdf, ["word"] * 2)
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
            with patch.object(book_qa, "MAX_TEXT_CHARACTERS", 5):
                self.assertEqual(knowledge_base.extract_text_from_pdf(pdf), "")
                self.assertIsNone(knowledge_base.extract_pdf_pages(pdf))

    def test_token_path_page_range_starts_at_the_first_word_of_the_chunk(self):
        model = LimitedEmbeddingModel(max_seq_length=12, tokenizer=CountingTokenizer())
        knowledge_base = BookKnowledgeBase(model=model)  # budget 10, overlap 1
        words = [f"{chr(97 + i // 7)}{i % 7}{chr(120 + i // 7)}" for i in range(21)]
        pages = [" ".join(words[i : i + 7]) for i in (0, 7, 14)]  # 1 token per word

        chunks, ranges = knowledge_base.chunk_pages(pages)

        # Chunks are words 0-9, 9-18 (one shared word) and 18-20.
        self.assertEqual(
            chunks, [" ".join(words[a:b]) for a, b in ((0, 10), (9, 19), (18, 21))]
        )
        self.assertEqual(ranges, [(1, 2), (2, 3), (3, 3)])

    def test_pieces_of_a_long_word_keep_the_page_of_that_word(self):
        model = LimitedEmbeddingModel(max_seq_length=80, tokenizer=CountingTokenizer())
        knowledge_base = BookKnowledgeBase(model=model)  # budget 78
        long_one, long_two = "x" * 250, "y" * 250  # three 100/100/50-char pieces
        pages = [f"aaa {long_one}", f"{long_two} bbb"]

        chunks, ranges = knowledge_base.chunk_pages(pages)

        self.assertEqual(
            chunks,
            [
                f"aaa {'x' * 100} {'x' * 100}",
                f"{'x' * 50} {'y' * 100}",
                f"{'y' * 100} {'y' * 50} bbb",
            ],
        )
        self.assertEqual(ranges, [(1, 1), (1, 2), (2, 2)])

    def test_oversize_pieces_split_to_the_budget_keep_the_page_of_their_word(self):
        model = LimitedEmbeddingModel(max_seq_length=12, tokenizer=CountingTokenizer())
        knowledge_base = BookKnowledgeBase(model=model)  # budget 10, 50 chars = 17
        pages = [f"aaa {'x' * 50}", f"{'y' * 50} bbb"]

        chunks, ranges = knowledge_base.chunk_pages(pages)

        self.assertEqual(
            chunks,
            [
                f"aaa {'x' * 25}",
                "x" * 25,
                "y" * 25,
                f"{'y' * 25} bbb",
            ],
        )
        self.assertEqual(ranges, [(1, 1), (1, 1), (2, 2), (2, 2)])

    def test_page_numbers_beyond_the_page_limit_are_rejected_in_the_cache(self):
        for bad in ([[1, 501]], [[1, 10**30]], [[501, 501]]):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "notes.txt"
                source.write_text("alpha beta gamma", encoding="utf-8")
                cache = root / "cache"
                writer = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=cache
                )
                self.assertTrue(writer.load_book(source, "notes"))
                payload = json.loads((cache / "book_documents.json").read_text())
                payload["pages"] = bad
                (cache / "book_documents.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                restored = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=cache
                )
                self.assertFalse(restored.load_embeddings())
                self.assertIsNone(restored.pages)

    def _loaded_pdf_knowledge_base(self, root, **kwargs):
        pdf = root / "book.pdf"
        _write_simple_pdf(pdf, _numbered_pages(3, 100))
        knowledge_base = BookKnowledgeBase(
            model=kwargs.pop("model", FakeEmbeddingModel()),
            storage_dir=root / "cache",
        )
        self.assertTrue(knowledge_base.load_book(pdf, "book"))
        self.assertIsNotNone(knowledge_base.pages)
        return knowledge_base

    def test_loading_a_txt_after_a_pdf_leaves_no_page_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_base = self._loaded_pdf_knowledge_base(root)
            source = root / "notes.txt"
            source.write_text("alpha beta gamma", encoding="utf-8")

            self.assertTrue(knowledge_base.load_book(source, "notes"))

            self.assertIsNone(knowledge_base.pages)
            _, chunks = knowledge_base.answer_question("alpha")
            self.assertIsNone(getattr(chunks[0], "pages", None))

    def test_failed_loads_clear_page_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_base = self._loaded_pdf_knowledge_base(root)
            other = root / "other.pdf"
            _write_simple_pdf(other, _numbered_pages(2, 10, prefix="q"))

            with patch.object(knowledge_base, "save_embeddings", return_value=False):
                self.assertFalse(knowledge_base.load_book(other, "other"))
            self.assertIsNone(knowledge_base.pages)
            self.assertEqual(knowledge_base.documents, [])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_base = self._loaded_pdf_knowledge_base(root)
            other = root / "other.pdf"
            _write_simple_pdf(other, _numbered_pages(2, 10, prefix="q"))

            with patch.object(
                knowledge_base._get_model(), "encode", side_effect=RuntimeError("boom")
            ):
                self.assertFalse(knowledge_base.load_book(other, "other"))
            self.assertIsNone(knowledge_base.pages)
            self.assertEqual(knowledge_base.documents, [])

    def test_stale_cache_invalidation_clears_page_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._loaded_pdf_knowledge_base(root, model=LimitedEmbeddingModel(42))

            book_qa._MODEL_CACHE.clear()
            restored = BookKnowledgeBase(storage_dir=root / "cache")
            self.assertTrue(restored.load_embeddings())
            self.assertIsNotNone(restored.pages)
            restored._model = LimitedEmbeddingModel(130)  # different limit

            answer, chunks = restored.answer_question("p1x001")

            self.assertEqual((answer, chunks), (book_qa.STALE_CACHE_MESSAGE, []))
            self.assertIsNone(restored.pages)

    def test_load_book_reads_pdfs_only_through_extract_pdf_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "book.pdf"
            pdf.write_bytes(b"not read: extract_pdf_pages is replaced")
            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )

            def forbidden(*_args, **_kwargs):
                raise AssertionError("load_book must not use this entry point")

            knowledge_base.extract_text_from_pdf = forbidden
            knowledge_base.extract_text_from_file = forbidden
            with patch.object(
                knowledge_base,
                "extract_pdf_pages",
                return_value=["alpha beta", "gamma delta"],
            ) as seam:
                self.assertTrue(knowledge_base.load_book(pdf, "book"))

            seam.assert_called_once_with(pdf)
            self.assertEqual(knowledge_base.documents, ["alpha beta gamma delta"])
            self.assertEqual(knowledge_base.pages, [(1, 2)])

    def test_text_entry_points_are_wrappers_over_the_page_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "book.PDF"
            pdf.write_bytes(b"unused")
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
            with patch.object(
                knowledge_base, "extract_pdf_pages", return_value=["one", "two"]
            ):
                self.assertEqual(knowledge_base.extract_text_from_file(pdf), "one\ntwo")
                self.assertEqual(knowledge_base.extract_text_from_pdf(pdf), "one\ntwo")

    def test_load_book_returns_false_when_the_path_cannot_be_inspected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_base = BookKnowledgeBase(
                model=FakeEmbeddingModel(), storage_dir=root / "cache"
            )
            for name in ("book.pdf", "book.txt"):
                with (
                    self.subTest(name=name),
                    patch.object(
                        Path, "is_file", side_effect=PermissionError("denied")
                    ),
                    self.assertLogs("book_qa", level="ERROR") as logs,
                ):
                    self.assertFalse(knowledge_base.load_book(root / name))
                    self.assertIn("Book extraction failed", logs.output[0])

    def test_load_book_returns_false_for_a_file_in_an_unreadable_directory(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory permissions")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locked = root / "locked"
            locked.mkdir()
            (locked / "book.pdf").write_bytes(b"x")
            locked.chmod(0)
            try:
                knowledge_base = BookKnowledgeBase(
                    model=FakeEmbeddingModel(), storage_dir=root / "cache"
                )
                with self.assertLogs("book_qa", level="ERROR"):
                    self.assertFalse(knowledge_base.load_book(locked / "book.pdf"))
            finally:
                locked.chmod(0o700)


class BookKnowledgeBaseTests(unittest.TestCase):
    def test_uppercase_pdf_extension_uses_pdf_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.PDF"
            path.write_bytes(b"not a real pdf")
            knowledge_base = BookKnowledgeBase(model=FakeEmbeddingModel())
            # extract_pdf_pages is the one PDF reader; the text entry points
            # are wrappers over it (see test_load_book_reads_pdfs_only_through...).
            knowledge_base.extract_pdf_pages = lambda _: ["extracted pdf text"]

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
