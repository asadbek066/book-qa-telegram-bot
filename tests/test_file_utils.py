import unittest

from file_utils import normalize_book_filename


class NormalizeBookFilenameTests(unittest.TestCase):
    def test_keeps_plain_supported_filename(self):
        self.assertEqual(normalize_book_filename("notes.pdf"), "notes.pdf")

    def test_removes_untrusted_path_components(self):
        self.assertEqual(normalize_book_filename("../../private/book.pdf"), "book.pdf")
        self.assertEqual(normalize_book_filename(r"..\private\book.txt"), "book.txt")

    def test_accepts_uppercase_supported_extension(self):
        self.assertEqual(normalize_book_filename("BOOK.PDF"), "BOOK.PDF")

    def test_rejects_unsupported_or_empty_names(self):
        for filename in ("script.py", "archive.pdf.exe", "", "."):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    normalize_book_filename(filename)


if __name__ == "__main__":
    unittest.main()
