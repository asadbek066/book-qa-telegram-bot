import unicodedata
from pathlib import Path

ALLOWED_BOOK_EXTENSIONS = {".pdf", ".txt"}
MAX_BOOK_FILENAME_LENGTH = 128
# Format characters (bidi overrides, zero-width joiners, ...) and line or
# paragraph separators can visually reorder or split an echoed filename.
DISALLOWED_CHARACTER_CATEGORIES = {"Cf", "Zl", "Zp"}


def _is_disallowed_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        codepoint < 32
        or codepoint == 127
        or 0x80 <= codepoint <= 0x9F
        or unicodedata.category(character) in DISALLOWED_CHARACTER_CATEGORIES
    )


def normalize_book_filename(filename: str | None) -> str:
    """Return a safe local filename for a supported uploaded book."""
    if not isinstance(filename, str):
        # Keep invalid filenames on the handler's single validation-error path.
        raise ValueError("Book filename is empty")  # noqa: TRY004
    if "\x00" in filename:
        raise ValueError("Book filename contains an invalid character")
    name = Path((filename or "").replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise ValueError("Book filename is empty")
    if len(name) > MAX_BOOK_FILENAME_LENGTH:
        raise ValueError("Book filename is too long")
    if any(_is_disallowed_character(character) for character in name):
        raise ValueError("Book filename contains an invalid character")
    if Path(name).suffix.lower() not in ALLOWED_BOOK_EXTENSIONS:
        raise ValueError("Only PDF or TXT files are supported")
    return name
