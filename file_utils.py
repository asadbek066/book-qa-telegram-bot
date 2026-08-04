from pathlib import Path


ALLOWED_BOOK_EXTENSIONS = {".pdf", ".txt"}


def normalize_book_filename(filename: str | None) -> str:
    """Return a safe local filename for a supported uploaded book."""
    name = Path((filename or "").replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise ValueError("Book filename is empty")
    if Path(name).suffix.lower() not in ALLOWED_BOOK_EXTENSIONS:
        raise ValueError("Only PDF or TXT files are supported")
    return name
