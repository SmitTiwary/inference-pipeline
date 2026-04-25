import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize and sanitize model input text."""
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-printable control characters (keep newlines)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", "", text)
    # Truncate to a safe max length to prevent prompt injection via overflow
    return text[:32_000]
