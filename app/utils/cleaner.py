"""
Input sanitation for text headed into the model.

Don't trust raw user input. This module runs a small, fixed pipeline of
defensive transforms on every incoming string:
  1. Unicode normalization (so 'é' from one source matches 'é' from another).
  2. Whitespace collapse (newlines/tabs/multiple spaces → one space).
  3. Strip control characters (NULL, bell, etc.) that have no business in text.
  4. Hard length cap (defends against accidental or malicious giant inputs).
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize and sanitize model input text. Always returns a safe string."""
    # NFKC: Unicode "compatibility" normalization. Folds visually-equivalent
    # sequences into a canonical form so 'é' (e + combining-acute) and the
    # precomposed 'é' tokenize identically.
    text = unicodedata.normalize("NFKC", text)

    # Collapse any whitespace runs into a single space and trim ends.
    # \s matches spaces, tabs, newlines, etc.
    text = re.sub(r"\s+", " ", text).strip()

    # Strip control characters except tab (\x09), newline (\x0A), and CR (\x0D).
    # The \x20-\x7E range is printable ASCII; \x80-\xFF keeps extended Latin chars.
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", "", text)

    # Hard cap. Even though Pydantic already rejects > 32k inputs at the
    # API boundary, keeping this here means callers of clean_text() can't
    # accidentally bypass that limit.
    return text[:32_000]
