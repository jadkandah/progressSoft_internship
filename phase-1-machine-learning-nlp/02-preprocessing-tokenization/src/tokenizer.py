"""Basic tokenizers implemented from scratch.

CHECKLIST
---------
[x] Implement word-level tokenizers from scratch.
[x] Implement a character-level tokenizer from scratch.
"""

import re


def whitespace_tokenize(text: str) -> list[str]:
    """Split text at whitespace boundaries."""
    return text.split()


def regex_tokenize(text: str) -> list[str]:
    """Extract sequences of Unicode word characters as tokens."""
    return re.findall(r"\w+", text)


def character_tokenize(text: str, include_spaces: bool = False) -> list[str]:
    """Split text into characters, optionally retaining whitespace."""
    if include_spaces:
        return list(text)
    return [character for character in text if not character.isspace()]
