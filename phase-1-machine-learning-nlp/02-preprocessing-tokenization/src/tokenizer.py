"""TODO: Implement basic tokenizers from scratch.

CHECKLIST
---------
[ ] Validate input types.
[ ] Implement whitespace tokenization.
[ ] Implement word-level regex tokenization.
[ ] Decide how contractions, numbers, and punctuation should be handled.
[ ] Implement character-level tokenization.
[ ] Test empty strings and difficult examples.
[ ] Document limitations of each approach.
"""

# TODO: Add imports only when you need them.


def whitespace_tokenize(text: str) -> list[str]:
    """TODO: Split text at whitespace boundaries."""
    raise NotImplementedError


def regex_tokenize(text: str) -> list[str]:
    """TODO: Extract tokens using a regex pattern you can explain."""
    raise NotImplementedError


def character_tokenize(text: str, include_spaces: bool = False) -> list[str]:
    """TODO: Split text into characters with optional spaces."""
    raise NotImplementedError
