import re


def whitespace_tokenize(text: str) -> list[str]:
    """
    Split text using whitespace.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    list[str]
        A list of tokens.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return text.split()


def regex_tokenize(text: str) -> list[str]:
    """
    Extract word tokens using a regular expression.

    The tokenizer keeps alphabetic words and contractions such as:
    don't, it's, and I've.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    token_pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?"

    return re.findall(token_pattern, text)


def character_tokenize(
    text: str,
    include_spaces: bool = False,
) -> list[str]:
    """
    Split text into individual characters.

    Parameters
    ----------
    text:
        Input text.

    include_spaces:
        Keep spaces as tokens when True.

    Returns
    -------
    list[str]
        A list of character tokens.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if include_spaces:
        return list(text)

    return [
        character
        for character in text
        if not character.isspace()
    ]

