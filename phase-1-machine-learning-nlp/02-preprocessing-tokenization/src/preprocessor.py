"""TODO: Implement the IMDB text preprocessor.

CHECKLIST
---------
[x] Convert text to lowercase.
[x] Remove HTML tags using a regex.
[x] Remove URLs using a regex.
[x] Remove numbers using a regex.
[x] Remove punctuation using a regex.
[x] Remove special characters using a regex.
[x] Convert newlines, tabs, and carriage returns to spaces.
[x] Collapse repeated spaces.
[x] Add stop-word removal.
[x] Explain stemming and add optional stemming in the correct pipeline position.
[x] Combine the operations in one configurable ``preprocess_text`` function.
"""

# TODO: Add imports only when you need them.
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


STOP_WORDS = set(stopwords.words("english"))
Porter_Stemmer = PorterStemmer() 


def lowercase_text(text: str):
    """TODO: Validate and lowercase text."""
    return text.lower()


def remove_html_tags(text: str):
    """TODO: Remove HTML tags with a regex."""
    return re.sub(r'<.*?>', '', text) # the ? is put because if not it will remove <...> ... <...> so the stuff in between will be removed


def remove_urls(text: str):
    """TODO: Remove HTTP, HTTPS, and www URLs with a regex."""
    return re.sub(r'https?://\S+|www\.\S+', '', text) # \S means any non whitespace character and + means one or more of the previous token


def remove_numbers(text: str):
    """TODO: Remove numbers with a regex."""
    return re.sub(r'\d+', '', text)


def remove_punctuation(text: str):
    """TODO: Remove punctuation with a regex."""
    return re.sub(r'[^\w\s]', '', text) # this removes any character that is not a word character or whitespace


def remove_special_characters(text: str):
    """TODO: Remove unwanted special characters with a regex."""
    return remove_punctuation(text) # since we already removes anything that is not a character or a space we can just call that function


def normalize_whitespace(text: str):
    """TODO: Normalize all whitespace and remove repetitions."""
    return re.sub(r'\s+', ' ', text).strip() 


def remove_stop_words(text: str):
    """TODO: Remove stop words while considering sentiment negations."""
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in STOP_WORDS]
    return ' '.join(filtered_tokens)
    

def stem_text(text: str):
    """TODO: Apply a stemmer token by token."""
    tokens = text.split()
    stemmed_tokens = [Porter_Stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)
    


def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    apply_stemming: bool = False,
) -> str:
    """TODO: Apply preprocessing steps in a justified order."""
    text = lowercase_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    if remove_stopwords:
        text = remove_stop_words(text)

    if apply_stemming:
        text = stem_text(text)

    return text
