import re
from .numbers import normalize_numbers
from typing import Pattern, List, Tuple, Literal


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')
_numbers_re = re.compile(r"[0-9]")
_invalid_symbols = re.compile(r"[\[\]~`@#$%^&*()\-_+=|\"\'<>/]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations: List[Tuple[Pattern[str], str]] = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def lowercase(text: str) -> str:
    return text.lower()

def replace_invalid_symbols(text: str) -> str:
    return re.sub(_invalid_symbols, ' ', text)

def expand_abbreviations(text: str) -> str:
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def remove_numbers(text: str) -> str:
    return re.sub(_numbers_re, '', text)

def expand_numbers(text: str) -> str:
    return normalize_numbers(text)

def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, ' ', text)

_CLEANER_TYPES = Literal[
    "base_cleaners"
]

def base_cleaners(text: str, language: str = "english", remove_numbers: bool = False):
    text = lowercase(text)
    if remove_numbers:
        text = remove_numbers(text)
    if (language == "english"):
        text = expand_numbers(text)
        text = expand_abbreviations(text)
    text = replace_invalid_symbols(text)
    text = collapse_whitespace(text)
    return text
