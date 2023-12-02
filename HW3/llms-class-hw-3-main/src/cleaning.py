import datasets
import numpy as np
from get_data import fetch_data
import re
import tiktoken

num_proc = 8

def clean_pii(text):
    """Returns text with email addresses, social security numbers, and phone numbers removed."""
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ssn_regex = r'\b\d{3}-\d{2}-\d{4}\b'
    ph_regex = r'(\+1\s*)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}'
    text = re.sub(email_regex, "", text)
    text = re.sub(ssn_regex, "", text)
    text = re.sub(ph_regex, "", text)
    return text

def filter_noneng(text):
    """Returns True if the text is contains non-English characters."""
    non_english_regex = re.compile(r'[^\u0000-\u007F]+')
    return bool(non_english_regex.search(text))
 
def clean_other(text, bad_words):
    """Takes a list of Bad Words and remove the document if it conatins the word of the phrase"""
    text = text.lower()
    words = set(text.split())

    bad_phrases = [b for b in bad_words if " " in b]
    bad_words = set(bad_words)

    contains = (len(words.intersection(bad_words)) > 0) or (any([bp in text for bp in bad_phrases]))
    return not contains

