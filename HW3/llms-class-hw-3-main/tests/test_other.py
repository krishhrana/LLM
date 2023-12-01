# your tests here!
import sys
sys.path.append('llms-class-hw-3-main/src')
from cleaning import clean_other

bad_words = open('llms-class-hw-3-main/bad_words/bad_words_list.txt').read().split('\n')

def test_no_bad_words():
    original_string = "This is a clean document."
    assert clean_other(original_string, bad_words)

def test_contains_bad_word():
    original_string = "This document contains an arsehole"
    assert not clean_other(original_string, bad_words)

def test_case_insensitivity():
    original_string = "This document contains a ARSEHOLE"
    assert not clean_other(original_string, bad_words)

def test_contains_bad_phrase():
    original_string = "This document has a very bad phrase: baby juice"
    assert not clean_other(original_string, bad_words)

def test_word_boundary_accuracy():
    original_string = "This document has a word: @rseh0le."
    assert clean_other(original_string, bad_words)

def test_mixed_bad_words_phrases():
    original_string = "This is arsehole, really bad phrase: baby juice"
    assert not clean_other(original_string, bad_words)



