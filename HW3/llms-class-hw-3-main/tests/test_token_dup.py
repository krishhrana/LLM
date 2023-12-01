import tiktoken 
import sys
sys.path.append('llms-class-hw-3-main/src')
tokenizer = tiktoken.get_encoding('gpt2')

from tokenization import return_duplicate_tokens

def test_duplicate_decodings():
    token_set_one, token_set_two = return_duplicate_tokens()
    
    assert token_set_one != token_set_two
    assert tokenizer.decode(token_set_one) == tokenizer.decode(token_set_two)

test_duplicate_decodings()