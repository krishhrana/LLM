import tiktoken 

tokenizer = tiktoken.get_encoding('gpt2')

from src.tokenization import return_duplicate_tokens

def test_duplicate_decodings():
    token_set_one, token_set_two = return_duplicate_tokens()
    
    assert token_set_one != token_set_two
    assert tokenizer.decode(token_set_one) == tokenizer.decode(token_set_two)