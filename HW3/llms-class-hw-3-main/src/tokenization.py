import tiktoken 

tokenizer = tiktoken.get_encoding('gpt2')


def return_duplicate_tokens():
    token_list_one = [31085, 41194, 49924, 25630,  5857]
    token_list_two = [31085, 41194, 12323, 354, 786, 25630, 5857]
    
    return token_list_one, token_list_two


