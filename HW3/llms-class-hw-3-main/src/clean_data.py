import datasets
import numpy as np
from get_data import fetch_data
import re
import tiktoken
from cleaning import * 
num_proc = 8

# Filtering non-english text at "word" level
non_english_regex = re.compile(r'[^\u0000-\u007F]+')
def filter_non_english(example):
    text = example['text']
    words = text.split()
    english_words = [w for w in words if not filter_noneng(w)]
    return {'text': " ".join(english_words), 'before_filter': len(words), 'after_filter': len(english_words)}

# to use datasets.map() from huggingface
def filter_pii(example):
    filtered_text = clean_pii(example['text'])
    return {'text': filtered_text, 'before_filter': len(example['text']), 'after_filter': len(filtered_text)}

# Over all cleaning function
def clean(dataset):
    print("Filtering non-english words: ")
    dataset = dataset.map(filter_non_english, num_proc = num_proc)
    before_filter = np.sum(dataset['before_filter'])
    after_filter = np.sum(dataset['after_filter'])
    print(f'Words before filtering: {before_filter}')
    print(f'Words after filtering: {after_filter}')
    print(f'Words removed: {before_filter - after_filter}')
    print()

    print("Applying Offensive Words filtering: ")
    before_filter = len(dataset)
    bad_words_list = open('HW3/llms-class-hw-3-main/bad_words/bad_words_list.txt').read().split('\n')
    dataset = dataset.filter(lambda x: clean_other(x['text'], bad_words=bad_words_list), num_proc=num_proc)
    after_filter = len(dataset)
    print(f'Length before filtering: {before_filter}')
    print(f'Length after filtering: {after_filter}')
    print(f'Documents removed: {before_filter - after_filter}')
    print()

    print("Applying PII filter: ")
    dataset = dataset.map(filter_pii, num_proc=8)
    before_filter = np.sum(dataset['before_filter'])
    after_filter = np.sum(dataset['after_filter'])
    print(f"Charaters before filtering: {before_filter}")
    print(f"Charaters after filtering: {after_filter}")
    print(f'Characters removed: {before_filter - after_filter}')
    
    return dataset


# Cleaning code
train_data, val_data = fetch_data()
train_data_filtered = clean(train_data)
val_data_filtered = clean(val_data)

cols_to_remove = ['before_filter', 'after_filter']
filtered_dataset = datasets.DatasetDict(
    {
        'train': train_data_filtered.remove_columns(cols_to_remove), 
        'val': val_data_filtered .remove_columns(cols_to_remove)
    }
)
filtered_dataset.save_to_disk('cleaned_data.arrow')

# Tokenise the data
tokenizer = tiktoken.get_encoding('gpt2')
def tokenise(sample):
    tokenised_text = tokenizer.encode(sample['text'])
    tokenised_text.append(tokenizer.eot_token)
    return {"tokenised_text": tokenised_text}

train_data_tokenised = train_data_filtered.map(tokenise, num_proc=num_proc)
val_data_tokenised = val_data_filtered.map(tokenise, num_proc=num_proc)

train_data = np.concatenate(train_data_tokenised['tokenised_text'])
val_data = np.concatenate(val_data_tokenised['tokenised_text'])

np.savez('HW3/llms-class-hw-3-main/dataset/tokens.npz', train = train_data, val = val_data)