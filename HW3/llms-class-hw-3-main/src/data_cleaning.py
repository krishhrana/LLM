import datasets
import numpy as np
import cleaning
from get_data import fetch_data
import re
num_proc = 8

train_data, val_data = fetch_data()

# Filtering non-english text at word level
non_english_regex = re.compile(r'[^\u0000-\u007F]+')
def filter_non_english(example):
    text = example['text']
    words = text.split()
    english_words = [w for w in words if not cleaning.filter_noneng(w)]
    return {'text': " ".join(english_words), 'before_filter': len(words), 'after_filter': len(english_words)}



def clean(dataset):
    print("Filtering non-english words: ")
    dataset = dataset.map(filter_non_english, num_proc = num_proc)
    print(dataset)
    before_filter = np.sum(dataset['before_filter'])
    after_filter = np.sum(dataset['after_filter'])
    print(f'Words before filtering: {before_filter}')
    print(f'Words after filtering: {after_filter}')
    print(f'Words removed: {before_filter - after_filter}')
    print()

    print("Applying Offensive Words filtering: ")
    print(f'Length before filtering: {before_filter}')
    bad_words_list = open('data/bad_words_list.txt').read().split('\n')
    dataset = dataset.filter(lambda x: cleaning.clean_other(x, bad_words_list))
    print(f'Length after filtering: {after_filter}')
    print(f'Words removed: {before_filter - after_filter}')

    
    return dataset


bad_words_list = open('data/bad_words_list.txt').read().split('\n')
fil_data = train_data[0].filter(lambda x: cleaning.clean_other(x['text'], bad_words_list))
print(fil_data)