from get_data import fetch_data
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
from urllib.parse import urlparse
from collections import Counter
import math

tokenizer = tiktoken.get_encoding('gpt2')

train_data, val_data = fetch_data()
print(f"LENGTH TRAIN DATA: {len(train_data)}")

# Get total words in the dataset
def get_word_count(dataset):
    total_words = sum([len(i['text'].split(' ')) for i in dataset])
    return total_words
total_words = get_word_count(train_data)
print(f"TOTAL WORDS: {total_words}")


# Tokenise the dataset
def tokenise(sample):
    tokenised_text = tokenizer.encode(sample['text'])
    tokenised_text.append(tokenizer.eot_token)
    return {"tokenised_text": tokenised_text}
train_data_tokenised = train_data.map(tokenise, num_proc=8)
val_data_tokenised = val_data.map(tokenise, num_proc=8)

# Count number of tokens in the dataset
def get_token_counts(dataset):
    total_tokens = sum([len(i['tokenised_text']) for i in dataset])
    return total_tokens
total_tokens = get_token_counts(train_data_tokenised)
print(f"TOTAL TOKENS: {total_tokens}")

# Plot Document size
def plot_document_size_hist(dataset):
    token_counts = [len(i) for i in dataset['tokenised_text']]
    print(f"Minimum Token Count: {min(token_counts)}")
    print(f"Maximum token Count: {max(token_counts)}")
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Document Sizes (Number of Tokens)')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Documents')
    plt.grid(axis='y', alpha=0.75)

    # Showing the histogram
    plt.show()

plot_document_size_hist(train_data_tokenised)


def plot_domain_hist(domains, token_counts, xlabel = "Documents"):
    log_token_counts = np.log10(token_counts)
    plt.figure(figsize=(10, 8))
    plt.barh(domains, log_token_counts, color='skyblue')
    plt.xlabel(f'# {xlabel} (log scale, base 10)')
    plt.ylabel('Domain')
    plt.title('Histogram of Document Sizes')
    plt.gca().invert_yaxis()  # Inverts the Y-axis so the largest bar is at the top
    plt.show()


def get_domain(url):
    import tldextract
    extracted = tldextract.extract(url)
    tld = extracted.suffix
    return tld

def get_website(url):
    parsed_url = urlparse(url)
    website = parsed_url.netloc
    return website

def get_key_values(c, top_n):
    keys = [item[0] for item in c.most_common(top_n)]
    values = [item[1] for item in c.most_common(top_n)]
    return keys, values

from collections import Counter
def domain_token_counts(dataset, tld = True):
    domain_counter = Counter()
    for i in dataset:
        if tld:
            count_key = get_domain(i['url'])
        else:
            count_key = get_website(i['url'])
        domain_counter[count_key] = domain_counter[count_key] + len(i['tokenised_text'])
    return domain_counter

# Plot for top level domain tokens
domain_counts = domain_token_counts(train_data_tokenised, tld=True)
top_level_domains, tld_counts = get_key_values(domain_counts, 25)
plot_domain_hist(domains=top_level_domains, token_counts=tld_counts)

# Plot for url tokens
url_counts = domain_token_counts(train_data_tokenised, tld = False)
urls, url_token_counts = get_key_values(url_counts, 25)
plot_domain_hist(domains=urls, token_counts=url_token_counts)


def domain_document_counts(dataset, tld = True):
    domain_counter = Counter()
    for i in dataset:
        if tld:
            count_key = get_domain(i['url'])
        else:
            count_key = get_website(i['url'])
        domain_counter[count_key] = domain_counter[count_key] + 1
    return domain_counter

# Plot for doc counts
domain_doc_counts = Counter(list(map(get_domain, train_data_tokenised['url'])))
url_doc_counts = Counter(list(map(get_website, train_data_tokenised['url'])))

top_level_domains, tld_doc_counts = get_key_values(domain_doc_counts, 25)
urls, u_doc_count = get_key_values(url_doc_counts, 25)

plot_domain_hist(domains=top_level_domains, token_counts=tld_doc_counts, xlabel="documents")
plot_domain_hist(domains=urls, token_counts=u_doc_count, xlabel='documents')

def plot_digit_hist(token_counts, n):
    plt.hist(token_counts, bins=range(min(token_counts), max(token_counts) + 2), align='left', rwidth=0.8)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Number of Tokens in Every {n}-Digit Number')
    plt.xticks(range(min(token_counts), max(token_counts) + 1))
    plt.show()

def get_n_digit_nums(n):
    numbers = [str(i) for i in range(int(math.pow(10, n - 1)), int(math.pow(10, n)))]
    print(numbers[0], numbers[-1])
    token_counts = [len(i) for i in tokenizer.encode_batch(numbers)]
    return token_counts

plot_digit_hist(get_n_digit_nums(3), 3)
plot_digit_hist(get_n_digit_nums(5), 5)

