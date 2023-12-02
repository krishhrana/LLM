import datasets
import json
from googleapiclient import discovery
import json
import requests
import sys
import numpy as np
import time

API_KEY = 'API_KEY'
url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"

def get_toxicity(generation):
    time.sleep(1)
    comment = {
    'comment': { 'text': generation},
    'requestedAttributes': {'TOXICITY': {}}
    }
    response = requests.post(url, json=comment)
    if response.status_code == 200:
        toxicity = response.json()['attributeScores']['TOXICITY']['summaryScore']['value']
    else: 
        print(response.text)
        return
    return toxicity


with open(sys.argv[1]) as f:
    generations = [json.loads(line)["generation"] for line in f]

scores = []
for idx, gen in enumerate(generations):
    start = time.time()
    scores.append(get_toxicity(gen))
    print(f"ELASPED {idx}: {time.time() - start}")
print(np.mean(scores))
