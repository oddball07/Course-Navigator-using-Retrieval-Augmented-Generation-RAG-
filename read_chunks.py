import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json ={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()['embeddings']
    return embedding




jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0
for json_file in jsons:
    with open(f'jsons/{json_file}') as f:
        content = json.load(f)
        embeddings = create_embedding([c['text'] for c in content])
    for i, chunk in enumerate(content) :
        chunk['chunk_id'] = chunk_id
        chunk['embeddings'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)

#Save this  DataFrame using joblib
joblib.dump(df,"embddings.joblib")

