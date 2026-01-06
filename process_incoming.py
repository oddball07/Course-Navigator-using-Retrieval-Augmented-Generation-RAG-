import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json ={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()['embeddings']
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json ={
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,})
    
    response = r.json()
    # print(response)
    return response


df = joblib.load("embddings.joblib")







incoming_query = input("Enter your query: ")
question_embedding = create_embedding([incoming_query])[0]

# Find similarities between the question_embedding and chunk embeddings
similarities = cosine_similarity(np.vstack(df['embeddings'].values),[question_embedding]).flatten()

top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]
# print(new_df[["title", "text"]])

prompt = f''' Here are video subtitle chunks containing video title, video number, video start time, video end time and transcription text. The data is about a python course:

{new_df[["title", "text", "id", "start", "end"]].to_json(orient="records")}
-------------------------------------------------
{incoming_query}
User asked this question related to the video chunks. you have to answer where and how much content is taught in which video and at what timestamp and guide the user to go to that video.
If user asked unrelated question, tell them that you can only answer questions related to the course videos.
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)
    
    
response = inference(prompt)['response']
print(response)
print(type(response))

with open("response.txt", "w") as f:
    f.write(response)


# for index, item in new_df.iterrows():
#     print(f"{index} | transcription: {item['text']} | Title: {item['title']} | ID: {item['id']} | Start: {item['start']} | End: {item['end']}")
