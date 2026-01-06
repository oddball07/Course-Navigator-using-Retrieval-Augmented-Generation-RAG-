import whisper
import importlib
import json
import os

model = whisper.load_model("large-v2")

result = model.transcribe(audio="./01.mp4.mp3", language = "hi", task = "translate")

print(result["segments"])

audios = os.listdir("./audios")
for audio in audios:
    result = model.transcribe(audio=f"./audios/{audio}", language="hi", task="translate")
    print(result["segments"])
    
os.makedirs('jsons', exist_ok=True)
for audio in audios:
  result = model.transcribe(audio = f"./audios/{audio}", language = "hi", task = "translate")
  chunks = []

  for segment in result["segments"]:
    chunks.append({"id": segment["id"],"title": audio,"text": segment["text"], "start": segment["start"], "end": segment["end"]})

  # chunks_with_metadata = {chunks: chunk, text: result}

  with open(f"./jsons/{audio}.json", "w") as f:
      json.dump(chunks, f)