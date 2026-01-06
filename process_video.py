# Program to convert mp4 to mp3
import os
import subprocess

files = os.listdir('videos')
for file in files:
    print(file)
    subprocess.run(["ffmpeg","-i", f"videos/{file}", f"audios/{file}.mp3"])
