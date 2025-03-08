import ollama
from transformers import VitsModel, AutoTokenizer
import torch
import pyaudio
import numpy as np
import wave
import scipy
from openai import OpenAI
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


ret, frame = cap.read()

if not ret:
    print("Error: Could not capture frame.")
    exit()


image_path = "captured_img.jpg"
cv2.imwrite(image_path, frame)
print(f"Image saved to {image_path}")


cap.release()


prompt_simple = "Describe the image in an objective manner such that you only explain "
prompt_artful = "Describe this image in an artful manner such that a blind person should be able to visualize it in their mind."

def img_to_text(prompt):
    response = ollama.chat(
        model="llava",
        messages=[
            {"role": "user", "content": prompt, "images": ["./captured_img.jpg"]}
        ],
    )
    return response

selection = input("Choose prompt(Simple:0/Artful:1): ")
if selection == "0":
    response = img_to_text(prompt_simple)
    print("simple")
else:
    response = img_to_text(prompt_artful)
    print("artful")
print(response.message.content)


client = OpenAI(
api_key = os.environ.get("OPENAI_API_KEY"),
)

text = response.message.content

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Translate this into Turkish: {text}",
        }
    ],
    model="gpt-4o",
)

print(chat_completion.choices[0].message.content)
text = chat_completion.choices[0].message.content

model = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

inputs = tokenizer(text, return_tensors="pt")


with torch.no_grad():
	output = model(**inputs).waveform

	import soundfile as sf

	# Save audio using soundfile (supports float32 and more formats)
	sf.write("output_audio.wav", output.T, samplerate=16000)  # Adjust as needed
	print("Saved to output_audio.wav")

with wave.open("output_audio.wav", 'rb') as wf:
    # Instantiate PyAudio and initialize PortAudio system resources (1)
    p = pyaudio.PyAudio()

    # Open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Play samples from the wave file (3)
    while len(data := wf.readframes(1)):  # Requires Python 3.8+ for :=
        stream.write(data)

    # Close stream (4)
    stream.close()

    # Release PortAudio system resources (5)
    p.terminate()
