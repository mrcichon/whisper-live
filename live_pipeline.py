import wave as wav
import pyaudio
import numpy as np
from whisper import load_model, transcribe

CHUNK = 1024
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
p = pyaudio.PyAudio()

# starts recording
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER)

model = load_model('medium')

def live_pipeline():
    piskorski_counter = 0
    while True:
        # data = stream.read(FRAMES_PER_BUFFER)
        # data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        transcript = transcribe(model, data)
        print(transcript)
        if transcript["text"] == "actually":
            piskorski_counter += 1
            print("PISKORSKI_COUNTER: ", piskorski_counter)

live_pipeline()