import wave as wav
import asyncio
import pyaudio
import numpy as np
from whisper import load_model, transcribe

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

# starts recording
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER)

model = load_model('medium')

async def live_pipeline():
    while True:
        data = stream.read(FRAMES_PER_BUFFER)
        data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        print(data.shape)
        return data

async def live_transcribe():
    while True:
        transcript = transcribe(model, await live_pipeline())
        print(transcript)
        await asyncio.sleep(2)

async def main():
    await asyncio.gather(live_transcribe())


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
asyncio.run(main())
