import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import tempfile
import wave
import os

# Audio settings
fs = 16000
silence_threshold = 0.015
silence_duration_limit = 1.0  # seconds of silence = STOP speaking

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

audio_queue = queue.Queue()

recording = []
silence_time = 0

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

print("Listening... (Speak anytime)")

with sd.InputStream(
    samplerate=fs,
    channels=1,
    callback=callback
):

    while True:

        audio_chunk = audio_queue.get()
        audio_chunk = audio_chunk.flatten()

        volume = np.max(np.abs(audio_chunk))

        # If speech detected → keep recording
        if volume > silence_threshold:
            recording.append(audio_chunk)
            silence_time = 0

        # If silence → start counting
        else:
            if len(recording) > 0:
                silence_time += 0.1  # approximate time step

        # If silence too long → finalize speech
        if recording and silence_time > silence_duration_limit:

            print("\nProcessing speech...")

            full_audio = np.concatenate(recording)

            # normalize
            full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-6)

            # save temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_name = temp_file.name
            temp_file.close()

            with wave.open(temp_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes((full_audio * 32767).astype(np.int16))

            segments, info = model.transcribe(
                temp_name,
                language="en",
                beam_size=5,
                temperature=0
            )

            text = " ".join([s.text for s in segments]).strip()

            if text:
                print("Customer:", text)

            # reset
            recording = []
            silence_time = 0

            os.remove(temp_name)

#python realtime.py