import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

fs = 16000
seconds = 15

print("Speak now... (15 seconds recording)")

audio = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1
)

sd.wait()

write("recording.wav", fs, audio)

print("Processing speech...")

model = WhisperModel(
    "base",
    device="cpu"
)

segments, info = model.transcribe("recording.wav")

text = ""

for segment in segments:
    text += segment.text.strip() + " "

text = text.strip()

print("\nOutput:")

if text == "" or len(text) < 2:
    print("I didn’t catch that. Please say it again.")
else:
    print("Customer:", text)

#python speech.py
