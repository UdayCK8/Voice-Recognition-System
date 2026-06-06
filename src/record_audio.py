import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename, duration=3, sample_rate=22050):
    print("Speak now...")
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
    sd.wait()
    write(filename, sample_rate, recording)
    print("Recording saved:", filename)
