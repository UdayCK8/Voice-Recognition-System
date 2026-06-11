import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

commands = ["start", "stop", "left", "right", "yes", "no",]

print("Available commands:")
for i, cmd in enumerate(commands, start=1):
    print(f"{i}. {cmd}")

choice = int(input("\nChoose a command number: "))

if choice < 1 or choice > len(commands):
    print("Invalid choice!")
    exit()

command = commands[choice - 1]

sample_rate = 22050
duration = 2

folder = os.path.join("dataset", command)
os.makedirs(folder, exist_ok=True)

num_samples = int(input("How many samples do you want to record? "))

print(f"\nRecording dataset for '{command}'")

for i in range(1, num_samples + 1):

    input(
        f"\nPress Enter and say '{command}' "
        f"({i}/{num_samples})..."
    )

    print("Recording...")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )

    sd.wait()

    filename = os.path.join(
        folder,
        f"{command}{i}.wav"
    )

    write(
        filename,
        sample_rate,
        recording
    )

    print("Saved:", filename)

print("\nDataset creation complete!")