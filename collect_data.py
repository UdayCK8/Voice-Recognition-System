"""
collect_data.py — Voice Dataset Collector
-----------------------------------------
Usage:
    python collect_data.py

Output:
    dataset/<class>/<class><index>.wav
"""

import os
import sys
import time
import shutil
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# ==================================================
# CONFIG
# ==================================================

CLASSES = [
    "left",
    "no",
    "right",
    "start",
    "stop",
    "yes",
    "down",
    "speak",
    "rcb",
    "class",
    "mine"
]

SAMPLES_PER_CLASS = 100
SAMPLE_RATE = 16000
DURATION = 1.5
COUNTDOWN = 1
VAD_THRESHOLD = 0.005
DATASET_PATH = "dataset"

# ==================================================
# INPUT
# ==================================================

def safe_input(prompt="", default=""):
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return sys.stdin.readline().strip()
    except EOFError:
        return default

# ==================================================
# HELPERS
# ==================================================

def count_existing(folder):
    if not os.path.exists(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

def record_sample():
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    return np.squeeze(audio)

def save_wav(audio, path):
    mx = np.max(np.abs(audio))

    if mx > 0:
        audio = audio / mx

    audio_int16 = np.int16(audio * 32767)

    write(path, SAMPLE_RATE, audio_int16)

def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))

def is_loud_enough(audio):
    return rms(audio) > VAD_THRESHOLD

# ==================================================
# SUMMARY
# ==================================================

def print_summary():

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    total = 0

    for cls in CLASSES:

        folder = os.path.join(DATASET_PATH, cls)

        count = count_existing(folder)

        pct = min(
            100,
            int(count / SAMPLES_PER_CLASS * 100)
        )

        print(
            f"{cls:<10} "
            f"{count:>4} samples "
            f"({pct:>3}%)"
        )

        total += count

    print("-" * 60)
    print(f"Total WAV files: {total}")
    print("=" * 60)

# ==================================================
# CLEANUP
# ==================================================

def clean_invalid_folders():

    if not os.path.exists(DATASET_PATH):
        return

    for folder_name in os.listdir(DATASET_PATH):

        folder_path = os.path.join(
            DATASET_PATH,
            folder_name
        )

        if (
            os.path.isdir(folder_path)
            and folder_name not in CLASSES
        ):

            ans = safe_input(
                f"\nDelete unknown folder "
                f"'{folder_name}'? (y/n): ",
                "n"
            )

            if ans.lower() == "y":

                shutil.rmtree(folder_path)

                print(
                    f"Deleted: {folder_name}"
                )

# ==================================================
# MIC TEST
# ==================================================

def test_microphone():

    print("\nSpeak for 2 seconds...")

    audio = sd.rec(
        int(2 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = np.squeeze(audio)

    level = rms(audio)

    print(f"RMS = {level:.4f}")

    if level < VAD_THRESHOLD:
        print("⚠ Microphone level is low")
    else:
        print("✓ Microphone looks good")

# ==================================================
# RECORD CLASS
# ==================================================

def collect_class(label):

    folder = os.path.join(
        DATASET_PATH,
        label
    )

    os.makedirs(folder, exist_ok=True)

    existing = count_existing(folder)

    needed = (
        SAMPLES_PER_CLASS - existing
    )

    if needed <= 0:

        print(
            f"\n[{label}] already complete "
            f"({existing}/{SAMPLES_PER_CLASS})"
        )

        return

    print("\n" + "=" * 60)
    print(f"WORD   : {label}")
    print(f"HAVE   : {existing}")
    print(f"NEED   : {needed}")
    print("=" * 60)

    safe_input(
        "Press Enter when ready...",
        ""
    )

    idx = existing + 1
    recorded = 0

    while recorded < needed:

        print(
            f"\n[{recorded+1}/{needed}] "
            f"Get Ready..."
        )

        time.sleep(COUNTDOWN)

        print(">>> SPEAK NOW <<<")

        audio = record_sample()

        if not is_loud_enough(audio):

            print(
                "Too quiet. "
                "Please try again."
            )

            continue

        filename = os.path.join(
            folder,
            f"{label}{idx}.wav"
        )

        save_wav(audio, filename)

        print(
            f"✓ Saved: {filename} "
            f"(RMS={rms(audio):.3f})"
        )

        recorded += 1
        idx += 1

    print(
        f"\nCompleted "
        f"{recorded} recordings."
    )

# ==================================================
# CUSTOM RECORDING
# ==================================================

def collect_custom_samples(
    label,
    num_samples
):

    folder = os.path.join(
        DATASET_PATH,
        label
    )

    os.makedirs(folder, exist_ok=True)

    existing = count_existing(folder)

    idx = existing + 1

    print(
        f"\nRecording "
        f"{num_samples} new samples "
        f"for '{label}'"
    )

    recorded = 0

    while recorded < num_samples:

        print(
            f"\n[{recorded+1}/{num_samples}] "
            f"Get Ready..."
        )

        time.sleep(COUNTDOWN)

        print(">>> SPEAK NOW <<<")

        audio = record_sample()

        if not is_loud_enough(audio):

            print(
                "Too quiet. Try again."
            )

            continue

        filename = os.path.join(
            folder,
            f"{label}{idx}.wav"
        )

        save_wav(audio, filename)

        print(
            f"✓ Saved {filename}"
        )

        idx += 1
        recorded += 1

    print(
        f"\nAdded {recorded} samples "
        f"to '{label}'"
    )

# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 60)
    print("VOICE DATA COLLECTOR")
    print("=" * 60)

    clean_invalid_folders()

    print("\nCurrent Status\n")

    for i, cls in enumerate(CLASSES, 1):

        count = count_existing(
            os.path.join(
                DATASET_PATH,
                cls
            )
        )

        bar_len = int(
            min(
                count,
                SAMPLES_PER_CLASS
            )
            / SAMPLES_PER_CLASS
            * 20
        )

        bar = "█" * bar_len
        pad = "░" * (20 - bar_len)

        print(
            f"{i:>2}. "
            f"{cls:<8} "
            f"[{bar}{pad}] "
            f"{count}/{SAMPLES_PER_CLASS}"
        )

    print("\nOptions:")
    print(" 0 = Record ALL classes")

    for i, cls in enumerate(CLASSES, 1):
        print(f"{i:>2} = {cls}")

    print(" c = Custom sample count")
    print(" t = Test microphone")
    print(" s = Dataset summary")
    print(" q = Quit")

    while True:

        choice = safe_input(
            "\nEnter choice: ",
            "q"
        ).lower()

        if choice == "q":

            print("Goodbye.")
            break

        elif choice == "s":

            print_summary()

        elif choice == "t":

            test_microphone()

        elif choice == "0":

            for label in CLASSES:
                collect_class(label)

            print_summary()

            print(
                "\nRun next:\n"
                "python train_voice_model.py"
            )

            break

        elif choice == "c":

            print("\nChoose class:\n")

            for i, cls in enumerate(
                CLASSES,
                1
            ):
                count = count_existing(
                    os.path.join(
                        DATASET_PATH,
                        cls
                    )
                )

                print(
                    f"{i}. "
                    f"{cls} "
                    f"({count} samples)"
                )

            cls_choice = safe_input(
                "\nClass number: "
            )

            if not cls_choice.isdigit():
                continue

            cls_index = int(cls_choice)

            if (
                cls_index < 1
                or cls_index > len(CLASSES)
            ):
                continue

            label = CLASSES[
                cls_index - 1
            ]

            count = safe_input(
                "How many samples? "
            )

            if not count.isdigit():
                continue

            collect_custom_samples(
                label,
                int(count)
            )

        elif (
            choice.isdigit()
            and
            1 <= int(choice) <= len(CLASSES)
        ):

            collect_class(
                CLASSES[
                    int(choice) - 1
                ]
            )

        else:

            print(
                "Invalid choice."
            )

if __name__ == "__main__":
    main()