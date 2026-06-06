"""
collect_data.py — Record your own voice samples for training
--------------------------------------------------------------
Usage:
    python collect_data.py

Output: dataset/<class>/<class><index>.wav
"""

import os
import time
import shutil
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# ==================================================
# CONFIG
# ==================================================
CLASSES           = ["left", "no", "right", "start", "stop", "yes"]
SAMPLES_PER_CLASS = 100        # ✅ target: 100 per class
SAMPLE_RATE       = 16000
DURATION          = 1.5
DATASET_PATH      = "dataset"
COUNTDOWN         = 1
VAD_THRESHOLD     = 0.005

# ==================================================
# HELPERS
# ==================================================
def count_existing(folder: str) -> int:
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".wav")])

def record_sample() -> np.ndarray:
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def save_wav(audio: np.ndarray, path: str):
    mx = np.max(np.abs(audio))
    audio_int16 = np.int16(audio / mx * 32767) if mx > 0 else np.int16(audio)
    write(path, SAMPLE_RATE, audio_int16)

def is_loud_enough(audio: np.ndarray) -> bool:
    return float(np.sqrt(np.mean(audio ** 2))) > VAD_THRESHOLD

def print_summary():
    print("\n" + "="*50)
    print("  Dataset Summary")
    print("-"*30)
    total = 0
    for cls in os.listdir(DATASET_PATH) if os.path.exists(DATASET_PATH) else []:
        folder = os.path.join(DATASET_PATH, cls)
        if os.path.isdir(folder):
            count = count_existing(folder)
            flag  = "✓" if cls in CLASSES and count >= SAMPLES_PER_CLASS else "⚠" if cls not in CLASSES else " "
            print(f"  {flag} {cls:<10}: {count} samples")
            total += count
    print(f"\n  Total WAV files: {total}")
    print("="*50)

def clean_invalid_folders():
    """Remove folders that are not in CLASSES (e.g. 'copy')"""
    if not os.path.exists(DATASET_PATH):
        return
    removed = []
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if os.path.isdir(folder_path) and folder_name not in CLASSES:
            confirm = input(f"\n  ⚠ Found unknown folder '{folder_name}' — delete it? (y/n): ").strip().lower()
            if confirm == "y":
                shutil.rmtree(folder_path)
                removed.append(folder_name)
                print(f"  ✓ Deleted: {folder_name}")
            else:
                print(f"  Skipped: {folder_name}")
    if removed:
        print(f"\n  Cleaned up: {removed}")

# ==================================================
# COLLECT ONE CLASS
# ==================================================
def collect_class(label: str):
    folder   = os.path.join(DATASET_PATH, label)
    os.makedirs(folder, exist_ok=True)

    existing = count_existing(folder)
    needed   = SAMPLES_PER_CLASS - existing

    if needed <= 0:
        print(f"\n  [{label}] Already has {existing}/{SAMPLES_PER_CLASS} samples. ✓ Done.")
        return

    print(f"\n{'='*50}")
    print(f"  Word  : '{label.upper()}'")
    print(f"  Have  : {existing}  |  Need: {needed} more  |  Target: {SAMPLES_PER_CLASS}")
    print(f"  Say '{label}' clearly each time you see SPEAK NOW")
    print(f"{'='*50}")
    input("  Press Enter when ready...\n")

    recorded = 0
    idx      = existing + 1
    retries  = 0

    while recorded < needed:
        print(f"  [{recorded+1}/{needed}]  Get ready...", end="", flush=True)
        time.sleep(COUNTDOWN)
        print(f"\r  [{recorded+1}/{needed}]  >>> SPEAK NOW <<<          ", flush=True)

        audio = record_sample()

        if not is_loud_enough(audio):
            retries += 1
            print(f"           ⚠ Too quiet — retrying ({retries}) — speak louder!")
            if retries >= 5:
                print("           Too many quiet retries. Check your microphone.")
                retries = 0
            continue

        retries  = 0
        filename = os.path.join(folder, f"{label}{idx}.wav")
        save_wav(audio, filename)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"           ✓ {filename}  (RMS: {rms:.3f})")

        recorded += 1
        idx      += 1

    print(f"\n  ✓ Done! Recorded {recorded} samples for '{label}'.")
    print(f"  Total now: {count_existing(folder)}/{SAMPLES_PER_CLASS}")

# ==================================================
# MAIN
# ==================================================
def main():
    print("\n" + "="*50)
    print("  VOICE DATA COLLECTOR")
    print(f"  Target : {SAMPLES_PER_CLASS} samples per class")
    print(f"  Classes: {CLASSES}")
    print("="*50)

    # Clean up invalid folders first
    clean_invalid_folders()

    # Show current status
    print("\nCurrent status:")
    for i, cls in enumerate(CLASSES, 1):
        existing = count_existing(os.path.join(DATASET_PATH, cls))
        remaining = max(0, SAMPLES_PER_CLASS - existing)
        bar = "█" * int(existing / SAMPLES_PER_CLASS * 20)
        pad = "░" * (20 - len(bar))
        done = "✓" if remaining == 0 else f"{remaining} more needed"
        print(f"  {i}. {cls:<8} [{bar}{pad}] {existing:>3}/{SAMPLES_PER_CLASS}  {done}")

    print("\nOptions:")
    print("  0 = Record ALL classes")
    print("  1-6 = Record specific word")
    print("  s = Show dataset summary")
    print("  q = Quit")

    while True:
        choice = input("\nEnter choice: ").strip().lower()

        if choice == "q":
            print("\nExiting.")
            break
        elif choice == "s":
            print_summary()
        elif choice == "0":
            for label in CLASSES:
                collect_class(label)
            print_summary()
            print("\n  All done! Now run:  python train_voice_model.py")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(CLASSES):
            collect_class(CLASSES[int(choice) - 1])
            # Ask if they want to continue
            again = input("\n  Record another word? (0=all / 1-6=word / q=quit): ").strip().lower()
            if again == "q":
                break
            elif again == "0":
                for label in CLASSES:
                    collect_class(label)
                print_summary()
                break
            elif again.isdigit() and 1 <= int(again) <= len(CLASSES):
                collect_class(CLASSES[int(again) - 1])
        else:
            print("  Invalid choice — enter 0-6, s, or q")

if __name__ == "__main__":
    main()