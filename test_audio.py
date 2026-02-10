#!/usr/bin/env python3
"""
Test script to play all MP3 files in assets/sound/ continuously.
Use this to verify audio output is working on Raspberry Pi 5.
"""
import shutil
import subprocess
import time
from pathlib import Path


def select_backend():
    try:
        from playsound import playsound  # type: ignore
        return ("playsound", playsound)
    except Exception:
        if shutil.which("ffplay"):
            return ("ffplay", None)
        if shutil.which("mpg123"):
            return ("mpg123", None)
    return (None, None)


def play(audio_path, backend, handler):
    if backend == "playsound":
        handler(str(audio_path))
    elif backend == "ffplay":
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
            check=False,
        )
    elif backend == "mpg123":
        subprocess.run(["mpg123", "-q", str(audio_path)], check=False)


def main():
    sound_dir = Path("./assets/sound")
    mp3_files = sorted(sound_dir.glob("*.mp3"))

    if not mp3_files:
        print(f"No MP3 files found in {sound_dir.resolve()}")
        return

    backend, handler = select_backend()
    if backend is None:
        print("No audio backend available!")
        print("Install one of: pip install playsound / sudo apt install ffplay / sudo apt install mpg123")
        return

    print(f"Audio backend: {backend}")
    print(f"Found {len(mp3_files)} MP3 files:")
    for f in mp3_files:
        print(f"  - {f.name}")
    print()

    print("Playing all files continuously (Ctrl+C to stop)...")
    print("=" * 40)
    try:
        while True:
            for mp3 in mp3_files:
                print(f"Playing: {mp3.name}")
                play(mp3, backend, handler)
                time.sleep(0.5)
            print("-" * 40)
            print("Looping...\n")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
