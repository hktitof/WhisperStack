"""
enroll_voice.py — One-time voice calibration for Abdel in WhisperStack.

Records a 6-second sample of Abdel speaking cleanly, extracts the
192-dimensional ECAPA-TDNN speaker embedding, and saves it to
voiceprints/abdel_voiceprint.npy.

Usage:
    /Users/Work/Projects/Whisper-local/venv/bin/python enroll_voice.py
"""

import json
import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from pathlib import Path

from voice_filter import VoiceFilter, DEFAULT_PROFILE_PATH, SAMPLE_RATE

SETTINGS_FILE = Path("/Users/Work/Projects/Whisper-local/settings.json")
ENROLLMENT_WAV = Path(
    "/Users/Work/Projects/Whisper-local/voiceprints/abdel_enrollment.wav"
)
DURATION_SECONDS = 6.0


def get_configured_mic_index() -> int | None:
    """Reads settings.json to match Abdel's chosen microphone device index."""
    if not SETTINGS_FILE.exists():
        return None
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)
        mic_name = settings.get("mic_name", "System Default")
        if mic_name == "System Default":
            return None
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0 and dev.get("name") == mic_name:
                return idx
    except Exception as exc:
        print(f"⚠️ Could not resolve microphone from settings: {exc}")
    return None


def main():
    print("=" * 60)
    print("🎙️  WhisperStack Voice Enrollment & Calibration Tool")
    print("=" * 60)

    device_idx = get_configured_mic_index()
    device_name = "System Default"
    if device_idx is not None:
        try:
            device_name = sd.query_devices(device_idx).get("name", "Unknown")
        except Exception:
            pass

    print(f"🔌 Target Microphone: {device_name} (index: {device_idx})")
    print("\n👉 Instructions:")
    print("   Please speak naturally into your microphone for 6 seconds.")
    print("   You can say something like:")
    print("   'Hey, this is Abdel, calibrating my voice for WhisperStack.'\n")

    if "--auto" in sys.argv:
        print("⏳ Auto-start enabled. Recording will start in 3 seconds...")
        for count in range(3, 0, -1):
            print(f"   Starting in {count}...", flush=True)
            time.sleep(1.0)
    else:
        try:
            input("Press [Enter] when you are ready to begin speaking...")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1

    print("\n🔴 RECORDING NOW... (6 seconds) Speak clearly!")
    total_samples = int(SAMPLE_RATE * DURATION_SECONDS)

    try:
        recording = sd.rec(
            total_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=device_idx,
        )
        for remaining in range(int(DURATION_SECONDS), 0, -1):
            print(f"   ⏱️  {remaining} seconds remaining...", end="\r", flush=True)
            time.sleep(1.0)
        sd.wait()
        print("\n✅ Recording complete!")
    except Exception as exc:
        print(f"\n❌ Audio recording error: {exc}")
        return 1

    audio_flat = recording.flatten()
    peak = np.max(np.abs(audio_flat))
    print(f"📊 Peak volume level: {peak} (Minimum expected: 500)")

    if peak < 400:
        print(
            "⚠️ Warning: The recording was very quiet. Please check your mic volume and try again."
        )
        return 1

    # Save reference WAV
    ENROLLMENT_WAV.parent.mkdir(parents=True, exist_ok=True)
    wav_write(str(ENROLLMENT_WAV), SAMPLE_RATE, audio_flat)
    print(f"💾 Saved calibration audio to: {ENROLLMENT_WAV}")

    # Extract embedding
    print("⚙️ Initializing VoiceFilter neural encoder...")
    vf = VoiceFilter()
    emb = vf.compute_embedding(audio_flat, sample_rate=SAMPLE_RATE)

    if emb is None:
        print(
            "❌ Failed to extract speaker embedding. Please ensure you spoke during recording."
        )
        return 1

    # Save numpy embedding
    vf.save_profile(emb)

    # Self-test verification
    is_match, score = vf.verify_audio(audio_flat, sample_rate=SAMPLE_RATE)
    print(f"🔍 Self-verification test score: {score:.3f} (Match: {is_match})")

    if score > 0.85:
        print("\n🎉 PERFECT! Your voice profile is enrolled and calibrated.")
        print(f"   File: {DEFAULT_PROFILE_PATH}")
        print("   WhisperStack will now use this profile to isolate only your voice!")
        return 0
    else:
        print(f"\n⚠️ Calibration finished with moderate score ({score:.3f}).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
