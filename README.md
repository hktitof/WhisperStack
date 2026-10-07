# 🎙️ WhisperStack — Local Real-Time Dictation & Clipboard Stacking for macOS

WhisperStack is a high-performance native macOS menu-bar dictation application built specifically for Apple Silicon (M1/M2/M3/M4). It runs OpenAI's Whisper models 100% offline on the unified GPU using Apple's MLX framework, while simultaneously stacking your clipboard history in the background with intelligent delta deduplication.

Press a global hotkey anywhere, speak naturally, copy any code or text references mid-speech, press the hotkey again, and your merged speech-and-clipboard payload is typed directly into your focused window in under **1 second**.

---

## ✨ Key Features

* **⚡ Ultra-Fast Local MLX Inference**: Uses `mlx-community/whisper-large-v3-turbo` compiled for Apple Silicon Metal GPU. Zero audio leaves your Mac; zero cloud latency.
* **🚀 Smart Delta Near-Instant Paste (< 0.9s)**: Transcribes in real-time while you speak using a coverage watermark. On stop, it transcribes only the uncovered tail (~0.5–2s) and splices seamlessly with exact-match alignment.
* **🛡️ Target Speaker Verification & Voice Isolation**: Integrates SpeechBrain ECAPA-TDNN (192-dimensional x-vector embeddings on Apple Silicon `mps` GPU). Rejects foreign audio (Mac speakers, podcasts, YouTube videos, other people in the room) so only your enrolled voice is transcribed.
* **📋 Intelligent Clipboard Stacking**: Captures everything you `Cmd+C` while speaking. Features delta deduplication, substring excerpting with anchor tags, multi-screen detection, and Chrome window/tab/URL context injection.
* **🎙️ Lossless Audio Preservation (AI Voice Dataset Generation)**: Automatically archives high-resolution lossless audio (`.flac` or `.wav`) linked directly to exact transcripts in daily JSONL logs—creating a golden dataset for custom TTS voice cloning and ASR benchmarking.
* **🎨 Animated Circular HUD Visualizer**: Beautiful frameless circular overlay widget with dynamic sound wave visualizers, real-time streaming live words, and per-monitor coordinate/size persistence across multi-display setups.
* **⌨️ Native Carbon Global Hotkeys**: System-wide `Cmd+Shift+S` registration via Carbon APIs with `QLockFile` single-instance locking and 500ms debounce—fully compatible with Logitech MX Master mouse button mappings.

---

## 🏗️ Architecture & Pipeline

```text
[Global Hotkey / Logitech Button] -> Cmd+Shift+S
        |
        v
[Audio Capture] -> sounddevice PortAudio (16 kHz / mono / int16)
        |
        +---> [Real-Time Live Worker (Every ~1.0s)]
        |         |--> [VoiceFilter] SpeechBrain ECAPA-TDNN (MPS GPU)
        |         |--> [Live Transcriber] mlx_whisper (Large-v3-Turbo)
        |         +--> [Coverage Watermark] _live_covered_samples
        |
        +---> [Clipboard Monitor] NSPasteboard changeCount polling
        |         +--> Deduplication & Substring Delta Extraction
        |
[Stop Recording] -> Cmd+Shift+S
        |
        v
[Final Mode Engine (SMART Delta)]
        |--> Transcribes ONLY uncovered tail (~0.2-2.0s) with is_tail protection
        |--> Exact-match seam splicing (_merge_tail)
        |
        v
[Lossless Audio Preservation] -> logs/recordings/YYYY-MM-DD/session_N.flac
[Telemetry & Transcript Log]   -> logs/transcripts/YYYY-MM-DD.jsonl & .md
        |
        v
[Auto-Paste] -> AppleScript Cmd+V into frontmost application
```

---

## ⚙️ Configuration Reference (`settings.json`)

```json
{
    "mic_name": "BKD-11 Pro Audio Device",
    "hotkey": "<cmd>+<shift>+s",
    "model_name": "mlx-community/whisper-large-v3-turbo",
    "final_mode": "smart",
    "save_audio_recordings": true,
    "audio_format": "flac",
    "voice_filter_enabled": true
}
```

### Final Transcription Modes (`final_mode`)

| Mode | What Happens on Stop | Paste Latency | Recommended Use |
| :--- | :--- | :--- | :--- |
| **`smart` (Default)** | Live transcript + Whisper pass over ONLY uncovered tail remainder | **Near-instant (< 0.9s)** | Everyday dictation |
| **`full`** | Clean single pass over the entire recording from second 0 | Scales with length (~5-15s) | Context benchmarking |
| **`fast`** | Legacy last-3s slice with fuzzy string stitch | Instant | Regression comparison |

---

## 🚀 Quick Start

### Prerequisites
* macOS 13.0+ running on Apple Silicon (M1, M2, M3, M4).
* Python 3.10 or 3.11 with Homebrew:
  ```bash
  brew install portaudio ffmpeg
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hktitof/WhisperStack.git
   cd WhisperStack
   ```

2. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Enroll your voice for target speaker isolation (optional but recommended):
   ```bash
   python enroll_voice.py
   ```
   *(Speaks for 6 seconds to calibrate your 192-dimensional ECAPA-TDNN vocal fingerprint).*

4. Launch WhisperStack:
   ```bash
   python main.py
   ```

---

## 🔒 Privacy & Local Processing

* **100% Offline**: All audio capture, speaker verification, and speech-to-text inference runs locally on Apple Silicon GPU and Neural Engine.
* **Zero Telemetry Leakage**: No audio, transcripts, or personal embeddings are ever sent to the internet.
* **Git Safe**: Logs, recordings, transcripts, and personal voiceprints are strictly ignored by `.gitignore`.

---

## 📄 License

MIT License — built with passion for developer productivity and local AI workflows.