# WhisperStack 🎙️📋
**Version:** v0 MVP (Terminal Edition)

WhisperStack is a blazing fast macOS voice-and-clipboard workflow tool built specifically for Apple Silicon. It uses Apple's `mlx` framework to run local AI speech-to-text, while simultaneously "stacking" your clipboard history in the background.

## What it does
- Listens for a global hotkey (`cmd+shift+space`) to start/stop recording.
- Captures your voice using the highly accurate `whisper-large-v3-turbo` model.
- Monitors your clipboard in the background, capturing and stacking everything you copy (`cmd+c`) while speaking.
- Merges the transcribed audio and your copied text (code snippets, links, etc.) into one structured block.
- Automatically pastes the final output back into whatever app you were using.

## The AI Model
This tool uses `mlx-community/whisper-large-v3-turbo`. It is the absolute sweet spot for M-series chips (like the M1 Max), offering instant real-time dictation speed while perfectly understanding heavy developer jargon and complex code vocabulary without hallucinating.

## macOS Permissions Required ⚠️
Because this script runs in the background and simulates keystrokes, macOS security requires you to grant it specific permissions:
1. **Accessibility:** Go to `System Settings -> Privacy & Security -> Accessibility` and toggle ON your Terminal app (or VS Code). This is mandatory to detect the global hotkey and to perform the automatic `cmd+v` paste.
2. **Microphone / Input Monitoring:** macOS will ask for microphone access on the first run so Python can actually record your voice.

## Usage
1. Install requirements and run `python master.py`.
2. Focus the app where you want the final output pasted (e.g., Notes, VS Code, browser).
3. Press `cmd+shift+space` to start recording.
4. Speak naturally, and freely copy (`cmd+c`) any text or code snippets you need while the mic is live.
5. Press `cmd+shift+space` again to stop.
6. The script processes the audio on your GPU and instantly pastes the merged result!
