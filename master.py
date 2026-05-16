import os
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import mlx_whisper
from pynput import keyboard
import pyperclip
import threading
import time
import numpy as np
import warnings
import subprocess

warnings.filterwarnings("ignore")

is_recording = False
clipboard_history = []
audio_data = []
last_copied = ""
target_app = ""  # app to paste into

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_data.append(indata.copy())

def monitor_clipboard():
    global last_copied
    while is_recording:
        current = pyperclip.paste()
        if current != last_copied and current != "":
            clipboard_history.append(current)
            last_copied = current
            print(f"📋 captured copy: {current[:30]}...")
        time.sleep(0.5)

def get_frontmost_app():
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def paste_to_app(app_name):
    script = f'''
        tell application "{app_name}" to activate
        delay 0.3
        tell application "System Events" to keystroke "v" using command down
    '''
    subprocess.run(["osascript", "-e", script])

def on_activate():
    global is_recording, audio_data, clipboard_history, last_copied, stream, target_app
    
    if not is_recording:
        # capture the app that was focused BEFORE the hotkey
        target_app = get_frontmost_app()
        print(f"\n🎯 target app: {target_app}")
        print("🔴 started recording! speak and copy things... press cmd+shift+space to stop.")
        
        is_recording = True
        audio_data = []
        clipboard_history = []
        last_copied = pyperclip.paste()
        
        threading.Thread(target=monitor_clipboard, daemon=True).start()
        
        stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', device=5, callback=audio_callback)
        stream.start()
    else:
        print("\n✅ stopping and processing on m1 max gpu...")
        is_recording = False
        stream.stop()
        stream.close()
        
        if not audio_data:
            print("no audio recorded.")
            return
            
        audio_np = np.concatenate(audio_data, axis=0)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        write(temp_path, 16000, audio_np)
        
        try:
            result = mlx_whisper.transcribe(temp_path, path_or_hf_repo='mlx-community/whisper-large-v3-turbo', language='en')
            spoken_text = result['text'].strip()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        final_text = spoken_text
        if clipboard_history:
            final_text += "\n\n[COPIED ITEMS]\n"
            for idx, item in enumerate(clipboard_history, 1):
                final_text += f"--- Copied Item {idx} ---\n{item}\n\n"
                
        pyperclip.copy(final_text)
        print(f"\n--- DEBUG TEXT ---\n{final_text}\n------------------\n")
        print(f"🔥 pasting into {target_app}...")
        
        paste_to_app(target_app)

print("🚀 script is running! press cmd+shift+space to start/stop.")
with keyboard.GlobalHotKeys({'<cmd>+<shift>+<space>': on_activate}) as h:
    h.join()