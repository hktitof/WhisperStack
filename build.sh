#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build.sh — WhisperStack complete build pipeline (slim edition)
# Excludes torch/numba/llvmlite — not needed for the MLX inference path.
#
# Usage (from the project root, with venv active):
#   chmod +x build.sh && ./build.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WhisperStack — Build Pipeline (slim)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Install / upgrade Python dependencies ──────────────────────────────────
echo ""
echo "▸ Installing Python dependencies…"
pip install --upgrade \
  PyQt6 \
  sounddevice \
  scipy \
  pyperclip \
  pynput \
  mlx-whisper \
  numpy \
  pyinstaller \
  pyobjc-framework-Cocoa   # needed for NSApp dock-hide at runtime

# ── 2. Generate icon.icns ─────────────────────────────────────────────────────
echo ""
echo "▸ Generating icon.icns…"
python make_icon.py

# ── 3. Clean previous build ───────────────────────────────────────────────────
echo ""
echo "▸ Cleaning previous build artefacts…"
rm -rf build dist WhisperStack.spec
rm -rf /Applications/WhisperStack.app 2>/dev/null || true

# ── 4. Run PyInstaller ────────────────────────────────────────────────────────
echo ""
echo "▸ Running PyInstaller (excluding torch/numba — stubbed via runtime hook)…"

pyinstaller \
  --name "WhisperStack" \
  --windowed \
  --onefile \
  --icon "icon.icns" \
  --osx-bundle-identifier "com.whisperstack.app" \
  \
  --runtime-hook "hook_notorch.py" \
  \
  --collect-all mlx \
  --collect-all mlx_whisper \
  --collect-all pynput \
  --collect-all pyperclip \
  \
  --hidden-import "pynput.keyboard._darwin" \
  --hidden-import "pynput.mouse._darwin" \
  --hidden-import "scipy.io.wavfile" \
  --hidden-import "scipy._lib.messagestream" \
  --hidden-import "PyQt6.QtCore" \
  --hidden-import "PyQt6.QtGui" \
  --hidden-import "PyQt6.QtWidgets" \
  \
  --add-data "recorder.py:." \
  --add-data "overlay.py:." \
  --add-data "hook_notorch.py:." \
  \
  --exclude-module "torch" \
  --exclude-module "torchvision" \
  --exclude-module "torchaudio" \
  --exclude-module "torch._C" \
  --exclude-module "numba" \
  --exclude-module "llvmlite" \
  --exclude-module "tensorboard" \
  --exclude-module "tensorflow" \
  --exclude-module "matplotlib" \
  --exclude-module "IPython" \
  --exclude-module "PIL" \
  \
  main.py

# ── 5. Patch Info.plist ───────────────────────────────────────────────────────
echo ""
echo "▸ Patching Info.plist…"
PLIST="dist/WhisperStack.app/Contents/Info.plist"

/usr/libexec/PlistBuddy \
  -c "Add :LSUIElement bool true" \
  "$PLIST" 2>/dev/null || \
/usr/libexec/PlistBuddy \
  -c "Set :LSUIElement true" \
  "$PLIST"

/usr/libexec/PlistBuddy \
  -c "Add :NSMicrophoneUsageDescription string 'WhisperStack needs the microphone to transcribe speech.'" \
  "$PLIST" 2>/dev/null || true

/usr/libexec/PlistBuddy \
  -c "Add :NSAppleEventsUsageDescription string 'WhisperStack uses AppleScript to paste transcribed text into the active app.'" \
  "$PLIST" 2>/dev/null || true

# ── 6. Install to /Applications and strip quarantine ─────────────────────────
echo ""
echo "▸ Installing to /Applications and removing quarantine flag…"
cp -r dist/WhisperStack.app /Applications/
xattr -cr /Applications/WhisperStack.app
  echo "▸ Deep codesigning the app bundle..."
  codesign --force --deep --sign - /Applications/WhisperStack.app

APP_SIZE=$(du -sh /Applications/WhisperStack.app | cut -f1)
echo "  App size: ${APP_SIZE}"

# ── 7. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Build + install complete!"
echo ""
echo "  Installed: /Applications/WhisperStack.app"
echo ""
echo "  ⚠️  Grant permissions to the /Applications copy:"
echo ""
echo "  1. open /Applications/WhisperStack.app"
echo "  2. System Settings → Privacy → Accessibility"
echo "     remove old entry, add WhisperStack from /Applications"
echo "  3. System Settings → Privacy → Input Monitoring"
echo "     remove old entry, add WhisperStack from /Applications"
echo "  4. System Settings → Privacy → Automation"
echo "     allow WhisperStack → System Events"
echo "  5. Quit + reopen the app after all three are granted."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"