"""
recorder.py — Thread-safe audio capture, adaptive voice metering,
five-second live transcription, clipboard stacking, and final transcription.
"""

from __future__ import annotations

import os
import json
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import Quartz
from AppKit import NSEvent, NSPasteboard, NSScreen, NSWorkspace
import mlx_whisper
import numpy as np
import pyperclip
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from app_logger import logger, log_session

# >>> VOICE-FILTER-ISOLATION: Target speaker verification gate
# Filters out foreign audio (iPhone, podcasts, YouTube, other speakers)
try:
    from voice_filter import VoiceFilter

    _VOICE_FILTER_AVAILABLE = True
except Exception as _vf_err:
    print(f"⚠️ VoiceFilter import error: {_vf_err}")
    _VOICE_FILTER_AVAILABLE = False
# <<< VOICE-FILTER-ISOLATION

warnings.filterwarnings("ignore")

# Make Homebrew utilities visible to background threads and libraries.
_HOMEBREW_PATH = "/opt/homebrew/bin"
if _HOMEBREW_PATH not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_HOMEBREW_PATH}{os.pathsep}{os.environ.get('PATH', '')}"

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"

MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
LIVE_INTERVAL_SECONDS = 1.0
MIN_TRANSCRIPTION_SECONDS = 1.0

# Absolute limits prevent an unusually quiet calibration from opening the gate.
MIN_GATE_PEAK = 400.0
MAX_GATE_PEAK = 1_200.0
DEFAULT_GATE_PEAK = 650.0

# Calibrate from the first short section after recording starts.
CALIBRATION_SECONDS = 0.75
NOISE_GATE_MULTIPLIER = 2.8
NOISE_GATE_MARGIN = 120.0

# Meter behaviour.
VOICE_HOLD_SECONDS = 0.16
METER_ATTACK = 0.72
METER_RELEASE = 0.20
MAX_EXPECTED_VOICE_PEAK = 12_000.0

SILENT_HALLUCINATIONS = {
    "",
    "you",
    "you.",
    "thank you",
    "thank you.",
    "thank you for watching",
    "thank you for watching.",
    "thanks for watching",
    "thanks for watching.",
    "bye",
    "bye.",
}

PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"


def _get_system_timezone() -> str:
    iana_tz = ""
    try:
        if os.path.exists("/etc/localtime"):
            target = os.readlink("/etc/localtime")
            if "zoneinfo/" in target:
                iana_tz = target.split("zoneinfo/")[-1]
    except Exception:
        pass

    local_dt = datetime.now().astimezone()
    tz_abbrev = local_dt.tzname() or ""

    if iana_tz and tz_abbrev and tz_abbrev != iana_tz:
        return f"{iana_tz} ({tz_abbrev})"
    if iana_tz:
        return iana_tz
    if tz_abbrev:
        return tz_abbrev
    return "UTC"


def _detect_screen_for_x(target_x: float) -> str:
    screens = NSScreen.screens()
    total_screens = len(screens)
    for idx, s in enumerate(screens, 1):
        f = s.frame()
        if f.origin.x <= target_x <= (f.origin.x + f.size.width):
            s_name = (
                s.localizedName()
                if hasattr(s, "localizedName") and s.localizedName()
                else f"{int(f.size.width)}x{int(f.size.height)}"
            )
            return (
                f"Screen {idx} ({s_name})"
                if total_screens > 1
                else f"Screen 1 ({s_name})"
            )
    return "Screen 1"


def _get_copy_context() -> str:
    """
    Captures frontmost application, active screen based on window bounds,
    and Chrome tab/URL metadata. Fails safely without blocking.
    """
    try:
        ws = NSWorkspace.sharedWorkspace()
        app = ws.frontmostApplication()
        app_name = app.localizedName() if app else "Unknown"
        bundle_id = app.bundleIdentifier() if app else ""
        pid = app.processIdentifier() if app else None

        window_x = None
        parts = []

        # If Google Chrome is frontmost, extract Window ID, Tab Index, URL, and Window Bounds
        if bundle_id == "com.google.Chrome" or "Chrome" in app_name:
            script = """tell application "Google Chrome"
    if (count of windows) > 0 then
        set w to front window
        set t to active tab of w
        set tabIdx to active tab index of w
        set b to bounds of w
        return (get id of w as string) & "|||" & (tabIdx as string) & "|||" & (get URL of t) & "|||" & (item 1 of b as string) & "|||" & (item 3 of b as string)
    end if
end tell"""
            try:
                proc = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=1.0,
                )
                out = proc.stdout.strip()
                if out and "|||" in out:
                    chrome_parts = out.split("|||")
                    if len(chrome_parts) >= 3:
                        w_id, t_idx, url = (
                            chrome_parts[0],
                            chrome_parts[1],
                            chrome_parts[2],
                        )
                        parts.append(f"{app_name} (Window: {w_id}, Tab: {t_idx})")
                        if len(chrome_parts) >= 5:
                            try:
                                left = float(chrome_parts[3])
                                right = float(chrome_parts[4])
                                window_x = (left + right) / 2.0
                            except Exception:
                                pass
                        if url:
                            parts.append(f"URL: {url}")
            except Exception:
                pass

        # For non-Chrome apps or if Chrome bounds failed, query Quartz for front window bounds
        if window_x is None and pid is not None:
            try:
                w_list = Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionOnScreenOnly
                    | Quartz.kCGWindowListExcludeDesktopElements,
                    Quartz.kCGNullWindowID,
                )
                for w in w_list:
                    if (
                        w.get(Quartz.kCGWindowOwnerPID) == pid
                        and w.get(Quartz.kCGWindowLayer) == 0
                    ):
                        b = w.get(Quartz.kCGWindowBounds)
                        if b and b.get("Width", 0) > 50 and b.get("Height", 0) > 50:
                            window_x = b.get("X", 0) + (b.get("Width", 0) / 2.0)
                            break
            except Exception:
                pass

        # Fallback to mouse cursor location if window bounds couldn't be resolved
        if window_x is None:
            try:
                window_x = NSEvent.mouseLocation().x
            except Exception:
                window_x = 0.0

        screen_desc = _detect_screen_for_x(window_x)

        if not parts:
            if bundle_id:
                parts.append(f"{app_name} ({bundle_id})")
            else:
                parts.append(app_name)
            parts.append(screen_desc)
        else:
            # Place screen directly after the app/window/tab descriptor
            parts.insert(1, screen_desc)

        joined_parts = " | ".join(parts)
        return f"[Source: {joined_parts}]"
    except Exception as exc:
        print(f"⚠️ Copy context retrieval error: {exc}")
        return ""


class Recorder:
    def __init__(self):
        self.is_recording = False
        self.device_index: int | None = None
        self.device_name: str = "System Default"
        self._stream_using_target: bool = True
        self._session_max_peak: float = 0.0
        self._session_total_samples: int = 0

        self._audio_data: list[np.ndarray] = []
        self._clipboard_history: list[dict] = []
        self._last_copied: str | None = None
        self._last_change_count: int = 0
        self._target_app = ""

        self._stream: sd.InputStream | None = None

        self._audio_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._transcript_lock = threading.Lock()

        # MLX inference is serialised. Live and final inference must never overlap.
        self._mlx_lock = threading.RLock()

        self._stop_event = threading.Event()
        self._session_id = 0

        self._noise_samples: list[float] = []
        self._noise_gate = DEFAULT_GATE_PEAK
        self._calibration_deadline = 0.0
        self._last_voice_time = 0.0
        self._meter_volume = 0.0

        self._start_wall_time: datetime | None = None
        self._start_perf_time = 0.0
        self._stop_wall_time: datetime | None = None
        self._stop_perf_time = 0.0

        self._live_transcript = ""

        # >>> PATCH 2025-08-23 (speech-lab): selectable final-transcription mode.
        #   "full" = one clean full-audio pass on stop (default; fixes
        #            duplicated/dropped phrases from stitching two passes)
        #   "fast" = legacy last-3s slice + stitch (original behaviour)
        # Revert anytime: set "final_mode": "fast" in settings.json,
        # or restore recorder.py.backup / git checkout -- recorder.py
        self.final_mode = self._load_final_mode()
        # Samples of session audio already covered by the latest live pass.
        self._live_covered_samples = 0
        # <<< PATCH

        # >>> VOICE-FILTER-ISOLATION: Initialize neural speaker verification
        self.voice_filter = VoiceFilter() if _VOICE_FILTER_AVAILABLE else None
        self.voice_filter_enabled = self._load_voice_filter_setting()
        self._session_target_detected = False
        self.save_audio_recordings, self.audio_format = (
            self._load_audio_recording_settings()
        )
        # <<< VOICE-FILTER-ISOLATION

        # Fetch location in background daemon thread on startup
        # (External lookups removed for 100% offline privacy)

    def set_device(self, device_index: int | None, device_name: str | None = None):
        if self.is_recording:
            raise RuntimeError("Cannot change microphone while recording")

        self.device_index = device_index
        if device_name is not None:
            self.device_name = device_name

    def _refresh_portaudio(self):
        """Cleanly re-initialise PortAudio to purge stale CoreAudio AudioObjectIDs after sleep/wake."""
        try:
            sd._terminate()
        except Exception:
            pass
        try:
            sd._initialize()
        except Exception as exc:
            print(f"⚠️ PortAudio re-initialization warning: {exc}")

    def _resolve_target_device_index(self) -> int | None:
        """Resolve self.device_name to the current PortAudio device index."""
        if not self.device_name or self.device_name == "System Default":
            return None

        try:
            devices = sd.query_devices()
            for index, device in enumerate(devices):
                if (
                    device.get("max_input_channels", 0) >= 1
                    and device.get("name") == self.device_name
                ):
                    return index
        except Exception as exc:
            print(f"⚠️ Failed to query devices for {self.device_name}: {exc}")

        return None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Audio stream status: {status}")

        if not self.is_recording or indata.size == 0:
            return

        chunk = np.asarray(indata, dtype=np.int16).copy()
        peak = float(np.max(np.abs(chunk.astype(np.int32))))
        now = time.monotonic()

        with self._audio_lock:
            self._audio_data.append(chunk)
            if peak > self._session_max_peak:
                self._session_max_peak = peak
            self._session_total_samples += len(chunk)

        with self._state_lock:
            if now < self._calibration_deadline:
                self._noise_samples.append(peak)
                self._update_noise_gate_locked()

            gate = self._noise_gate

            if peak >= gate:
                self._last_voice_time = now

                active_peak = peak - gate
                active_range = max(1.0, MAX_EXPECTED_VOICE_PEAK - gate)
                normalised = min(1.0, active_peak / active_range)

                # Square root gives useful visual movement for normal speech.
                target = float(np.sqrt(normalised))

                # Minimum active value creates an immediate visible burst.
                target = max(0.18, target)

                self._meter_volume += (target - self._meter_volume) * METER_ATTACK

            elif now - self._last_voice_time <= VOICE_HOLD_SECONDS:
                # Briefly release instead of flashing off between syllables.
                self._meter_volume *= 1.0 - METER_RELEASE

            else:
                # A closed gate must return exact zero.
                self._meter_volume = 0.0

    def _update_noise_gate_locked(self):
        if not self._noise_samples:
            self._noise_gate = DEFAULT_GATE_PEAK
            return

        samples = np.asarray(self._noise_samples, dtype=np.float32)

        # The median is resistant to a few accidental clicks during calibration.
        noise_floor = float(np.median(samples))
        adaptive_gate = max(
            noise_floor * NOISE_GATE_MULTIPLIER,
            noise_floor + NOISE_GATE_MARGIN,
        )

        self._noise_gate = float(np.clip(adaptive_gate, MIN_GATE_PEAK, MAX_GATE_PEAK))

    def _monitor_clipboard(self, session_id: int):
        pb = NSPasteboard.generalPasteboard()

        while (
            self.is_recording
            and not self._stop_event.wait(0.5)
            and session_id == self._session_id
        ):
            try:
                current_count = pb.changeCount()
            except Exception:
                current_count = None

            # Detect copy event by changeCount
            has_new_copy = False
            if current_count is not None and current_count != self._last_change_count:
                has_new_copy = True
                self._last_change_count = current_count

            if not has_new_copy:
                continue

            try:
                current = pyperclip.paste()
            except Exception as exc:
                print(f"⚠️ Clipboard read error: {exc}")
                continue

            if isinstance(current, str) and current and current != self._last_copied:
                now_str = datetime.now().strftime("%H:%M:%S")
                source_meta = _get_copy_context()
                self._clipboard_history.append(
                    {
                        "content": current,
                        "time_str": now_str,
                        "source_meta": source_meta,
                    }
                )
                self._last_copied = current
                meta_log = f" from {source_meta}" if source_meta else ""
                print(f"📋 Captured copy at {now_str}{meta_log}: {current[:40]}...")

    def _get_frontmost_app(self) -> str:
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of '
                    "first process whose frontmost is true",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip()
        except Exception as exc:
            print(f"⚠️ Could not read frontmost application: {exc}")
            return ""

    def _paste_to_app(self, app_name: str):
        if not app_name:
            return

        escaped_app_name = app_name.replace("\\", "\\\\").replace('"', '\\"')

        script = f'''
        tell application "{escaped_app_name}" to activate
        delay 0.3
        tell application "System Events" to keystroke "v" using command down
        '''

        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            print(f"⚠️ Paste error: {exc}")

    def _audio_snapshot(self) -> np.ndarray | None:
        with self._audio_lock:
            if not self._audio_data:
                return None

            chunks = list(self._audio_data)

        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _write_temp_wav(audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tmp:
            temp_path = tmp.name

        wav_write(temp_path, SAMPLE_RATE, audio)
        return temp_path

    @staticmethod
    def _clean_transcript(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())

        if cleaned.casefold() in SILENT_HALLUCINATIONS:
            return ""

        return cleaned

    @staticmethod
    def _merge_transcripts(live_text: str, slice_text: str) -> str:
        if not live_text:
            return slice_text
        if not slice_text:
            return live_text
        import re

        def normalize(t):
            return re.sub(r"[^a-zA-Z0-9]", "", t.lower())

        lw = live_text.split()
        sw = slice_text.split()
        lw_norm = [normalize(w) for w in lw]
        sw_norm = [normalize(w) for w in sw]

        # Only search for overlaps in the last 20 words to prevent aggressive front-truncation
        search_start = max(0, len(lw) - 20)

        max_overlap = 0
        best_i = len(lw)
        best_j = 0

        for i in range(search_start, len(lw)):
            for j in range(len(sw)):
                k = 0
                while (
                    i + k < len(lw)
                    and j + k < len(sw)
                    and lw_norm[i + k] == sw_norm[j + k]
                    and lw_norm[i + k] != ""
                ):
                    k += 1
                if k > max_overlap:
                    # An overlap is only valid if it reaches the absolute end of the live transcript,
                    # OR if it's a strongly confident match of 3+ words.
                    if (i + k == len(lw)) or (k >= 3):
                        max_overlap = k
                        best_i = i
                        best_j = j

        if max_overlap > 0:
            return " ".join(lw[:best_i] + sw[best_j:])
        else:
            return live_text + " " + slice_text

    def _get_model_repo(self) -> str:
        import json

        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                return settings.get("model_name", "mlx-community/whisper-base-mlx")
        except Exception:
            pass
        return "mlx-community/whisper-base-mlx"

    # >>> PATCH 2025-08-23 (speech-lab): bounded splice for smart mode.
    # The tail slice begins slightly before the coverage watermark, so up to
    # max_overlap_words of run-in may repeat what the live pass already heard.
    # Match that short suffix/prefix exactly and splice; otherwise plain join.
    @staticmethod
    def _merge_tail(live_text: str, tail_text: str, max_overlap_words: int = 3) -> str:
        import re

        def norm(token: str) -> str:
            return re.sub("[^a-zA-Z0-9]", "", token.lower())

        if not live_text:
            return tail_text
        if not tail_text:
            return live_text

        live_words = live_text.split()
        tail_words = tail_text.split()
        live_norm = [norm(w) for w in live_words]
        tail_norm = [norm(w) for w in tail_words]

        best = 0
        top = min(max_overlap_words, len(live_norm), len(tail_norm))
        for k in range(top, 0, -1):
            segment = live_norm[-k:]
            if all(segment) and segment == tail_norm[:k]:
                best = k
                break

        if best:
            merged = live_words[: len(live_words) - best] + tail_words
            return " ".join(merged)
        return live_text + " " + tail_text

    # <<< PATCH

    # >>> PATCH 2025-08-23 (speech-lab): read final_mode from settings.json.
    # Defaults to "smart" when missing or unreadable.
    def _load_final_mode(self) -> str:
        import json

        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    mode = str(json.load(f).get("final_mode", "smart"))
                    return mode if mode in ("smart", "full", "fast") else "smart"
        except Exception:
            pass
        return "smart"

    # <<< PATCH

    # >>> VOICE-FILTER-ISOLATION: Voice filter settings and reload helpers
    def _load_voice_filter_setting(self) -> bool:
        """Reads voice_filter_enabled from settings.json (defaults to True)."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return bool(data.get("voice_filter_enabled", True))
        except Exception:
            pass
        return True

    def _load_audio_recording_settings(self) -> tuple[bool, str]:
        """Reads save_audio_recordings and audio_format from settings.json."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    save = bool(data.get("save_audio_recordings", True))
                    fmt = str(data.get("audio_format", "flac")).lower()
                    if fmt not in ("flac", "wav"):
                        fmt = "flac"
                    return save, fmt
        except Exception:
            pass
        return True, "flac"

    def reload_voice_filter(self):
        """Reloads the target voice profile from disk after calibration."""
        if self.voice_filter is not None:
            self.voice_filter.load_profile()

    # <<< VOICE-FILTER-ISOLATION

    def _transcribe_audio(self, audio: np.ndarray, is_tail: bool = False) -> str:
        # >>> VOICE-FILTER-ISOLATION: Verify target speaker before running Whisper inference
        # If a voice profile is enrolled and active, foreign voices (podcasts, YouTube, iPhone) are rejected
        if (
            self.voice_filter_enabled
            and self.voice_filter is not None
            and self.voice_filter.enrolled_embedding is not None
        ):
            filtered_audio, has_target, peak_score = self.voice_filter.filter_utterance(
                audio, is_tail=is_tail
            )
            if not has_target:
                print(
                    f"🛡️ [VoiceFilter] REJECTED foreign speaker (peak similarity: {peak_score:.3f} < threshold {self.voice_filter.threshold:.2f}). Skipping Whisper inference."
                )
                return ""
            else:
                self._session_target_detected = True
                print(
                    f"🛡️ [VoiceFilter] ACCEPTED target speaker (peak similarity: {peak_score:.3f} >= {self.voice_filter.threshold:.2f})"
                )
                audio = filtered_audio
        # <<< VOICE-FILTER-ISOLATION

        temp_path = self._write_temp_wav(audio)

        try:
            with self._mlx_lock:
                result = mlx_whisper.transcribe(
                    temp_path,
                    path_or_hf_repo=self._get_model_repo(),
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                )

            return self._clean_transcript(result.get("text", ""))

        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"⚠️ Temporary WAV cleanup error: {exc}")

    def _run_live_transcription_loop(self, session_id: int):
        print("🎙️ Five-second live transcription worker active")

        while not self._stop_event.wait(LIVE_INTERVAL_SECONDS):
            if not self.is_recording or session_id != self._session_id:
                return

            audio = self._audio_snapshot()
            if audio is None:
                continue

            minimum_samples = int(SAMPLE_RATE * MIN_TRANSCRIPTION_SECONDS)
            if len(audio) < minimum_samples:
                continue

            try:
                text = self._transcribe_audio(audio)

                # Ignore a result belonging to an old recording session.
                if session_id != self._session_id:
                    return

                with self._transcript_lock:
                    self._live_transcript = text
                    # >>> PATCH 2025-08-23 (speech-lab): advance coverage
                    # watermark so stop() only transcribes the remainder.
                    self._live_covered_samples = int(len(audio))
                    # <<< PATCH

                if text:
                    print(f"📝 Live transcript: {text}")

            except Exception as exc:
                # The worker remains alive after an individual inference error.
                print(f"⚠️ Live transcription error ({type(exc).__name__}): {exc}")

    def start(self):
        if self.is_recording:
            return

        self.final_mode = self._load_final_mode()
        self._session_target_detected = False
        self.save_audio_recordings, self.audio_format = (
            self._load_audio_recording_settings()
        )

        self._target_app = self._get_frontmost_app()
        self._session_id += 1
        session_id = self._session_id

        with self._audio_lock:
            self._audio_data = []
            self._session_max_peak = 0.0
            self._session_total_samples = 0

        self._clipboard_history = []

        try:
            self._last_change_count = NSPasteboard.generalPasteboard().changeCount()
        except Exception:
            self._last_change_count = 0

        self._last_copied = None

        with self._transcript_lock:
            self._live_transcript = ""
            # >>> PATCH 2025-08-23 (speech-lab): fresh session, no coverage.
            self._live_covered_samples = 0
            # <<< PATCH

        with self._state_lock:
            self._noise_samples = []
            self._noise_gate = DEFAULT_GATE_PEAK
            self._calibration_deadline = time.monotonic() + CALIBRATION_SECONDS
            self._last_voice_time = 0.0
            self._meter_volume = 0.0

        self._start_wall_time = datetime.now()
        self._start_perf_time = time.perf_counter()

        self._stop_event.clear()
        self.is_recording = True

        # Resolve target device dynamically before opening stream
        if self.device_name and self.device_name != "System Default":
            resolved_index = self._resolve_target_device_index()
            if resolved_index is not None:
                self.device_index = resolved_index
            else:
                self._refresh_portaudio()
                resolved_index = self._resolve_target_device_index()
                if resolved_index is not None:
                    self.device_index = resolved_index

        target_dev = self.device_index
        self._stream_using_target = True

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=target_dev,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()

        except Exception as primary_error:
            print(
                f"⚠️ Primary microphone '{self.device_name}' (device {target_dev}) failed: {primary_error}"
            )
            print("🔄 Attempting PortAudio re-initialization and retry...")

            recovered = False
            self._refresh_portaudio()
            refreshed_dev = self._resolve_target_device_index()
            if refreshed_dev is not None:
                self.device_index = refreshed_dev
                try:
                    self._stream = sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        device=refreshed_dev,
                        callback=self._audio_callback,
                        blocksize=1024,
                    )
                    self._stream.start()
                    recovered = True
                    print(
                        f"✅ Successfully recovered microphone '{self.device_name}' on device {refreshed_dev}!"
                    )
                except Exception as retry_err:
                    print(
                        f"⚠️ Retry on recovered device {refreshed_dev} failed: {retry_err}"
                    )

            if not recovered:
                self._stream_using_target = False
                print(
                    f"⚠️ Selected microphone '{self.device_name}' unavailable. Falling back to system default..."
                )
                try:
                    self._stream = sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        device=None,
                        callback=self._audio_callback,
                        blocksize=1024,
                    )
                    self._stream.start()
                except Exception:
                    self.is_recording = False
                    self._stop_event.set()
                    self._stream = None
                    raise

        # Workers start only after the stream is successfully open.
        threading.Thread(
            target=self._monitor_clipboard,
            args=(session_id,),
            daemon=True,
            name=f"ClipboardMonitor-{session_id}",
        ).start()

        threading.Thread(
            target=self._run_live_transcription_loop,
            args=(session_id,),
            daemon=True,
            name=f"LiveTranscriber-{session_id}",
        ).start()

        opened_desc = (
            self.device_name
            if self._stream_using_target
            else "System Default (Fallback)"
        )
        print(f"🎙️ Successfully opened microphone: {opened_desc}")

    def stop(
        self,
        on_done: Callable[[str], None] | None = None,
    ):
        if not self.is_recording:
            return

        print("✅ Stopping — processing final transcription on MLX GPU")

        self._stop_wall_time = datetime.now()
        self._stop_perf_time = time.perf_counter()
        start_wall = self._start_wall_time
        stop_wall = self._stop_wall_time
        start_perf = self._start_perf_time
        stop_perf = self._stop_perf_time

        self.is_recording = False
        self._stop_event.set()

        stream_to_close = self._stream
        self._stream = None

        if stream_to_close is not None:

            def _close_stream():
                try:
                    stream_to_close.stop()
                except Exception as exc:
                    print(f"⚠️ Stream stop error: {exc}")
                finally:
                    try:
                        stream_to_close.close()
                    except Exception as exc:
                        print(f"⚠️ Stream close error: {exc}")

            threading.Thread(
                target=_close_stream,
                daemon=True,
                name=f"StreamCleanup-{self._session_id}",
            ).start()

        with self._state_lock:
            self._meter_volume = 0.0
            self._last_voice_time = 0.0

        audio_snapshot = self._audio_snapshot()
        clipboard_snapshot = list(self._clipboard_history)
        target_app = self._target_app
        session_id = self._session_id
        mic_requested = self.device_name
        mic_actual = (
            self.device_name
            if self._stream_using_target
            else "System Default (Fallback)"
        )
        fallback_used = not self._stream_using_target

        def process_final():
            final_text = ""
            spoken_text = ""
            status = "SUCCESS"
            error_str = None
            try:
                if audio_snapshot is not None and len(audio_snapshot):
                    with self._mlx_lock:
                        with self._transcript_lock:
                            live_text = self._live_transcript
                            covered = int(self._live_covered_samples)

                        # >>> PATCH 2025-08-23 (speech-lab): mode-switchable pass.
                        # "smart" (default): the live worker already transcribed most
                        #   of this audio. Only the UNCOVERED remainder is sent to
                        #   whisper, then spliced onto the live transcript with a
                        #   bounded exact match. Near-instant paste, no double pass.
                        # "full": transcribe ENTIRE recording once — slowest, most
                        #   context; use to benchmark the other modes.
                        # "fast": legacy last-3s slice + fuzzy stitch (original).
                        if self.final_mode == "smart":
                            duration_samples = int(len(audio_snapshot))
                            uncovered = duration_samples - covered
                            min_tail_samples = int(SAMPLE_RATE * 0.20)
                            stale_limit_samples = int(SAMPLE_RATE * 45)

                            if (
                                not live_text
                                or covered <= 0
                                or uncovered > stale_limit_samples
                            ):
                                print(
                                    "🧪 Final mode SMART: no usable live transcript"
                                    " — clean full pass..."
                                )
                                spoken_text = self._transcribe_audio(audio_snapshot)
                            elif uncovered < min_tail_samples:
                                print(
                                    "🧪 Final mode SMART: live transcript already"
                                    " covers the audio — instant paste."
                                )
                                spoken_text = live_text
                            else:
                                tail_start = max(0, covered - int(SAMPLE_RATE * 0.30))
                                tail_audio = audio_snapshot[tail_start:]
                                remaining_seconds = len(tail_audio) / SAMPLE_RATE
                                print(
                                    f"🧪 Final mode SMART: transcribing remaining"
                                    f" {remaining_seconds:.2f}s of audio..."
                                )
                                tail_text = self._transcribe_audio(
                                    tail_audio,
                                    is_tail=self._session_target_detected,
                                )
                                spoken_text = self._merge_tail(live_text, tail_text)
                                print(f"✅ Smart Merged Text: {spoken_text}")

                        elif self.final_mode == "fast":
                            tail_samples = int(SAMPLE_RATE * 3.0)
                            if len(audio_snapshot) > tail_samples:
                                print(
                                    "⚡ Instant Smart Chunking: Transcribing only the last 3.0 seconds..."
                                )
                                tail_audio = audio_snapshot[-tail_samples:]
                                tail_text = self._transcribe_audio(tail_audio)
                                spoken_text = self._merge_transcripts(
                                    live_text, tail_text
                                )
                                print(f"✅ Stitched Text: {spoken_text}")
                            else:
                                print(
                                    "⚙️ Audio under 3 seconds, running full transcription..."
                                )
                                spoken_text = self._transcribe_audio(audio_snapshot)

                        else:
                            print(
                                "🧪 Final mode FULL: clean single pass over entire recording..."
                            )
                            spoken_text = self._transcribe_audio(audio_snapshot)
                            print(f"✅ Full Pass Text: {spoken_text}")
                        # <<< PATCH

                        if spoken_text:
                            if start_wall is not None and stop_wall is not None:
                                duration_seconds = max(0.0, stop_perf - start_perf)
                                start_date = start_wall.strftime("%Y-%m-%d")
                                start_time = start_wall.strftime("%H:%M:%S.%f")[:-3]
                                stop_time = stop_wall.strftime("%H:%M:%S.%f")[:-3]

                                if start_wall.date() == stop_wall.date():
                                    time_segment = (
                                        f"{start_date} Started: {start_time} | "
                                        f"Finished: {stop_time}"
                                    )
                                else:
                                    stop_date = stop_wall.strftime("%Y-%m-%d")
                                    time_segment = (
                                        f"{start_date} Started: {start_time} | "
                                        f"Finished: {stop_date} {stop_time}"
                                    )

                                if duration_seconds < 60:
                                    dur_str = f"{duration_seconds:.2f}s"
                                else:
                                    mins = int(duration_seconds // 60)
                                    secs = duration_seconds % 60
                                    dur_str = (
                                        f"{mins}m {secs:.2f}s ({duration_seconds:.2f}s)"
                                    )

                                timezone_str = _get_system_timezone()
                                tz_part = (
                                    f"Timezone: {timezone_str} | "
                                    if timezone_str
                                    else ""
                                )
                                header = (
                                    f"[TRANSCRIPTION RECORDING]\n"
                                    f"{tz_part}{time_segment} | Duration: {dur_str}\n\n"
                                )
                                final_text = header + spoken_text
                            else:
                                final_text = spoken_text
                        else:
                            final_text = ""
                else:
                    print("⚠️ No audio was recorded")

                if clipboard_snapshot:
                    copied_section = "\n\n[COPIED ITEMS]\n"
                    seen_items: dict[str, int] = {}
                    seen_sources: dict[str, int] = {}

                    for index, entry in enumerate(
                        clipboard_snapshot,
                        start=1,
                    ):
                        if isinstance(entry, dict):
                            content = entry.get("content", "")
                            time_str = entry.get("time_str", "")
                            source_meta = entry.get("source_meta", "")
                        else:
                            content = str(entry)
                            time_str = ""
                            source_meta = ""

                        header_time = f" ({time_str})" if time_str else ""
                        stripped = content.strip()

                        # 1. Content deduplication & delta detection
                        if stripped in seen_items:
                            orig_idx = seen_items[stripped]
                            body = f"[Duplicate of Copied Item {orig_idx}]"
                        else:
                            delta_body = None
                            for prev_stripped, prev_idx in seen_items.items():
                                if len(prev_stripped) < 30 or len(stripped) < 30:
                                    continue

                                # Extension: current item contains an earlier item and adds new text
                                if prev_stripped in stripped:
                                    if stripped.startswith(prev_stripped):
                                        remainder = stripped[
                                            len(prev_stripped) :
                                        ].strip()
                                        delta_body = f"[Extension of Copied Item {prev_idx} — Appended text:]\n{remainder}"
                                        break
                                    elif stripped.endswith(prev_stripped):
                                        remainder = stripped[
                                            : -len(prev_stripped)
                                        ].strip()
                                        delta_body = f"[Extension of Copied Item {prev_idx} — Prepended text:]\n{remainder}"
                                        break

                                # Excerpt: current item is a sub-selection of an earlier item
                                elif stripped in prev_stripped:
                                    words = stripped.split()
                                    total_words = len(words)
                                    lines = [
                                        l for l in stripped.splitlines() if l.strip()
                                    ]
                                    line_count = len(lines)

                                    # Short snippets (<= 16 words): pass through directly without compression
                                    if total_words <= 16:
                                        delta_body = (
                                            f"[Excerpt from Copied Item {prev_idx} ({line_count} lines):]\n"
                                            f"{content}"
                                        )
                                    else:
                                        # Substantial snippets (> 16 words): extract unambiguous 8-word anchors
                                        start_anchor = " ".join(words[:8]) + "..."
                                        end_anchor = "..." + " ".join(words[-8:])
                                        delta_body = (
                                            f"[Excerpt from Copied Item {prev_idx} ({line_count} lines, {total_words} words):]\n"
                                            f'Start: "{start_anchor}"\n'
                                            f'End:   "{end_anchor}"'
                                        )
                                    break

                            if delta_body:
                                body = delta_body
                            else:
                                body = content
                            seen_items[stripped] = index

                        # 2. Source metadata deduplication
                        if source_meta:
                            if source_meta in seen_sources:
                                orig_src_idx = seen_sources[source_meta]
                                rendered_source = (
                                    f"[Source: Same as Copied Item {orig_src_idx}]"
                                )
                            else:
                                seen_sources[source_meta] = index
                                rendered_source = source_meta
                            meta_line = f"{rendered_source}\n"
                        else:
                            meta_line = ""

                        copied_section += f"--- Copied Item {index}{header_time} ---\n{meta_line}{body}\n\n"

                    if final_text:
                        final_text += copied_section
                    else:
                        final_text = copied_section.strip()

                final_text = final_text.strip()

                # Do not paste an empty result.
                if final_text:
                    pyperclip.copy(final_text)
                    logger.info(f"🔥 Pasting {len(final_text)} chars into {target_app}")
                    print(f"\n--- OUTPUT ---\n{final_text}\n--------------\n")
                    self._paste_to_app(target_app)
                else:
                    if self._session_total_samples == 0:
                        status = "EMPTY_AUDIO"
                    elif self._session_max_peak < 50.0:
                        status = "SILENT_AUDIO"
                    else:
                        status = "EMPTY_TRANSCRIPT"
                    logger.warning(
                        f"⚠️ Final transcription was empty (status: {status})"
                    )
                    print("⚠️ Final transcription was empty")

            except Exception as exc:
                status = "ERROR"
                error_str = f"{type(exc).__name__}: {exc}"
                logger.error(f"⚠️ Final transcription error: {error_str}", exc_info=True)
                print(f"⚠️ Final transcription error ({type(exc).__name__}): {exc}")

            finally:
                dur_seconds = (
                    max(0.0, stop_perf - start_perf)
                    if start_perf and stop_perf
                    else 0.0
                )
                audio_file_rel = None
                if (
                    self.save_audio_recordings
                    and audio_snapshot is not None
                    and len(audio_snapshot) > 0
                ):
                    try:
                        import soundfile as sf

                        now_dt = start_wall or datetime.now()
                        date_folder = now_dt.strftime("%Y-%m-%d")
                        time_tag = now_dt.strftime("%H%M%S")
                        recordings_dir = (
                            PROJECT_ROOT / "logs" / "recordings" / date_folder
                        )
                        recordings_dir.mkdir(parents=True, exist_ok=True)
                        ext = (
                            self.audio_format
                            if self.audio_format in ("flac", "wav")
                            else "flac"
                        )
                        audio_filename = f"session_{session_id}_{time_tag}.{ext}"
                        audio_full_path = recordings_dir / audio_filename
                        audio_file_rel = (
                            f"logs/recordings/{date_folder}/{audio_filename}"
                        )

                        if ext == "flac":
                            sf.write(
                                str(audio_full_path),
                                audio_snapshot,
                                SAMPLE_RATE,
                                format="FLAC",
                            )
                        else:
                            wav_write(
                                str(audio_full_path),
                                SAMPLE_RATE,
                                audio_snapshot,
                            )
                        logger.info(f"🎙️ Preserved audio recording: {audio_file_rel}")
                    except Exception as save_err:
                        logger.error(f"⚠️ Failed to save session audio: {save_err}")

                session_record = {
                    "session_id": session_id,
                    "audio_file": audio_file_rel,
                    "timestamp": (
                        start_wall.isoformat(timespec="milliseconds")
                        if start_wall
                        else datetime.now().isoformat(timespec="milliseconds")
                    ),
                    "duration_seconds": round(dur_seconds, 2),
                    "target_app": target_app,
                    "mic_requested": mic_requested,
                    "mic_actual": mic_actual,
                    "fallback_used": fallback_used,
                    "audio_metrics": {
                        "peak_amplitude": round(float(self._session_max_peak), 1),
                        "noise_gate": round(float(self._noise_gate), 1),
                        "total_samples": int(self._session_total_samples),
                        "is_silent": bool(self._session_max_peak < 50.0),
                    },
                    "spoken_text": spoken_text.strip() if spoken_text else "",
                    "copied_items": [
                        {
                            "time_str": item.get("time_str", ""),
                            "source_meta": item.get("source_meta", ""),
                            "char_length": len(item.get("content", "")),
                            "content": item.get("content", ""),
                        }
                        for item in clipboard_snapshot
                    ],
                    "copied_items_count": len(clipboard_snapshot),
                    "final_output": final_text,
                    "status": status,
                    "error": error_str,
                }
                log_session(session_record)

                # Do not let an old session update a newer session.
                if on_done is not None and session_id == self._session_id:
                    on_done(final_text)

        threading.Thread(
            target=process_final,
            daemon=True,
            name=f"FinalTranscriber-{session_id}",
        ).start()

    def get_volume(self) -> float:
        """
        Return a gated visual volume from 0.0 to 1.0.

        A closed gate returns exactly 0.0, allowing the overlay to become
        completely stationary.
        """
        if not self.is_recording:
            return 0.0

        with self._state_lock:
            if time.monotonic() - self._last_voice_time > VOICE_HOLD_SECONDS:
                self._meter_volume = 0.0
                return 0.0

            value = float(np.clip(self._meter_volume, 0.0, 1.0))

        if value < 0.01:
            return 0.0

        return value

    def get_live_transcript(self) -> str:
        with self._transcript_lock:
            return self._live_transcript

    def get_noise_gate(self) -> float:
        with self._state_lock:
            return self._noise_gate
