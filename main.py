"""
main.py — WhisperStack menu bar application.

Architecture:
- QApplication remains alive when the HUD closes
- WhisperStackTray owns the tray icon, recorder, HUD, and hotkey
- pynput events cross into Qt through a signal
- Recorder completion crosses into Qt through a signal
"""

from __future__ import annotations

import os

from pathlib import Path

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"


import json
import os
import subprocess
import sys
import threading

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import objc
from AppKit import (
    NSWorkspace,
    NSWorkspaceDidWakeNotification,
    NSWorkspaceScreensDidWakeNotification,
)
from Foundation import NSObject
import sounddevice as sd
from pynput import keyboard as pynput_keyboard
from PyQt6.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
)

from overlay import RecordingOverlay
from recorder import Recorder
from app_logger import LOGS_DIR, logger, setup_logging

setup_logging()


PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = str(PROJECT_ROOT / "settings.json")


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {}

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️ Could not load settings: {exc}")
        return {}


def save_settings(settings: dict):
    try:
        settings_directory = os.path.dirname(SETTINGS_FILE)
        os.makedirs(settings_directory, exist_ok=True)

        temporary_file = f"{SETTINGS_FILE}.tmp"

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(settings, file, indent=4)

        os.replace(temporary_file, SETTINGS_FILE)

    except Exception as exc:
        print(f"⚠️ Could not save settings: {exc}")


def play_system_sound(sound_name: str, volume: float = 1.2):
    sound_path = f"/System/Library/Sounds/{sound_name}.aiff"

    if not os.path.exists(sound_path):
        return

    try:
        subprocess.Popen(
            ["afplay", "-v", str(volume), sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"⚠️ Could not play system sound: {exc}")


class HotkeyBridge(QObject):
    triggered = pyqtSignal()

    def __init__(self, hotkey_str: str, parent=None):
        super().__init__(parent)
        self.hotkey_str = hotkey_str
        self._listener = None
        self._thread = None
        self._carbon_registered = False
        self._carbon_hotkey_ref = None
        self._carbon_handler_ref = None
        self._last_trigger_time = 0.0

    def start(self):
        try:
            if self._start_carbon():
                print("✅ Registered native Carbon global hotkey.")
                self._carbon_registered = True
                return
        except Exception as exc:
            print(
                f"⚠️ Failed to initialize native Carbon hotkey: {exc}. Falling back to pynput."
            )

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="HotkeyThread",
        )
        self._thread.start()

    def _start_carbon(self) -> bool:
        import ctypes

        MAP_KEYS = {
            "a": 0,
            "s": 1,
            "d": 2,
            "f": 3,
            "h": 4,
            "g": 5,
            "z": 6,
            "x": 7,
            "c": 8,
            "v": 9,
            "b": 11,
            "q": 12,
            "w": 13,
            "e": 14,
            "r": 15,
            "y": 16,
            "t": 17,
            "1": 18,
            "2": 19,
            "3": 20,
            "4": 21,
            "6": 22,
            "5": 23,
            "9": 25,
            "7": 26,
            "8": 28,
            "0": 29,
            "o": 31,
            "u": 32,
            "i": 34,
            "p": 35,
            "l": 37,
            "j": 38,
            "k": 40,
            "n": 45,
            "m": 46,
            "space": 49,
            "tab": 48,
            "return": 36,
            "enter": 36,
            "up": 126,
            "down": 125,
            "left": 123,
            "right": 124,
            "f1": 122,
            "f2": 120,
            "f3": 99,
            "f4": 118,
            "f5": 96,
            "f6": 97,
            "f7": 98,
            "f8": 100,
            "f9": 101,
            "f10": 109,
            "f11": 103,
            "f12": 111,
        }

        parts = [p.strip().lower() for p in self.hotkey_str.split("+")]
        modifiers = 0
        key_code = None

        for part in parts:
            part_clean = part.replace("<", "").replace(">", "")
            if part_clean in ("cmd", "command"):
                modifiers |= 256
            elif part_clean in ("shift",):
                modifiers |= 512
            elif part_clean in ("alt", "option"):
                modifiers |= 2048
            elif part_clean in ("ctrl", "control"):
                modifiers |= 4096
            else:
                key_code = MAP_KEYS.get(part_clean)
                if key_code is None and len(part_clean) == 1:
                    key_code = MAP_KEYS.get(part_clean)

        if key_code is None:
            print(f"⚠️ Carbon parsing failed for hotkey: {self.hotkey_str}")
            return False

        class EventHotKeyID(ctypes.Structure):
            _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]

        class EventTypeSpec(ctypes.Structure):
            _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]

        carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")

        carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        carbon.GetApplicationEventTarget.argtypes = []

        EventHandlerProcType = ctypes.CFUNCTYPE(
            ctypes.c_int32,  # OSStatus
            ctypes.c_void_p,  # EventHandlerCallRef
            ctypes.c_void_p,  # EventRef
            ctypes.c_void_p,  # void *
        )

        carbon.InstallEventHandler.restype = ctypes.c_int32
        carbon.InstallEventHandler.argtypes = [
            ctypes.c_void_p,
            EventHandlerProcType,
            ctypes.c_uint32,
            ctypes.POINTER(EventTypeSpec),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        carbon.RegisterEventHotKey.restype = ctypes.c_int32
        carbon.RegisterEventHotKey.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            EventHotKeyID,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        def _carbon_callback(handler_call, event, user_data):
            self._on_activate()
            return 0

        self._carbon_callback_ref = EventHandlerProcType(_carbon_callback)
        app_target = carbon.GetApplicationEventTarget()

        spec = EventTypeSpec()
        spec.eventClass = 0x6B657962  # 'keyb'
        spec.eventKind = 5  # kEventHotKeyPressed

        handler_ref = ctypes.c_void_p()
        res_install = carbon.InstallEventHandler(
            app_target,
            self._carbon_callback_ref,
            1,
            ctypes.byref(spec),
            None,
            ctypes.byref(handler_ref),
        )
        if res_install != 0:
            print(f"⚠️ Carbon InstallEventHandler failed: {res_install}")
            return False

        self._carbon_handler_ref = handler_ref

        hotkey_id = EventHotKeyID()
        hotkey_id.signature = 0x484B3031  # 'HK01'
        hotkey_id.id = 1

        hotkey_ref = ctypes.c_void_p()
        res_register = carbon.RegisterEventHotKey(
            key_code, modifiers, hotkey_id, app_target, 0, ctypes.byref(hotkey_ref)
        )
        if res_register != 0:
            print(f"⚠️ Carbon RegisterEventHotKey failed: {res_register}")
            return False

        self._carbon_hotkey_ref = hotkey_ref
        return True

    def _run(self):
        try:
            with pynput_keyboard.GlobalHotKeys(
                {self.hotkey_str: self._on_activate}
            ) as listener:
                self._listener = listener
                listener.join()
        except Exception as exc:
            print(f"⚠️ Hotkey listener error: {exc}")

    def _on_activate(self):
        import time

        now = time.monotonic()
        if now - self._last_trigger_time < 0.5:
            return
        self._last_trigger_time = now
        self.triggered.emit()

    def stop(self):
        if self._carbon_registered:
            try:
                import ctypes

                carbon = ctypes.CDLL(
                    "/System/Library/Frameworks/Carbon.framework/Carbon"
                )
                if self._carbon_hotkey_ref is not None:
                    carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
                    carbon.UnregisterEventHotKey.restype = ctypes.c_int32
                    carbon.UnregisterEventHotKey(self._carbon_hotkey_ref)
                    self._carbon_hotkey_ref = None
                if self._carbon_handler_ref is not None:
                    carbon.RemoveEventHandler.argtypes = [ctypes.c_void_p]
                    carbon.RemoveEventHandler.restype = ctypes.c_int32
                    carbon.RemoveEventHandler(self._carbon_handler_ref)
                    self._carbon_handler_ref = None
            except Exception as exc:
                print(f"⚠️ Error stopping Carbon listener: {exc}")
            self._carbon_registered = False
            return

        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.stop()


class HotkeyDialog(QDialog):
    def __init__(self, current: str, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Configure Hotkey — WhisperStack")
        self.setFixedSize(380, 160)

        self.new_hotkey = current

        layout = QVBoxLayout()
        layout.setSpacing(10)

        title = QLabel("<b>Global hotkey</b> (pynput format):")
        layout.addWidget(title)

        hint = QLabel(
            "Examples:  <code><cmd>+<shift>+<space></code>  <code><ctrl>+<alt>+r</code>"
        )
        hint.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(hint)

        self._input = QLineEdit(current)
        self._input.setPlaceholderText("<cmd>+<shift>+<space>")
        layout.addWidget(self._input)

        buttons = QHBoxLayout()

        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")

        save_button.setDefault(True)
        save_button.clicked.connect(self._save)
        cancel_button.clicked.connect(self.reject)

        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        layout.addLayout(buttons)
        self.setLayout(layout)

    def _save(self):
        candidate = self._input.text().strip()

        if candidate:
            self.new_hotkey = candidate

        self.accept()


def make_tray_icon(recording: bool = False) -> QIcon:
    size = 22

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(
        QPainter.RenderHint.Antialiasing,
        True,
    )

    colour = QColor(220, 50, 50) if recording else QColor(255, 255, 255, 210)

    painter.setPen(QPen(colour, 1.6))
    painter.setBrush(QBrush(colour))
    painter.drawRoundedRect(8, 2, 6, 11, 3, 3)

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawArc(
        QRectF(5, 8, 12, 8),
        0,
        -180 * 16,
    )
    painter.drawLine(11, 16, 11, 20)
    painter.drawLine(8, 20, 14, 20)
    painter.end()

    return QIcon(pixmap)


class SystemWakeObserver(NSObject):
    def initWithCallback_(self, callback):
        self = objc.super(SystemWakeObserver, self).init()
        if self is not None:
            self._callback = callback
        return self

    def handleWake_(self, notification):
        if hasattr(self, "_callback") and self._callback:
            self._callback()


class WhisperStackTray(QObject):
    transcription_done = pyqtSignal(str)
    recording_failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        play_system_sound("Hero")

        settings = load_settings()

        self._hotkey_str = settings.get(
            "hotkey",
            "<cmd>+<shift>+s",
        )
        saved_mic = settings.get(
            "mic_name",
            "System Default",
        )

        self._selected_device: int | None = None
        self._selected_device_name = "System Default"

        self._recorder = Recorder()

        self._restore_microphone(saved_mic)

        self._overlay = RecordingOverlay(self._recorder)

        self.transcription_done.connect(self._on_transcription_done)
        self.recording_failed.connect(self._on_recording_failed)

        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(make_tray_icon(False))
        self._tray.setToolTip("WhisperStack — idle")

        self._build_menu()
        self._tray.show()

        self._bridge: HotkeyBridge | None = None
        self._start_hotkey()

        self._setup_wake_observer()

        print(f"🚀 WhisperStack running — hotkey: {self._hotkey_str}")

    def _restore_microphone(self, saved_mic: str):
        if not saved_mic or saved_mic == "System Default":
            self._selected_device = None
            self._selected_device_name = "System Default"
            self._recorder.set_device(None, "System Default")
            return

        try:
            devices = sd.query_devices()

            for index, device in enumerate(devices):
                if (
                    device.get("max_input_channels", 0) >= 1
                    and device.get("name") == saved_mic
                ):
                    self._selected_device = index
                    self._selected_device_name = saved_mic
                    self._recorder.set_device(index, saved_mic)

                    print(f"🔌 Restored microphone: {saved_mic} (device {index})")
                    return

            # If not immediately visible, retry after PortAudio refresh
            try:
                sd._terminate()
                sd._initialize()
                devices = sd.query_devices()
                for index, device in enumerate(devices):
                    if (
                        device.get("max_input_channels", 0) >= 1
                        and device.get("name") == saved_mic
                    ):
                        self._selected_device = index
                        self._selected_device_name = saved_mic
                        self._recorder.set_device(index, saved_mic)
                        print(
                            "🔌 Restored microphone after PortAudio refresh: "
                            f"{saved_mic} (device {index})"
                        )
                        return
            except Exception:
                pass

            print(
                f"⚠️ Saved microphone '{saved_mic}' not currently attached, "
                "keeping preference for auto-recovery"
            )
            self._selected_device = None
            self._selected_device_name = saved_mic
            self._recorder.set_device(None, saved_mic)

        except Exception as exc:
            print(f"⚠️ Microphone discovery error: {exc}")

    def _build_menu(self):
        self._main_menu = QMenu()
        self._main_menu.aboutToShow.connect(self._update_voice_filter_menu)

        self._status_action = QAction("⚪ Idle")
        self._status_action.setEnabled(False)
        self._main_menu.addAction(self._status_action)
        self._main_menu.addSeparator()

        self._mic_menu = QMenu()
        self._mic_menu.aboutToShow.connect(self._rebuild_mic_submenu)
        self._rebuild_mic_submenu()
        self._main_menu.addMenu(self._mic_menu)

        self._hotkey_action = QAction(f"⌨️ Hotkey: {self._hotkey_str}")
        self._hotkey_action.triggered.connect(self._configure_hotkey)
        self._main_menu.addAction(self._hotkey_action)
        self._main_menu.addSeparator()

        # >>> VOICE-FILTER-ISOLATION: Menu actions for speaker verification
        self._voice_filter_action = QAction("🛡️ Voice Filter: Checking...")
        self._voice_filter_action.setCheckable(True)
        self._voice_filter_action.triggered.connect(self._toggle_voice_filter)
        self._main_menu.addAction(self._voice_filter_action)

        self._calibrate_voice_action = QAction("🎙️ Calibrate My Voice...")
        self._calibrate_voice_action.triggered.connect(self._launch_voice_calibration)
        self._main_menu.addAction(self._calibrate_voice_action)
        self._main_menu.addSeparator()
        self._update_voice_filter_menu()
        # <<< VOICE-FILTER-ISOLATION

        self._open_logs_action = QAction("📂 Open Logs Folder")
        self._open_logs_action.triggered.connect(self._open_logs_folder)
        self._main_menu.addAction(self._open_logs_action)
        self._main_menu.addSeparator()

        self._quit_action = QAction("Quit WhisperStack")
        self._quit_action.triggered.connect(self._quit_application)
        self._main_menu.addAction(self._quit_action)

        self._tray.setContextMenu(self._main_menu)

    def _open_logs_folder(self):
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(LOGS_DIR)])
        except Exception as exc:
            logger.error(f"Could not open logs folder: {exc}")

    # >>> VOICE-FILTER-ISOLATION: Voice filter menu updater and calibration launcher
    def _update_voice_filter_menu(self):
        vf = getattr(self._recorder, "voice_filter", None)
        enabled = getattr(self._recorder, "voice_filter_enabled", True)
        has_profile = vf is not None and vf.enrolled_embedding is not None

        if not enabled:
            self._voice_filter_action.setText("🛡️ Voice Filter: Disabled")
            self._voice_filter_action.setChecked(False)
        elif has_profile:
            self._voice_filter_action.setText("🛡️ Voice Filter: Active (Enrolled)")
            self._voice_filter_action.setChecked(True)
        else:
            self._voice_filter_action.setText("🛡️ Voice Filter: No Profile Enrolled")
            self._voice_filter_action.setChecked(False)

    def _toggle_voice_filter(self):
        current = getattr(self._recorder, "voice_filter_enabled", True)
        new_state = not current
        self._recorder.voice_filter_enabled = new_state
        settings = load_settings()
        settings["voice_filter_enabled"] = new_state
        save_settings(settings)
        self._update_voice_filter_menu()
        label = "Enabled" if new_state else "Disabled"
        self._tray.showMessage(
            "WhisperStack",
            f"Voice Filter: {label}",
            QSystemTrayIcon.MessageIcon.Information,
            1500,
        )

    def _launch_voice_calibration(self):
        project_root = Path(__file__).resolve().parent
        enroll_script = str(project_root / "enroll_voice.py")
        python_bin = sys.executable
        apple_script = f'''
        tell application "Terminal"
            activate
            do script "{python_bin} {enroll_script}"
        end tell
        '''
        try:
            subprocess.Popen(["osascript", "-e", apple_script])
        except Exception as exc:
            print(f"⚠️ Could not open Terminal for voice calibration: {exc}")

    # <<< VOICE-FILTER-ISOLATION

    def _rebuild_mic_submenu(self):
        self._mic_menu.clear()
        self._mic_menu.menuAction().setText(f"🎤 Mic: {self._selected_device_name}")

        default_action = QAction(
            "System Default",
            self._mic_menu,
        )
        default_action.setCheckable(True)
        default_action.setChecked(self._selected_device_name == "System Default")
        default_action.triggered.connect(
            lambda checked: self._select_mic(
                None,
                "System Default",
            )
        )

        self._mic_menu.addAction(default_action)
        self._mic_menu.addSeparator()

        try:
            devices = sd.query_devices()
        except Exception:
            try:
                sd._terminate()
                sd._initialize()
                devices = sd.query_devices()
            except Exception as exc:
                print(f"⚠️ Microphone query error: {exc}")
                return

        seen_names = set()
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) < 1:
                continue

            name = device["name"]
            if name in seen_names:
                continue
            seen_names.add(name)

            action = QAction(name, self._mic_menu)
            action.setCheckable(True)
            action.setChecked(self._selected_device_name == name)
            action.triggered.connect(
                lambda checked, i=index, n=name: self._select_mic(i, n)
            )

            self._mic_menu.addAction(action)

    def _select_mic(
        self,
        device_index: int | None,
        name: str,
    ):
        if self._recorder.is_recording:
            self._tray.showMessage(
                "WhisperStack",
                "Stop recording before changing microphone",
                QSystemTrayIcon.MessageIcon.Warning,
                1800,
            )
            return

        try:
            self._recorder.set_device(device_index, name)
        except Exception as exc:
            self._tray.showMessage(
                "WhisperStack",
                str(exc),
                QSystemTrayIcon.MessageIcon.Warning,
                1800,
            )
            return

        self._selected_device = device_index
        self._selected_device_name = name

        settings = load_settings()
        settings["mic_name"] = name
        save_settings(settings)

        self._rebuild_mic_submenu()

        self._tray.showMessage(
            "WhisperStack",
            f"Mic set to: {name}",
            QSystemTrayIcon.MessageIcon.Information,
            1800,
        )

    def _setup_wake_observer(self):
        try:
            self._wake_observer = SystemWakeObserver.alloc().initWithCallback_(
                self._on_system_wake
            )
            center = NSWorkspace.sharedWorkspace().notificationCenter()
            center.addObserver_selector_name_object_(
                self._wake_observer,
                "handleWake:",
                NSWorkspaceDidWakeNotification,
                None,
            )
            center.addObserver_selector_name_object_(
                self._wake_observer,
                "handleWake:",
                NSWorkspaceScreensDidWakeNotification,
                None,
            )
            print("☀️ Registered macOS wake notification observer.")
        except Exception as exc:
            print(f"⚠️ Could not register wake observer: {exc}")

    def _on_system_wake(self):
        print("☀️ macOS wake detected — refreshing audio devices...")
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        self._restore_microphone(self._selected_device_name)
        self._rebuild_mic_submenu()
        QTimer.singleShot(2000, self._on_wake_delayed_refresh)

    def _on_wake_delayed_refresh(self):
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
        self._restore_microphone(self._selected_device_name)
        self._rebuild_mic_submenu()
        print(
            f"☀️ Delayed wake refresh complete. Active mic: {self._selected_device_name} (device {self._selected_device})"
        )

    def _configure_hotkey(self):
        dialog = HotkeyDialog(self._hotkey_str)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._hotkey_str = dialog.new_hotkey
        self._hotkey_action.setText(f"⌨️ Hotkey: {self._hotkey_str}")

        settings = load_settings()
        settings["hotkey"] = self._hotkey_str
        save_settings(settings)

        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None

        self._start_hotkey()

    def _start_hotkey(self):
        self._bridge = HotkeyBridge(
            self._hotkey_str,
            parent=self,
        )
        self._bridge.triggered.connect(self._toggle_recording)
        self._bridge.start()

    def _toggle_recording(self):
        if self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        play_system_sound("Ping")
        try:
            self._recorder.start()
        except Exception as exc:
            self.recording_failed.emit(str(exc))
            return

        self._overlay.show_recording()
        self._tray.setIcon(make_tray_icon(True))
        self._tray.setToolTip("WhisperStack — recording")
        self._status_action.setText("🔴 Recording")

    def _stop_recording(self):
        self._overlay.show_processing()
        play_system_sound("Glass")

        self._tray.setIcon(make_tray_icon(False))
        self._tray.setToolTip("WhisperStack — processing")
        self._status_action.setText("⚙️ Processing")

        self._recorder.stop(on_done=lambda text: self.transcription_done.emit(text))

    def _on_transcription_done(self, text: str):
        self._overlay.hide_recording()
        self._status_action.setText("⚪ Idle")
        self._tray.setToolTip("WhisperStack — idle")
        self._tray.setIcon(make_tray_icon(False))

        if text:
            message = "✅ Transcribed and pasted!"
            icon = QSystemTrayIcon.MessageIcon.Information
        else:
            message = "No speech or copied items detected"
            icon = QSystemTrayIcon.MessageIcon.Warning

        self._tray.showMessage(
            "WhisperStack",
            message,
            icon,
            2500,
        )

    def _on_recording_failed(self, error: str):
        self._overlay.hide_recording()
        self._status_action.setText("⚪ Idle")
        self._tray.setIcon(make_tray_icon(False))
        self._tray.setToolTip("WhisperStack — idle")

        print(f"⚠️ Recording failed: {error}")

        self._tray.showMessage(
            "WhisperStack",
            f"Could not open microphone: {error}",
            QSystemTrayIcon.MessageIcon.Critical,
            3500,
        )

    def _quit_application(self):
        if self._bridge is not None:
            self._bridge.stop()

        if self._recorder.is_recording:
            self._recorder.stop()

        QApplication.quit()


def main():
    setup_logging()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("WhisperStack")
    app.setApplicationVersion("1.1.0")

    from PyQt6.QtCore import QLockFile, QDir

    lock_file = QLockFile(QDir.tempPath() + "/whisperstack.lock")
    if not lock_file.tryLock(100):
        print("⚠️ WhisperStack is already running! Showing notification and exiting.")
        from PyQt6.QtCore import QTimer, Qt
        from PyQt6.QtWidgets import (
            QWidget,
            QLabel,
            QVBoxLayout,
            QGraphicsDropShadowEffect,
        )
        from PyQt6.QtGui import QColor

        window = QWidget()
        window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        window.resize(320, 100)

        screen = app.primaryScreen().geometry()
        window.move(
            int((screen.width() - window.width()) / 2),
            int((screen.height() - window.height()) / 2),
        )

        layout = QVBoxLayout(window)
        layout.setContentsMargins(15, 15, 15, 15)

        container = QWidget()
        container.setObjectName("Container")
        container.setStyleSheet("""
            QWidget#Container {
                background-color: rgba(30, 30, 35, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 12px;
            }
        """)

        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(4)
        container_layout.setContentsMargins(10, 12, 10, 12)

        label = QLabel("ℹ️  WhisperStack is already running")
        label.setStyleSheet(
            "color: #ffffff; font-size: 14px; font-weight: bold; background: transparent;"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sublabel = QLabel("Check your top menu bar for the mic icon")
        sublabel.setStyleSheet(
            "color: rgba(255, 255, 255, 0.60); font-size: 11px; background: transparent;"
        )
        sublabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container_layout.addWidget(label)
        container_layout.addWidget(sublabel)
        layout.addWidget(container)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 5)
        container.setGraphicsEffect(shadow)

        # Keep global reference so garbage collection doesn't delete it
        app.setProperty("notification_window", window)

        window.show()
        QTimer.singleShot(3000, app.quit)
        sys.exit(app.exec())

    app.setProperty("lock_file", lock_file)

    try:
        from AppKit import (
            NSApp,
            NSApplicationActivationPolicyAccessory,
        )

        NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except ImportError:
        pass

    tray = WhisperStackTray()
    app.setProperty("whisperstack_tray", tray)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
