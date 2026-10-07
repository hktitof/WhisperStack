"""
app_logger.py — Production-grade logging and structured transcript session archiving for WhisperStack.

Features:
1. Rotating file handler for application diagnostics (logs/whisperstack.log, 10MB x 5 backups).
2. Millisecond ISO timestamps on all log entries.
3. Daily partitioned structured transcript history (logs/transcripts/YYYY-MM-DD.jsonl).
4. Human-readable daily markdown transcript log (logs/transcripts/YYYY-MM-DD.md).
5. Comprehensive session telemetry (audio metrics, mic used/fallback, duration, copied items, status).
6. Uncaught exception hook to capture fatal crashes.
7. CLI inspection utilities (--recent, --errors, --tail).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path("/Users/Work/Projects/Whisper-local")
LOGS_DIR = BASE_DIR / "logs"
TRANSCRIPTS_DIR = LOGS_DIR / "transcripts"
APP_LOG_FILE = LOGS_DIR / "whisperstack.log"

_session_lock = threading.Lock()
_logger_initialized = False

logger = logging.getLogger("WhisperStack")


class MillisecondFormatter(logging.Formatter):
    """Formats timestamps with exact milliseconds: 2026-09-30 09:35:47.216"""

    def formatTime(self, record, datefmt=None):
        created_dt = datetime.fromtimestamp(record.created)
        s = created_dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")
        return f"{s}.{int(record.msecs):03d}"


class StreamToLogger:
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self._in_write = False

    def write(self, buf):
        if self._in_write:
            return
        self._in_write = True
        try:
            for line in buf.rstrip().splitlines():
                s = line.strip()
                if s:
                    self.logger_func(s)
        finally:
            self._in_write = False

    def flush(self):
        pass


def setup_logging() -> logging.Logger:
    global _logger_initialized
    if _logger_initialized:
        return logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = MillisecondFormatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Rotating file handler (10MB, 5 backups)
    file_handler = RotatingFileHandler(
        APP_LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console stream handler using pristine original stdout
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    sys.stdout = StreamToLogger(logger.info)
    sys.stderr = StreamToLogger(logger.error)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "💥 Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception
    _logger_initialized = True
    return logger


def log_session(session_record: dict):
    """
    Persist structured session telemetry and spoken transcript to daily JSONL and Markdown files.
    """
    try:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

        jsonl_path = TRANSCRIPTS_DIR / f"{date_str}.jsonl"
        md_path = TRANSCRIPTS_DIR / f"{date_str}.md"

        if "timestamp" not in session_record:
            session_record["timestamp"] = now.isoformat(timespec="milliseconds")

        record_json = json.dumps(session_record, ensure_ascii=False)

        with _session_lock:
            # 1. Append to daily JSONL archive
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(record_json + "\n")

            # 2. Append to human-readable Markdown log
            md_entry = _format_session_markdown(session_record)
            with open(md_path, "a", encoding="utf-8") as f:
                f.write(md_entry + "\n\n")

        status = session_record.get("status", "SUCCESS")
        dur = session_record.get("duration_seconds", 0.0)
        logger.info(
            f"💾 Session {session_record.get('session_id')} archived ({status}, {dur:.2f}s) -> transcripts/{date_str}.jsonl"
        )

    except Exception as exc:
        logger.error(f"⚠️ Failed to log session transcript: {exc}", exc_info=True)


def _format_session_markdown(rec: dict) -> str:
    ts = rec.get("timestamp", "")
    dur = rec.get("duration_seconds", 0.0)
    app = rec.get("target_app", "Unknown")
    status = rec.get("status", "SUCCESS")
    mic_actual = rec.get("mic_actual", "Default")
    fallback = " ⚠️ [FALLBACK ACTIVE]" if rec.get("fallback_used") else ""
    spoken = rec.get("spoken_text", "").strip()
    copied = rec.get("copied_items", [])
    error = rec.get("error")

    status_badge = "✅" if status == "SUCCESS" else "⚠️"

    lines = [
        f"### {status_badge} Session #{rec.get('session_id')} — {ts}",
        f"- **Duration**: `{dur:.2f}s` | **App**: `{app}` | **Mic**: `{mic_actual}{fallback}`",
        f"- **Status**: `{status}`",
    ]

    audio_file = rec.get("audio_file")
    if audio_file:
        lines.append(f"- **Audio Recording**: `{audio_file}`")

    audio_m = rec.get("audio_metrics")
    if audio_m:
        lines.append(
            f"- **Audio Metrics**: Peak: `{audio_m.get('peak_amplitude')}` | Gate: `{audio_m.get('noise_gate')}` | Samples: `{audio_m.get('total_samples')}` | Silent: `{audio_m.get('is_silent')}`"
        )

    if error:
        lines.append(f"- **Error**: `{error}`")

    if spoken:
        lines.append(f"\n**Spoken Transcript**:\n> {spoken}")
    else:
        lines.append("\n**Spoken Transcript**: *(None / Empty)*")

    if copied:
        lines.append(f"\n**Copied Items ({len(copied)})**:")
        for idx, item in enumerate(copied, 1):
            t_str = item.get("time_str", "")
            src = item.get("source_meta", "")
            preview = item.get("preview", item.get("content", "")[:200])
            lines.append(f"{idx}. `{t_str}` {src}:\n   ```\n   {preview}\n   ```")

    lines.append("\n---")
    return "\n".join(lines)


def get_recent_sessions(limit: int = 10, errors_only: bool = False) -> list[dict]:
    """Retrieve recent session records across transcripts."""
    records = []
    if not TRANSCRIPTS_DIR.exists():
        return records

    jsonl_files = sorted(TRANSCRIPTS_DIR.glob("*.jsonl"), reverse=True)
    for jf in jsonl_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if errors_only and rec.get("status") == "SUCCESS":
                            continue
                        records.append(rec)
                        if len(records) >= limit:
                            return records
                    except Exception:
                        pass
        except Exception:
            pass
    return records


def main_cli():
    parser = argparse.ArgumentParser(
        description="WhisperStack Log & Transcript Inspector"
    )
    parser.add_argument(
        "--recent", "-r", type=int, default=0, help="Show N recent sessions"
    )
    parser.add_argument(
        "--errors", "-e", action="store_true", help="Show only failed or empty sessions"
    )
    parser.add_argument(
        "--tail", "-t", type=int, default=0, help="Tail N lines of application log"
    )
    args = parser.parse_args()

    if args.tail > 0:
        if not APP_LOG_FILE.exists():
            print("No log file found at", APP_LOG_FILE)
            return
        with open(APP_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-args.tail :]
            print("".join(lines), end="")
        return

    count = args.recent if args.recent > 0 else (10 if args.errors else 5)
    sessions = get_recent_sessions(limit=count, errors_only=args.errors)
    if not sessions:
        print("No sessions found.")
        return

    print(f"=== Last {len(sessions)} Sessions (Errors only: {args.errors}) ===")
    for s in sessions:
        badge = "✅" if s.get("status") == "SUCCESS" else "⚠️"
        print(
            f"\n{badge} Session #{s.get('session_id')} [{s.get('timestamp')}] ({s.get('duration_seconds')}s)"
        )
        print(
            f"   App: {s.get('target_app')} | Mic: {s.get('mic_actual')} (Fallback: {s.get('fallback_used')})"
        )
        audio = s.get("audio_metrics", {})
        print(
            f"   Metrics: Peak={audio.get('peak_amplitude')}, Silent={audio.get('is_silent')}, Status={s.get('status')}"
        )
        if s.get("error"):
            print(f"   Error: {s.get('error')}")
        spoken = s.get("spoken_text", "")
        if spoken:
            preview = (spoken[:90] + "...") if len(spoken) > 90 else spoken
            print(f"   Speech: {preview}")
        copied = s.get("copied_items", [])
        if copied:
            print(f"   Copied: {len(copied)} item(s)")


if __name__ == "__main__":
    main_cli()
