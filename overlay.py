"""
overlay.py — Minimalist, floating circular recording HUD for WhisperStack.

Key Architectural Components:
1. Floating Circular Puck with Per-Monitor Profile Memory:
   - Ultra-compact, matte-black circular HUD. Default diameter 38px (configurable per monitor).
   - Tracks circle center (cx, cy) and circle size independently per connected monitor.
   - Real-time monitor tracking: seamlessly jumps across displays to follow the mouse cursor,
     instantly recalling the specific position AND custom circle size saved for that display.
   - Draggable & Sticky: can be moved anywhere on screen; immediately persists exact circle center
     coordinates and monitor margins to settings.json so it sticks in place permanently.

2. Interactive Edge Resizing (Mouse Drag & Scroll Wheel):
   - Hovering over the circular edge shows a diagonal resize cursor (SizeFDiagCursor).
   - Dragging the edge inward shrinks the circle down to 26px; dragging outward expands up to 76px.
   - Mouse wheel or trackpad scroll over the circle also dynamically adjusts size.
   - The word popup badge and typography remain fixed at a comfortable, readable size (12pt),
     always floating directly above the top rim of the circle regardless of circle size.

3. Reactive Audio Equalizer Waves:
   - Centred within the circle: 5 sleek, crimson equalizer bars with smooth round caps.
   - Dimensions scale proportionally with circle diameter.
   - Dynamic real-time response to microphone volume: dancing with organic sinusoidal harmonics.
   - When silent: settles into minimal, dim resting dots.

4. Interactive Word-by-Word Floating Transcription:
   - Real-time text streams from the live transcription engine.
   - Each word pops up gracefully from the top of the circle, enclosed in a frosted translucent pill.
   - Rises smoothly with cubic easing, rests clearly for easy reading, and then fades/disappears
     upward as the next word emerges.
   - Adaptive queue timing ensures live pacing never falls behind fast speech.

5. Spinner State (Processing):
   - When speech stops and transcription is active, the audio bars smoothly transition to a
     spinning crimson loading arc, giving clear visual feedback that transcription is underway.
"""

from __future__ import annotations

import collections
import json
import math
import os

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import QApplication, QWidget

# Path to the persistent settings JSON file.
SETTINGS_FILE = "/Users/Work/Projects/Whisper-local/settings.json"


def load_settings() -> dict:
    """Safely load settings dictionary from disk, returning an empty dict on error."""
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(settings: dict):
    """Atomically persist settings dictionary to disk using a temporary swap file."""
    try:
        settings_directory = os.path.dirname(SETTINGS_FILE)
        os.makedirs(settings_directory, exist_ok=True)
        temporary_file = f"{SETTINGS_FILE}.tmp"
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(settings, file, indent=4)
        os.replace(temporary_file, SETTINGS_FILE)
    except OSError:
        pass


class RecordingOverlay(QWidget):
    """
    Floating, non-focusable circular overlay widget for WhisperStack.
    Displays dynamic audio waves when recording, a spinner when processing,
    and streams transcription words popping up one-by-one from the top of the circle.
    Supports per-monitor custom sizing and positioning with sticky persistence.
    """

    # ── Geometry & Layout Constants ──────────────────────────────────────────
    # Transparent container canvas dimensions.
    # Canvas provides ample room for word pills to float above the circle.
    WINDOW_WIDTH = 220
    WINDOW_HEIGHT = 160

    # Circle sizing bounds (in pixels).
    DEFAULT_CIRCLE_DIAMETER = 38.0
    MIN_CIRCLE_DIAMETER = 26.0
    MAX_CIRCLE_DIAMETER = 76.0

    # Edge resize detection threshold in pixels.
    EDGE_RESIZE_MARGIN = 6.0

    # 5-bar equalizer parameters inside the circle.
    BAR_COUNT = 5

    # Word popup geometry parameters.
    # Text and badge size stay constant and legible regardless of circle size.
    WORD_PILL_HEIGHT = 24.0

    # Default margins from the bottom-right corner of a monitor when no profile exists.
    DEFAULT_MARGIN_RIGHT = 24.0
    DEFAULT_MARGIN_BOTTOM = 28.0

    def __init__(self, recorder=None):
        super().__init__(None)

        self._recorder = recorder

        # Volume state and visual wave animation variables.
        self._current_volume: float = 0.0
        self._display_volume: float = 0.0
        self._phase: float = 0.0
        self._state: str = "idle"  # "idle", "recording", "processing"

        # Circle diameter for the active monitor (initialized to default).
        self._circle_diameter: float = self.DEFAULT_CIRCLE_DIAMETER

        # Active monitor tracking to detect when mouse crosses displays.
        self._current_screen_name: str | None = None

        # Dragging and resizing interaction state.
        self._is_dragging: bool = False
        self._is_resizing: bool = False
        self._drag_start_pos: QPointF | None = None
        self._drag_start_center: QPointF | None = None
        self._resize_start_dist: float = 0.0
        self._resize_start_diameter: float = self.DEFAULT_CIRCLE_DIAMETER

        # Global cursor override tracking (needed for non-activating panels on macOS).
        self._cursor_override_active: bool = False
        self._active_cursor_shape: Qt.CursorShape = Qt.CursorShape.ArrowCursor

        # Word-by-word streaming animation state machine.
        self._word_queue: collections.deque[str] = collections.deque()
        self._current_word: str | None = None
        self._word_elapsed: float = 0.0
        self._word_duration: float = 0.40
        self._seen_word_count: int = 0
        self._last_transcript: str = ""

        # Configure macOS window flags for a non-activating, always-on-top, frameless tool window.
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )

        # Enable translucent background rendering so only the circle and pill are visible.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)

        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        # Precise 33ms animation timer (~30 FPS) driving audio smoothing, wave phase, and words.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

        # Initial layout on the active screen containing the cursor.
        self._apply_initial_screen_profile()

    # ── Mathematical Coordinate Helpers ──────────────────────────────────────
    @property
    def circle_radius(self) -> float:
        """Current radius of the circular HUD puck."""
        return self._circle_diameter / 2.0

    @property
    def local_circle_center_x(self) -> float:
        """X-coordinate of the circle center relative to this widget canvas."""
        return self.WINDOW_WIDTH / 2.0

    @property
    def local_circle_center_y(self) -> float:
        """Y-coordinate of the circle center relative to this widget canvas (constant center)."""
        return 115.0

    def _get_circle_center_global(self) -> QPointF:
        """Calculate the exact absolute global coordinates of the circle center on screen."""
        return QPointF(
            self.x() + self.local_circle_center_x,
            self.y() + self.local_circle_center_y,
        )

    def _move_circle_center_to(self, global_cx: float, global_cy: float):
        """Move the transparent widget window so that the circle center lands exactly at (global_cx, global_cy)."""
        target_win_x = round(global_cx - self.local_circle_center_x)
        target_win_y = round(global_cy - self.local_circle_center_y)
        self.move(target_win_x, target_win_y)

    # ── Screen Profiles & Multi-Monitor Tracking ─────────────────────────────
    def _apply_initial_screen_profile(self):
        """Position and size the overlay on the screen currently containing the mouse cursor."""
        cursor_pos = QCursor.pos()
        screen = QApplication.screenAt(cursor_pos) or QApplication.primaryScreen()
        if screen:
            self._apply_screen_profile(screen)

    def _apply_screen_profile(self, screen):
        """
        Load and apply the custom position and size profile for the given screen.
        If a saved profile exists for this display, restores exact circle center and size.
        If no profile exists, defaults to bottom-right placement and default size.
        """
        if screen is None:
            return

        self._current_screen_name = screen.name()
        # Use full physical screen geometry so the macOS Dock never restricts placement
        geometry = screen.geometry()
        settings = load_settings()
        monitors = settings.get("monitors", {})

        # Check for modern monitor profile or legacy screen_positions entry.
        profile = monitors.get(screen.name())
        if not profile:
            # Check fallback legacy dict from previous versions.
            legacy_pos = settings.get("screen_positions", {}).get(screen.name())
            if legacy_pos:
                profile = {
                    "circle_cx": legacy_pos.get("x", 0) + self.WINDOW_WIDTH / 2.0,
                    "circle_cy": legacy_pos.get("y", 0)
                    + self.WINDOW_HEIGHT
                    - self.DEFAULT_CIRCLE_DIAMETER / 2.0
                    - 6.0,
                    "circle_size": self.DEFAULT_CIRCLE_DIAMETER,
                }

        if profile:
            # 1. Restore saved size for this monitor.
            saved_size = profile.get("circle_size", self.DEFAULT_CIRCLE_DIAMETER)
            self._circle_diameter = max(
                self.MIN_CIRCLE_DIAMETER,
                min(self.MAX_CIRCLE_DIAMETER, float(saved_size)),
            )

            # 2. Restore saved circle center coordinates.
            saved_cx = profile.get("circle_cx")
            saved_cy = profile.get("circle_cy")

            if saved_cx is not None and saved_cy is not None:
                r = self.circle_radius
                # Clamp center so the visible circle remains 100% on this display.
                cx = max(
                    geometry.left() + r, min(geometry.right() - r, float(saved_cx))
                )
                cy = max(
                    geometry.top() + r, min(geometry.bottom() - r, float(saved_cy))
                )
                self._move_circle_center_to(cx, cy)
                self.update()
                return

        # Fallback for unconfigured screens: default size and bottom-right corner alignment.
        self._circle_diameter = self.DEFAULT_CIRCLE_DIAMETER
        r = self.circle_radius
        cx = geometry.right() - self.DEFAULT_MARGIN_RIGHT - r
        cy = geometry.bottom() - self.DEFAULT_MARGIN_BOTTOM - r
        self._move_circle_center_to(cx, cy)
        self.update()

    def _save_current_monitor_profile(self):
        """
        Persist the active monitor's circle center (cx, cy) and circle size to settings.json.
        Ensures both custom position and custom size stick permanently for this monitor.
        """
        center_global = self._get_circle_center_global()
        screen = (
            QApplication.screenAt(center_global.toPoint())
            or QApplication.primaryScreen()
        )
        if screen is None:
            return

        self._current_screen_name = screen.name()
        # Use full physical screen geometry to allow placement right down to the screen edge
        geom = screen.geometry()
        settings = load_settings()
        monitors = settings.get("monitors", {})

        # Calculate exact center coordinates and right/bottom offsets.
        cx = round(center_global.x())
        cy = round(center_global.y())
        margin_right = max(0, round(geom.right() - cx))
        margin_bottom = max(0, round(geom.bottom() - cy))

        monitors[screen.name()] = {
            "circle_cx": cx,
            "circle_cy": cy,
            "circle_size": round(self._circle_diameter, 1),
            "margin_right": margin_right,
            "margin_bottom": margin_bottom,
        }

        settings["monitors"] = monitors
        # Legacy compatibility updates.
        settings["pos_x"] = self.x()
        settings["pos_y"] = self.y()
        save_settings(settings)

    # ── Global Cursor Management ─────────────────────────────────────────────
    def _set_global_cursor(self, shape: Qt.CursorShape):
        """Set global macOS cursor override for smooth drag and hover feedback."""
        if self._cursor_override_active and shape == self._active_cursor_shape:
            return

        if self._cursor_override_active:
            QApplication.restoreOverrideCursor()
            self._cursor_override_active = False

        QApplication.setOverrideCursor(QCursor(shape))
        self._cursor_override_active = True
        self._active_cursor_shape = shape

    def _restore_global_cursor(self):
        """Restore default system cursor."""
        if not self._cursor_override_active:
            return

        QApplication.restoreOverrideCursor()
        self._cursor_override_active = False
        self._active_cursor_shape = Qt.CursorShape.ArrowCursor

    def _refresh_hover_cursor(self):
        """
        Poll global cursor position to show:
        - SizeFDiagCursor when hovering near the circular border (edge resize mode).
        - OpenHandCursor when hovering inside the circle (drag/move mode).
        - Standard arrow cursor when outside.
        """
        if not self.isVisible():
            self._restore_global_cursor()
            return

        if self._is_resizing:
            self._set_global_cursor(Qt.CursorShape.SizeFDiagCursor)
            return

        if self._is_dragging:
            self._set_global_cursor(Qt.CursorShape.ClosedHandCursor)
            return

        global_pos = QCursor.pos()
        center_global = self._get_circle_center_global()
        dist = math.hypot(
            global_pos.x() - center_global.x(),
            global_pos.y() - center_global.y(),
        )

        r = self.circle_radius
        # Edge hit test: within EDGE_RESIZE_MARGIN pixels of the circumference.
        if abs(dist - r) <= self.EDGE_RESIZE_MARGIN:
            self._set_global_cursor(Qt.CursorShape.SizeFDiagCursor)
        elif dist < r - self.EDGE_RESIZE_MARGIN:
            self._set_global_cursor(Qt.CursorShape.OpenHandCursor)
        else:
            self._restore_global_cursor()

    # ── Main Animation Tick ──────────────────────────────────────────────────
    def _tick(self):
        """
        Called every 33ms (~30 FPS) to update:
        1. Processing spinner rotation angle.
        2. Real-time monitor jumping when the cursor crosses displays.
        3. Microphone audio volume smoothing and wave phase.
        4. Ingestion of live transcript words from the recorder.
        5. Word-by-word popup animation progress and queue drainage.
        """
        self._refresh_hover_cursor()

        # State: Processing — advance spinner rotation angle.
        if self._state == "processing":
            self._phase += 0.18
            self.update()
            return

        # Real-time Multi-Monitor Tracking:
        # If the user moves their mouse to a different display while speaking, jump the circle!
        if (
            self._state == "recording"
            and not self._is_dragging
            and not self._is_resizing
        ):
            cursor_pos = QCursor.pos()
            target_screen = QApplication.screenAt(cursor_pos)
            if (
                target_screen is not None
                and self._current_screen_name != target_screen.name()
            ):
                self._apply_screen_profile(target_screen)

        # Microphone volume sampling & visual smoothing.
        if self._recorder and self._recorder.is_recording:
            volume = self._recorder.get_volume()
        else:
            volume = 0.0

        self._current_volume = volume

        if volume <= 0.0:
            self._display_volume = 0.0
        else:
            # Fast attack (0.72) for instant responsiveness, smooth decay (0.28) for fluidity.
            smoothing = 0.72 if volume >= self._display_volume else 0.28
            self._display_volume += (volume - self._display_volume) * smoothing
            self._phase += 0.32 + self._display_volume * 0.30

        # Ingest live transcript stream and enqueue new words.
        if self._recorder and self._state == "recording":
            live_transcript = self._recorder.get_live_transcript()
            if live_transcript and live_transcript != self._last_transcript:
                self._last_transcript = live_transcript
                words = live_transcript.strip().split()
                if len(words) > self._seen_word_count:
                    # New words discovered in live stream — push onto queue.
                    new_words = words[self._seen_word_count :]
                    for w in new_words:
                        cleaned = w.strip()
                        if cleaned:
                            self._word_queue.append(cleaned)
                    self._seen_word_count = len(words)
                elif len(words) < self._seen_word_count:
                    # Reset if engine re-anchored text.
                    self._seen_word_count = len(words)

        # Advance word-by-word popup animation.
        dt = 0.033
        if self._current_word is None:
            if self._word_queue:
                self._current_word = self._word_queue.popleft()
                self._word_elapsed = 0.0
                # Adaptive pacing: speed up display duration if queue accumulates.
                q_len = len(self._word_queue)
                if q_len >= 5:
                    self._word_duration = 0.20
                elif q_len >= 3:
                    self._word_duration = 0.28
                elif q_len >= 1:
                    self._word_duration = 0.36
                else:
                    self._word_duration = 0.44
        else:
            self._word_elapsed += dt
            if self._word_elapsed >= self._word_duration:
                # Word display complete: clear so next word pops up immediately.
                self._current_word = None
                self._word_elapsed = 0.0

        self.update()

    # ── Mouse & Wheel Interaction (Move + Resize) ────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        """
        Handles mouse press:
        - If clicking on the circular edge: begins interactive resizing.
        - If clicking inside the circle: begins moving/dragging.
        """
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        pos = event.position()
        center_local = QPointF(self.local_circle_center_x, self.local_circle_center_y)
        dist = math.hypot(pos.x() - center_local.x(), pos.y() - center_local.y())
        r = self.circle_radius

        # Check edge resize hit first.
        if abs(dist - r) <= self.EDGE_RESIZE_MARGIN:
            self._is_resizing = True
            self._is_dragging = False
            self._resize_start_dist = dist
            self._resize_start_diameter = self._circle_diameter
            self.grabMouse()
            self._set_global_cursor(Qt.CursorShape.SizeFDiagCursor)
            event.accept()
            return

        # Check inside circle drag hit or active word pill hit.
        is_inside_circle = dist < r - self.EDGE_RESIZE_MARGIN
        is_on_word = (
            self._current_word is not None
            and 0 <= pos.y() <= (self.local_circle_center_y - r)
            and abs(pos.x() - self.local_circle_center_x) <= 80
        )

        if is_inside_circle or is_on_word:
            self._is_dragging = True
            self._is_resizing = False
            self._drag_start_pos = event.globalPosition()
            self._drag_start_center = self._get_circle_center_global()
            self.grabMouse()
            self._set_global_cursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handles mouse movement:
        - If resizing: scales the circle diameter based on radial mouse displacement.
        - If dragging: moves the circle center following the mouse deltas.
        """
        if self._is_resizing:
            pos = event.position()
            center_local = QPointF(
                self.local_circle_center_x, self.local_circle_center_y
            )
            curr_dist = math.hypot(
                pos.x() - center_local.x(),
                pos.y() - center_local.y(),
            )
            delta_radius = curr_dist - self._resize_start_dist
            new_diameter = self._resize_start_diameter + 2.0 * delta_radius

            # Clamp between minimum (26px) and maximum (76px) bounds.
            self._circle_diameter = max(
                self.MIN_CIRCLE_DIAMETER,
                min(self.MAX_CIRCLE_DIAMETER, round(new_diameter, 1)),
            )

            # Center is fixed within the canvas — window remains stationary during resize!
            self.update()
            event.accept()
            return

        if self._is_dragging and self._drag_start_pos and self._drag_start_center:
            delta = event.globalPosition() - self._drag_start_pos
            target_cx = self._drag_start_center.x() + delta.x()
            target_cy = self._drag_start_center.y() + delta.y()

            # Clamp circle center within the full physical screen bounds (Dock never restricts).
            screen = (
                QApplication.screenAt(QPoint(round(target_cx), round(target_cy)))
                or QApplication.primaryScreen()
            )
            if screen:
                geom = screen.geometry()
                r = self.circle_radius
                clamped_cx = max(geom.left() + r, min(geom.right() - r, target_cx))
                clamped_cy = max(geom.top() + r, min(geom.bottom() - r, target_cy))
                self._move_circle_center_to(clamped_cx, clamped_cy)

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """End dragging or resizing and persist updated monitor profile to settings.json."""
        if event.button() == Qt.MouseButton.LeftButton and (
            self._is_dragging or self._is_resizing
        ):
            self._is_dragging = False
            self._is_resizing = False
            self._drag_start_pos = None
            self._drag_start_center = None

            if QWidget.mouseGrabber() is self:
                self.releaseMouse()

            self._restore_global_cursor()

            # Save the new coordinates and custom size for this monitor.
            self._save_current_monitor_profile()
            self._refresh_hover_cursor()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """
        Convenience feature: scrolling while hovering over the circle smoothly
        resizes the circle puck and immediately saves the size for this monitor.
        """
        center_local = QPointF(self.local_circle_center_x, self.local_circle_center_y)
        pos = event.position()
        dist = math.hypot(pos.x() - center_local.x(), pos.y() - center_local.y())

        # Only resize if hovering within or directly on the circle.
        if dist <= self.circle_radius + self.EDGE_RESIZE_MARGIN:
            delta_y = event.angleDelta().y()
            step = 2.0 if delta_y > 0 else -2.0
            new_diameter = self._circle_diameter + step

            self._circle_diameter = max(
                self.MIN_CIRCLE_DIAMETER,
                min(self.MAX_CIRCLE_DIAMETER, new_diameter),
            )

            self._save_current_monitor_profile()
            self.update()
            event.accept()
            return

        super().wheelEvent(event)

    # ── Paint Event & Visual Rendering ───────────────────────────────────────
    def _bar_height(self, index: int) -> float:
        """Calculate dynamic bar height proportional to current circle diameter."""
        volume = self._display_volume
        max_h = max(10.0, self._circle_diameter * 0.44)
        silent_h = max(2.0, self._circle_diameter * 0.08)

        if volume <= 0.0:
            return silent_h

        centre = (self.BAR_COUNT - 1) / 2.0
        distance = abs(index - centre) / centre
        centre_weight = 1.0 - 0.25 * distance

        # Dual wave frequencies for organic, natural fluid motion.
        primary = 0.5 + 0.5 * math.sin(self._phase + index * 0.95)
        secondary = 0.5 + 0.5 * math.sin(self._phase * 1.52 - index * 1.30)
        movement = 0.30 + 0.45 * primary + 0.25 * secondary

        visual_volume = min(1.0, 0.25 + volume * 1.15)
        height = (
            silent_h + (max_h - silent_h) * visual_volume * movement * centre_weight
        )
        return max(silent_h + 1.0, min(max_h, height))

    def paintEvent(self, event):
        """Render the circle, audio waves, processing spinner, and popping word pills."""
        del event

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Clear entire transparent backing canvas buffer every single frame.
        # This completely eliminates dirty buffer remnants and prevents ghost words/badges.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        cx = self.local_circle_center_x
        cy = self.local_circle_center_y
        radius = self.circle_radius
        circle_rect = QRectF(
            cx - radius, cy - radius, self._circle_diameter, self._circle_diameter
        )

        # ── 1. RENDER FLOATING WORD-BY-WORD POPUP (Active Recording) ─────────
        # Words always float cleanly directly above the circle's top rim.
        if self._state == "recording" and self._current_word:
            progress = min(1.0, self._word_elapsed / max(0.01, self._word_duration))

            # 3-Phase Animation Lifecycle:
            # Phase A (0.0 to 0.22): Rise up from circle + fade in (cubic ease-out).
            # Phase B (0.22 to 0.72): Hold stable at full opacity for reading.
            # Phase C (0.72 to 1.00): Float gently upward + fade out.
            if progress < 0.22:
                sub = progress / 0.22
                ease = 1.0 - math.pow(1.0 - sub, 3)
                alpha = ease
                y_offset = (1.0 - ease) * 12.0
            elif progress < 0.72:
                alpha = 1.0
                y_offset = 0.0
            else:
                sub = (progress - 0.72) / 0.28
                ease = sub * sub
                alpha = max(0.0, 1.0 - ease)
                y_offset = -ease * 8.0

            if alpha > 0.05:
                font = QFont("Helvetica Neue", 12)
                font.setWeight(QFont.Weight.DemiBold)
                painter.setFont(font)

                metrics = QFontMetrics(font)
                text_width = metrics.horizontalAdvance(self._current_word)
                pill_width = min(self.WINDOW_WIDTH - 20.0, max(36.0, text_width + 22.0))

                # Position pill directly 6px above the circle's top rim, dynamically tracking circle size
                pill_resting_y = (cy - radius) - self.WORD_PILL_HEIGHT - 6.0
                pill_rect = QRectF(
                    cx - pill_width / 2.0,
                    pill_resting_y + y_offset,
                    pill_width,
                    self.WORD_PILL_HEIGHT,
                )

                # Solid matte-black capsule badge.
                painter.setPen(QPen(QColor(255, 255, 255, int(45 * alpha)), 1.0))
                painter.setBrush(QColor(12, 12, 16, int(245 * alpha)))
                painter.drawRoundedRect(
                    pill_rect, self.WORD_PILL_HEIGHT / 2.0, self.WORD_PILL_HEIGHT / 2.0
                )

                # Crisp bright white text.
                painter.setPen(QPen(QColor(255, 255, 255, int(255 * alpha))))
                painter.drawText(
                    pill_rect,
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter,
                    self._current_word,
                )

        # ── 2. RENDER MATTE-BLACK CIRCLE PUCK ────────────────────────────────
        # Deep matte black fill with reactive border glow.
        painter.setBrush(QColor(14, 14, 18, 246))

        if self._state == "processing":
            painter.setPen(QPen(QColor(255, 80, 80, 150), 1.4))
        elif self._display_volume > 0.01:
            border_alpha = int(70 + 175 * min(1.0, self._display_volume))
            painter.setPen(QPen(QColor(255, 65, 65, border_alpha), 1.5))
        else:
            painter.setPen(QPen(QColor(255, 255, 255, 34), 1.1))

        painter.drawEllipse(circle_rect)

        # ── 3. STATE: PROCESSING (Loading Spinner) ───────────────────────────
        if self._state == "processing":
            spinner_size = max(14.0, self._circle_diameter * 0.50)
            spinner_rect = QRectF(
                cx - spinner_size / 2.0,
                cy - spinner_size / 2.0,
                spinner_size,
                spinner_size,
            )

            pen_width = max(1.6, min(2.8, 2.2 * (self._circle_diameter / 40.0)))
            painter.setPen(
                QPen(
                    QColor(255, 80, 80, 230),
                    pen_width,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)

            start_angle = int((self._phase * 55) % 360) * -16
            span_angle = 270 * 16
            painter.drawArc(spinner_rect, start_angle, span_angle)
            painter.end()
            return

        # ── 4. STATE: RECORDING (5-Bar Audio Equalizer) ───────────────────────
        scale = self._circle_diameter / 40.0
        bar_w = max(1.8, min(3.4, 2.4 * scale))
        bar_gap = max(1.5, min(3.0, 2.0 * scale))
        total_bar_width = self.BAR_COUNT * bar_w + (self.BAR_COUNT - 1) * bar_gap
        start_x = cx - total_bar_width / 2.0
        is_silent = self._display_volume <= 0.0

        painter.setPen(Qt.PenStyle.NoPen)

        for index in range(self.BAR_COUNT):
            bar_height = self._bar_height(index)
            bar_x = start_x + index * (bar_w + bar_gap)
            bar_y = cy - bar_height / 2.0

            if is_silent:
                colour = QColor(255, 255, 255, 45)
            else:
                centre = (self.BAR_COUNT - 1) / 2.0
                distance = abs(index - centre) / centre

                red = 255
                green = int(45 + 30 * (1.0 - distance))
                blue = int(50 + 20 * (1.0 - distance))
                alpha = int(180 + 75 * min(1.0, self._display_volume))
                colour = QColor(red, green, blue, alpha)

            painter.setBrush(colour)
            painter.drawRoundedRect(
                QRectF(bar_x, bar_y, bar_w, bar_height),
                bar_w / 2.0,
                bar_w / 2.0,
            )

        painter.end()

    # ── Public Lifecycle API ─────────────────────────────────────────────────
    def show_recording(self):
        """Prepare visual state, position HUD at screen bottom-right, and start animation."""
        self._state = "recording"
        self._current_volume = 0.0
        self._display_volume = 0.0
        self._phase = 0.0

        # Reset streaming transcription queues.
        self._word_queue.clear()
        self._current_word = None
        self._word_elapsed = 0.0
        self._seen_word_count = 0
        self._last_transcript = ""

        self._apply_initial_screen_profile()
        self._timer.start()
        self.show()
        self.raise_()

    def show_processing(self):
        """Transition into processing mode to show the spinning loader arc."""
        self._state = "processing"
        self._phase = 0.0
        # Clear word queues so the spinner is unobstructed.
        self._word_queue.clear()
        self._current_word = None
        self.update()

    def hide_recording(self):
        """Stop animation timers, reset dragging/cursor state, and hide the HUD."""
        self._timer.stop()
        self._state = "idle"
        self._current_volume = 0.0
        self._display_volume = 0.0
        self._is_dragging = False
        self._is_resizing = False

        self._word_queue.clear()
        self._current_word = None

        if QWidget.mouseGrabber() is self:
            self.releaseMouse()

        self._restore_global_cursor()
        self.hide()
