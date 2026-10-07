"""
make_icon.py — Run this ONCE before building with PyInstaller.
Generates icon.iconset/ and icon.icns using PyQt6 + macOS iconutil.

Usage:
    python make_icon.py
"""

import os
import subprocess
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication


def draw_icon(size: int) -> QPixmap:
    """Draw a microphone icon at the given pixel size."""
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    s = size / 22.0   # scale factor relative to design at 22px

    white = QColor(30, 30, 30, 240)   # dark for light backgrounds (icns standard)
    pen = QPen(white, max(1.0, 1.6 * s))
    p.setPen(pen)
    p.setBrush(QBrush(white))

    # Mic body
    bx = int(8 * s);  by = int(2 * s)
    bw = int(6 * s);  bh = int(11 * s)
    br = int(3 * s)
    p.drawRoundedRect(bx, by, bw, bh, br, br)

    # Stand arc + stem
    p.setBrush(Qt.BrushStyle.NoBrush)
    from PyQt6.QtCore import QRectF
    ax = 5 * s;  ay = 8 * s;  aw = 12 * s;  ah = 8 * s
    p.drawArc(QRectF(ax, ay, aw, ah), 0, -180 * 16)

    lw = int(11 * s)
    p.drawLine(lw, int(16 * s), lw, int(20 * s))
    p.drawLine(int(8 * s), int(20 * s), int(14 * s), int(20 * s))

    p.end()
    return px


def main():
    app = QApplication(sys.argv)

    iconset_dir = "icon.iconset"
    os.makedirs(iconset_dir, exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512, 1024]
    retina_map = {16: "16", 32: "16@2x", 64: "32@2x", 128: "128",
                  256: "128@2x", 512: "512", 1024: "512@2x"}

    for size in sizes:
        px = draw_icon(size)
        filename = f"icon_{retina_map[size]}.png"
        path = os.path.join(iconset_dir, filename)
        px.save(path, "PNG")
        print(f"  wrote {path}")

    print("\nRunning iconutil…")
    result = subprocess.run(
        ["iconutil", "-c", "icns", iconset_dir, "-o", "icon.icns"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("iconutil failed:", result.stderr)
        sys.exit(1)

    print("✅ icon.icns created successfully.")


if __name__ == "__main__":
    main()