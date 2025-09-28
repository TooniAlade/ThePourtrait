import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import serial
from PIL import Image

# This script listens to an Arduino over serial for six color selections,
# writes colors.json, and optionally invokes marble_render.py
# Expected Arduino output: six lines, each a decimal integer like 16711680 (0xFF0000)
# in the order selected. End when six values are printed.

DEFAULT_COLORS_JSON = os.path.join(os.path.dirname(__file__), "colors.json")
MARBLER = os.path.join(os.path.dirname(__file__), "marble_render.py")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "marble_output.png")


def int_to_rgb_tuple(n: int) -> Tuple[int, int, int]:
    r = (n >> 16) & 0xFF
    g = (n >> 8) & 0xFF
    b = n & 0xFF
    return (r, g, b)


def read_six_colors(port: str, baud: int = 115200, timeout: float = 10.0) -> List[Tuple[int, int, int]]:
    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    try:
        colors: List[Tuple[int, int, int]] = []
        start = time.time()
        while len(colors) < 6:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for color lines from Arduino")
                continue
            # Accept either hex (0xRRGGBB) or decimal
            try:
                if line.lower().startswith("0x"):
                    val = int(line, 16)
                else:
                    val = int(line)
            except ValueError:
                # ignore non-number lines
                continue
            colors.append(int_to_rgb_tuple(val))
            print(f"Got color {len(colors)}: #{val:06X}")
        return colors
    finally:
        ser.close()


def write_colors_json(colors: List[Tuple[int, int, int]], path: str = DEFAULT_COLORS_JSON) -> None:
    # Match marble_render.py expectations: list of items each with "color"
    payload = [{"color": "#{:02x}{:02x}{:02x}".format(r, g, b)} for (r, g, b) in colors]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(colors)} colors to {path}")


def main():
    p = argparse.ArgumentParser(description="Read 6 colors from Arduino and run marbler")
    p.add_argument("--port", required=True, help="Serial port (e.g., COM3 on Windows)")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p.add_argument("--colors-json", default=DEFAULT_COLORS_JSON, help="Path to write colors.json")
    p.add_argument("--run", action="store_true", help="Run marble_render.py after writing colors.json")
    p.add_argument("--input", default=None, help="Silhouette image path (if running marbler)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path (if running marbler)")
    args = p.parse_args()

    colors = read_six_colors(args.port, baud=args.baud)
    write_colors_json(colors, args.colors_json)

    if args.run:
        # Build command to run marbler with preview (default settings)
        import subprocess
        cmd = [sys.executable, MARBLER]
        if args.input:
            cmd += ["--input", args.input]
        cmd += ["--colors", args.colors_json, "--output", args.output]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
