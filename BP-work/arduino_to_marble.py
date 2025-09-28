import argparse
import json
import os
import sys
import time
from typing import List, Tuple, Optional

import serial
import serial.tools.list_ports as list_ports

# Bridge script: read 6 colors from Arduino over serial, write colors.json,
# optionally wait for a tilt trigger, then run marble_render.py.
# Expected Arduino output:
#   - Six lines containing color integers (decimal like 16711680 or hex like 0xFF0000)
#     in the exact order selected
#   - Later, a line containing the tilt token (default: "TILT") to trigger rendering

DEFAULT_COLORS_JSON = os.path.join(os.path.dirname(__file__), "colors.json")
MARBLER = os.path.join(os.path.dirname(__file__), "marble_render.py")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "marble_output.png")


def int_to_rgb_tuple(n: int) -> Tuple[int, int, int]:
    r = (n >> 16) & 0xFF
    g = (n >> 8) & 0xFF
    b = n & 0xFF
    return (r, g, b)


def detect_port(preferred: Optional[str] = None) -> Optional[str]:
    """Return a likely Arduino COM port. If preferred is provided, return it if present."""
    ports = list(list_ports.comports())
    if preferred:
        for p in ports:
            if p.device.lower() == preferred.lower():
                return p.device
    # Heuristics: common USB serial descriptions/vendors
    candidates = []
    for p in ports:
        desc = (p.description or "").lower()
        manu = (p.manufacturer or "").lower()
        if any(k in desc for k in ["arduino", "silicon labs", "usb serial", "ch340", "wch", "cp210x", "adafruit", "seeed"]):
            candidates.append(p.device)
        elif any(k in manu for k in ["arduino", "adafruit", "wch", "silicon labs", "seeed"]):
            candidates.append(p.device)
    if len(candidates) == 1:
        return candidates[0]
    # Fallback: if only one port total, use it
    if len(ports) == 1:
        return ports[0].device
    return None


def list_available_ports() -> List[Tuple[str, str]]:
    return [(p.device, p.description or "") for p in list_ports.comports()]


def read_six_colors(port: str, baud: int = 115200, timeout: Optional[float] = None) -> List[Tuple[int, int, int]]:
    print(f"Opening serial port {port} @ {baud}...")
    ser = serial.Serial(port, baudrate=baud, timeout=1.0)
    try:
        colors: List[Tuple[int, int, int]] = []
        last_activity = time.time()
        infinite = (timeout is None) or (timeout <= 0)
        print("Waiting for six color lines from Arduino (decimal like 16711680 or hex like 0xFF0000)...")
        while len(colors) < 6:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                if (not infinite) and (time.time() - last_activity > float(timeout)):
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
            last_activity = time.time()
        return colors
    finally:
        ser.close()


def wait_for_token(port: str, token: str = "TILT", baud: int = 115200, timeout: Optional[float] = None) -> bool:
    """Block until a line containing the token is received (case-insensitive) or timeout occurs."""
    if timeout is None or timeout <= 0:
        print(f"Waiting for trigger token '{token}' on {port} (no timeout)...")
    else:
        print(f"Waiting for trigger token '{token}' on {port} (timeout {int(timeout)}s)...")
    token_lc = token.lower()
    ser = serial.Serial(port, baudrate=baud, timeout=1.0)
    try:
        start = time.time()
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                # Print minimal feedback, but avoid flooding the console
                print(f"[serial] {line}")
                if token_lc in line.lower():
                    print("Trigger received.")
                    return True
            if timeout is not None and timeout > 0 and (time.time() - start > timeout):
                print("Timed out waiting for trigger.")
                return False
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
    p.add_argument("--port", default=None, help="Serial port (e.g., COM3 on Windows). If omitted, auto-detect.")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    p.add_argument("--colors-json", default=DEFAULT_COLORS_JSON, help="Path to write colors.json")
    p.add_argument("--run", action="store_true", help="Run marble_render.py after writing colors.json")
    p.add_argument("--wait-for-tilt", action="store_true", help="After reading 6 colors, wait for a tilt trigger before running")
    p.add_argument("--tilt-token", default="TILT", help="Serial line token that signals the tilt trigger")
    p.add_argument("--tilt-timeout", type=float, default=0.0, help="Seconds to wait for tilt trigger; 0 means no timeout")
    p.add_argument("--color-timeout", type=float, default=0.0, help="Seconds to wait for color lines; 0 means no timeout")
    p.add_argument("--input", default=None, help="Silhouette image path (if running marbler)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path (if running marbler)")
    args = p.parse_args()

    if args.list_ports:
        ports = list_available_ports()
        if not ports:
            print("No serial ports found.")
        else:
            print("Available serial ports:")
            for dev, desc in ports:
                print(f"  {dev}: {desc}")
        return

    port = args.port or detect_port()
    if not port:
        print("Could not auto-detect a serial port. Use --list-ports to see options and pass --port COMx.")
        return

    colors = read_six_colors(port, baud=args.baud, timeout=(None if args.color_timeout <= 0 else args.color_timeout))
    write_colors_json(colors, args.colors_json)

    if args.run:
        # Optionally wait for tilt trigger before running
        if args.wait_for_tilt:
            ok = wait_for_token(port, token=args.tilt_token, baud=args.baud, timeout=(None if args.tilt_timeout <= 0 else args.tilt_timeout))
            if not ok:
                print("Did not receive tilt trigger; exiting without rendering.")
                return
        # Build command to run marbler with preview (default settings)
        import subprocess
        cmd = [sys.executable, MARBLER]
        if args.input:
            cmd += ["--input", args.input]
        cmd += ["--colors", args.colors_json, "--output", args.output]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
