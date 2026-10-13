"""Capture raw serial output from the ESP32-S3 for a fixed time (T22a diagnostics).

Usage: python serial_capture.py --port COM9 --seconds 30 --out log.txt [--reset]
"""
import argparse
import time

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reset", action="store_true", help="pulse EN via RTS before capturing")
    a = ap.parse_args()

    with serial.Serial(a.port, a.baud, timeout=0.2) as s:
        if a.reset:
            s.dtr = False
            s.rts = True
            time.sleep(0.1)
            s.rts = False
        t_end = time.time() + a.seconds
        buf = bytearray()
        while time.time() < t_end:
            buf += s.read(4096)
    text = buf.decode("utf-8", errors="replace")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
