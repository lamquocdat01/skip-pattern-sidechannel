"""dump_trace.py — T22a B2: talk to fw_replay over the CH343 serial port.

  python dump_trace.py --port COM9                 DUMP every LittleFS file -> results/real_trace/raw/<UTC stamp>/
  python dump_trace.py --port COM9 --cmd STATUS    print the firmware status (schedules loaded, files, state)
  python dump_trace.py --port COM9 --cmd CLEAR     delete trace_*.csv, meta.json, error.txt, armed
  python dump_trace.py --port COM9 --cmd ARM       arm a run at the next boot (refused while results exist)
  also: DISARM, INA (5 live readings), LED (6 green blinks)

DUMP verifies every file against the size and CRC-32 the firmware sends, writes dump_manifest.json (host UTC time,
port, per-file size/crc/sha256) and refuses to call the dump good if any file mismatches. Opening the port does not
reset the board (DTR/RTS are held released), so a finished run is never disturbed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parent.parent / "results" / "real_trace" / "raw"


def open_port(port: str, baud: int) -> serial.Serial:
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 2.0
    s.dtr = False                       # keep EN / IO0 released: no auto-reset on open
    s.rts = False
    s.open()
    time.sleep(0.3)
    s.reset_input_buffer()
    return s


def send(s: serial.Serial, cmd: str) -> None:
    s.write((cmd + "\n").encode("ascii"))
    s.flush()


def read_until(s: serial.Serial, marker: str, timeout: float) -> list[str]:
    lines, t_end = [], time.time() + timeout
    while time.time() < t_end:
        raw = s.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        lines.append(line)
        if line.startswith(marker):
            return lines
    raise TimeoutError(f"no '{marker}' within {timeout:.0f} s; got: {lines[-5:]}")


def read_exact(s: serial.Serial, n: int, timeout: float) -> bytes:
    buf, t_end = bytearray(), time.time() + timeout
    while len(buf) < n and time.time() < t_end:
        buf += s.read(n - len(buf))
    if len(buf) != n:
        raise TimeoutError(f"short read {len(buf)}/{n}")
    return bytes(buf)


def simple(s: serial.Serial, cmd: str) -> None:
    send(s, cmd)
    marker = {"STATUS": "STATUS_END", "LED": "LED_TEST_DONE"}.get(cmd)
    if marker:
        lines = read_until(s, marker, 30)
    else:
        time.sleep(1.5 if cmd != "INA" else 2.0)
        lines = s.read(s.in_waiting or 1).decode("utf-8", errors="replace").splitlines()
    print("\n".join(lines))


def dump(s: serial.Serial, port: str, raw_dir: Path) -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = raw_dir / stamp
    out.mkdir(parents=True, exist_ok=False)
    send(s, "DUMP")
    head = read_until(s, "DUMP_BEGIN", 10)[-1]
    n_files = int(head.split()[1])
    manifest = dict(host_utc=stamp, port=port, n_files=n_files, files=[])
    bad = 0
    for _ in range(n_files):
        hdr = read_until(s, "FILE ", 30)[-1].split()
        name, size, crc = hdr[1], int(hdr[2]), int(hdr[3], 16)
        data = read_exact(s, size, timeout=60 + size / 5000)
        tail = read_until(s, "ENDFILE", 10)
        if any(t.strip() not in ("", "ENDFILE") for t in tail):
            print(f"  ! unexpected bytes after {name}: {tail}")
        ok = (zlib.crc32(data) & 0xFFFFFFFF) == crc
        bad += not ok
        (out / name).write_bytes(data)
        manifest["files"].append(dict(name=name, size=size, crc32=f"{crc:08x}", crc_ok=ok,
                                      sha256=hashlib.sha256(data).hexdigest()))
        print(f"  {name:32s} {size:9d} B  crc {'OK' if ok else 'MISMATCH'}")
    read_until(s, "DUMP_END", 10)
    (out / "dump_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{n_files} file(s) -> {out}  ({'ALL OK' if not bad else f'{bad} CRC MISMATCH'})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", default="DUMP", choices=["DUMP", "STATUS", "CLEAR", "ARM", "DISARM", "INA", "LED"])
    ap.add_argument("--out", type=Path, default=RAW_DIR, help="parent folder of the dump (default results/real_trace/raw)")
    a = ap.parse_args()
    with open_port(a.port, a.baud) as s:
        if a.cmd == "DUMP":
            return dump(s, a.port, a.out)
        simple(s, a.cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
