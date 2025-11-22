#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trap_tape_live — MT4 shared-memory → PNG+NDJSON (vision tape) writer
- EURUSD feed via your FeedSnapshot (Win SHM + mutex)
- Outputs under: E:\\eurusd\\trap_zone\\vision_model
- Exactly 5 decimals on all price-like fields (0.00000). Never more.

USAGE
=====
# simplest: just import (auto-starts a daemon thread)
import trap_tape_live

# or explicit:
import trap_tape_live as ttl
ttl.start()      # no-op if already running
...
ttl.stop()       # clean shutdown
"""


import os, sys, math, time, json, zipfile, ctypes, threading, atexit
from ctypes import wintypes, Structure, c_double, c_uint32, c_uint64, sizeof
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

# ----------------- Config -----------------
MAP_NAME   = r"Local\EURUSD_LIVE_FEED_V1"
MUTEX_NAME = r"Local\EURUSD_LIVE_FEED_MUTEX_V1"

OUT_DIR = Path(r"E:\eurusd\trap_zone\vision_model").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR = os.getenv("PAIR", "eurusd").lower()

IMG_W, IMG_H = 240, 200
MAIN_H       = 160
BOTTOM_H     = IMG_H - MAIN_H
SEC_PER_BLOCK = int(os.getenv("SEC_PER_BLOCK", "120"))  # 120 bins = 0.5s x 60s
if SEC_PER_BLOCK <= 0 or IMG_W % SEC_PER_BLOCK != 0:
    raise ValueError("SEC_PER_BLOCK must be >0 and divide IMG_W")
PX_PER_SEC   = IMG_W // SEC_PER_BLOCK  # usually 2 px per second-bin

PRICE_PER_PX = float(os.getenv("PRICE_PER_PX", "0.00001"))  # 1 pipette/px base
VISUAL_ZOOM  = max(0.1, float(os.getenv("VISUAL_ZOOM", "2.0")))
_PRICE_PER_PX_VIS = PRICE_PER_PX / VISUAL_ZOOM

CLAMP_FLOW = True
Y_REF = MAIN_H // 2

STD_WINDOW        = int(os.getenv("STD_WINDOW", "8"))
MAX_HAZE_THICK_PX = int(os.getenv("MAX_HAZE_THICK_PX", "6"))
VEL_INTENSITY_MIN = 0.35
VEL_INTENSITY_MAX = 1.00

POLL_HZ = float(os.getenv("POLL_HZ", "60.0"))  # ~60Hz is fine; mutex is cheap
SLEEP_S = 1.0 / max(1.0, POLL_HZ)

DOW_NAMES = {1:"mon", 2:"tue", 3:"wed", 4:"thu", 5:"fri"}

# Colors
CLR_BG       = (0,0,0)
CLR_MID_BASE = (180,220,255)
CLR_UP       = (80,160,255)
CLR_DN       = (255,80,80)
CLR_FL       = (140,140,140)
CLR_VEL      = (0,200,0)
CLR_VOL      = (255,255,255)
CLR_SEP      = (40,40,40)
CLR_YREF     = (60,60,60)
CLR_ANCHOR   = (255,255,255)
CLR_HAZE     = (200,200,200,32)

# ----------------- 5-decimal quantizer -----------------
def q5(x: float) -> float:
    """Return a float with exactly 5 places when dumped (no extra tails)."""
    # Faster than Decimal here; keeps NDJSON numeric (not strings).
    try:
        return float(f"{float(x):.5f}")
    except Exception:
        return 0.0

def q5_seq(seq):
    return [q5(v) for v in seq]

# ----------------- Win32 glue -----------------
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
LPSECURITY_ATTRIBUTES = ctypes.c_void_p
HANDLE = wintypes.HANDLE
DWORD  = wintypes.DWORD
BOOL   = wintypes.BOOL
LPCWSTR = wintypes.LPCWSTR
LPVOID = wintypes.LPVOID
SIZE_T = ctypes.c_size_t

WAIT_OBJECT_0 = 0x00000000
INFINITE = 0xFFFFFFFF

CreateMutexW = kernel32.CreateMutexW
CreateMutexW.argtypes = (LPSECURITY_ATTRIBUTES, BOOL, LPCWSTR)
CreateMutexW.restype  = HANDLE

OpenMutexW = kernel32.OpenMutexW
OpenMutexW.argtypes = (DWORD, BOOL, LPCWSTR)
OpenMutexW.restype  = HANDLE

WaitForSingleObject = kernel32.WaitForSingleObject
WaitForSingleObject.argtypes = (HANDLE, DWORD)
WaitForSingleObject.restype  = DWORD

ReleaseMutex = kernel32.ReleaseMutex
ReleaseMutex.argtypes = (HANDLE,)
ReleaseMutex.restype  = BOOL

CreateFileMappingW = kernel32.CreateFileMappingW
CreateFileMappingW.argtypes = (HANDLE, LPSECURITY_ATTRIBUTES, DWORD, DWORD, DWORD, LPCWSTR)
CreateFileMappingW.restype  = HANDLE

OpenFileMappingW = kernel32.OpenFileMappingW
OpenFileMappingW.argtypes = (DWORD, BOOL, LPCWSTR)
OpenFileMappingW.restype  = HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = (HANDLE, DWORD, DWORD, DWORD, SIZE_T)
MapViewOfFile.restype  = LPVOID

UnmapViewOfFile = kernel32.UnmapViewOfFile
UnmapViewOfFile.argtypes = (LPVOID,)
UnmapViewOfFile.restype  = BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = (HANDLE,)
CloseHandle.restype  = BOOL

PAGE_READWRITE     = 0x04
FILE_MAP_ALL_ACCESS = 0xF001F
MUTEX_ALL_ACCESS    = 0x1F0001

# --------------- Shared struct ---------------
class FeedSnapshot(Structure):
    _pack_ = 1
    _fields_ = [
        ("bid", c_double),
        ("ask", c_double),
        ("mid", c_double),
        ("spread", c_double),
        ("tick_ms", c_uint64),     # GetTickCount64
        ("mt5_ms",  c_uint64),     # epoch ms (preferred clock)
        ("m1_open",  c_double),
        ("m1_high",  c_double),
        ("m1_low",   c_double),
        ("m1_close", c_double),
        ("m1_close_ts", c_uint64), # epoch seconds of *closed* M1
        ("m5_open",  c_double),
        ("m5_high",  c_double),
        ("m5_low",   c_double),
        ("m5_close", c_double),
        ("m5_close_ts", c_uint64),
        ("seq", c_uint32),
    ]

class LiveFeedReader:
    def __init__(self):
        self.hMutex = OpenMutexW(MUTEX_ALL_ACCESS, False, MUTEX_NAME)
        if not self.hMutex:
            self.hMutex = CreateMutexW(None, False, MUTEX_NAME)
        if not self.hMutex:
            raise OSError("mutex open/create failed")

        self.hMap = OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, MAP_NAME)
        if not self.hMap:
            self.hMap = CreateFileMappingW(HANDLE(-1), None, PAGE_READWRITE, 0, sizeof(FeedSnapshot), MAP_NAME)
        if not self.hMap:
            raise OSError("mapping open/create failed")

        view = MapViewOfFile(self.hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(FeedSnapshot))
        if not view:
            raise OSError("MapViewOfFile failed")
        self._view = view
        self._buf = FeedSnapshot.from_address(view)

    def read(self) -> FeedSnapshot:
        WaitForSingleObject(self.hMutex, INFINITE)
        snap = FeedSnapshot()
        ctypes.memmove(ctypes.addressof(snap), ctypes.addressof(self._buf), sizeof(FeedSnapshot))
        ReleaseMutex(self.hMutex)
        return snap

    def close(self):
        if getattr(self, "_view", None):
            UnmapViewOfFile(self._view); self._view = None
        if getattr(self, "hMap", None):
            CloseHandle(self.hMap); self.hMap = None
        if getattr(self, "hMutex", None):
            CloseHandle(self.hMutex); self.hMutex = None

# ----------------- Helpers -----------------
def _weekday_label(dt_utc: datetime) -> Optional[str]:
    return DOW_NAMES.get(dt_utc.isoweekday())

def _weekday_paths(label: str):
    z = OUT_DIR / f"market_tape_{label}.zip"
    j = OUT_DIR / f"market_tape_{label}.ndjson"
    t = OUT_DIR / f"market_tape_{label}.tells.ndjson"
    tmp = OUT_DIR / "tmp" / label
    tmp.mkdir(parents=True, exist_ok=True)
    return z, j, t, tmp

def _append_lines(path: Path, lines: List[str]):
    mode = "a" if path.exists() else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for ln in lines:
            fh.write(ln + "\n")

def _price_to_y(price: float, baseline: float) -> int:
    dy = (price - baseline) / _PRICE_PER_PX_VIS
    y  = MAIN_H - 1 - int(round(dy))
    return 0 if y < 0 else (MAIN_H-1 if y >= MAIN_H else y)

def _draw_slope_baseline(g: ImageDraw.ImageDraw, y0: int, y1: int):
    clr = CLR_FL if y1 == y0 else (CLR_UP if y1 < y0 else CLR_DN)
    g.line((0, y0, IMG_W-1, y1), fill=clr)
    g.point((0, y0), fill=CLR_ANCHOR)
    g.point((IMG_W-1, y1), fill=CLR_ANCHOR)

def _render_block(mids: List[float], dmids: List[float], baseline: float) -> Image.Image:
    img = Image.new("RGBA", (IMG_W, IMG_H), CLR_BG + (255,))
    g = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", (IMG_W, IMG_H), (0,0,0,0))
    go = ImageDraw.Draw(overlay)

    open_mid, close_mid = (mids[0], mids[-1]) if mids else (0.0, 0.0)
    y_open  = _price_to_y(open_mid, baseline)
    y_close = _price_to_y(close_mid, baseline)
    _draw_slope_baseline(g, y_open, y_close)

    if CLAMP_FLOW:
        g.line((0, Y_REF, IMG_W-1, Y_REF), fill=CLR_YREF)

    # haze via rolling std (VISUAL_ZOOM aware)
    stds = [0.0]*SEC_PER_BLOCK
    if len(mids) == SEC_PER_BLOCK and STD_WINDOW > 1:
        half = STD_WINDOW//2
        for i in range(SEC_PER_BLOCK):
            a = max(0, i-half); b = min(SEC_PER_BLOCK, i+half+1)
            w = b-a
            if w > 1:
                m = sum(mids[a:b])/w
                var = sum((mids[k]-m)**2 for k in range(a,b))/max(1, w-1)
                stds[i] = math.sqrt(var)
    max_std = max(stds) if stds else 0.0
    for i in range(SEC_PER_BLOCK):
        if max_std <= 0: break
        x0 = i*PX_PER_SEC
        x1 = x0 + PX_PER_SEC - 1
        y_mid = _price_to_y(mids[i], baseline)
        thick_px = int(round(min(MAX_HAZE_THICK_PX, (stds[i] / _PRICE_PER_PX_VIS))))
        alpha = int(round(32 * (stds[i] / max_std))) if max_std > 0 else 0
        if thick_px > 0 and alpha > 0:
            y0 = max(0, y_mid - thick_px)
            y1 = min(MAIN_H - 1, y_mid + thick_px)
            go.rectangle((x0, y0, x1, y1), fill=(CLR_HAZE[0], CLR_HAZE[1], CLR_HAZE[2], alpha))

    # mid line + velocity intensity
    max_abs = max((abs(d) for d in dmids), default=0.0)
    prev_xy = None
    for i in range(SEC_PER_BLOCK):
        x1 = i*PX_PER_SEC + (PX_PER_SEC-1)
        y1 = _price_to_y(mids[i], baseline)
        inten = (abs(dmids[i]) / max_abs) if max_abs > 0 else 0.0
        inten = max(VEL_INTENSITY_MIN, min(VEL_INTENSITY_MAX, inten))
        r = int(CLR_MID_BASE[0]*inten); gcol = int(CLR_MID_BASE[1]*inten); b = int(CLR_MID_BASE[2]*inten)
        if prev_xy is not None:
            g.line((prev_xy[0], prev_xy[1], x1, y1), fill=(r,gcol,b,255))
        else:
            g.point((x1, y1), fill=(r,gcol,b,255))
        if PX_PER_SEC > 1:
            g.line((x1-(PX_PER_SEC-1), y1, x1, y1), fill=(r,gcol,b,255))
        prev_xy = (x1, y1)

    # separator
    g.line((0, MAIN_H, IMG_W-1, MAIN_H), fill=CLR_SEP)

    # velocity trace
    vmax = max(1e-12, max(abs(v) for v in dmids))
    y_mid = MAIN_H + BOTTOM_H//2
    prev_xy = None
    for i in range(SEC_PER_BLOCK):
        x = i*PX_PER_SEC
        amp = int(round((dmids[i] / vmax) * (BOTTOM_H//2 - 2)))
        y = y_mid - amp
        if prev_xy is not None:
            g.line((prev_xy[0], prev_xy[1], x, y), fill=CLR_VEL)
        g.line((x, y, x+PX_PER_SEC-1, y), fill=CLR_VEL)
        prev_xy = (x+PX_PER_SEC-1, y)

    # “volume” bars = |dmids|
    vols = [abs(v) for v in dmids]
    vbar_max = max(1e-12, max(vols))
    y_bottom = IMG_H - 1
    for i in range(SEC_PER_BLOCK):
        x0 = i*PX_PER_SEC
        x1 = x0 + PX_PER_SEC - 1
        h  = int(round((vols[i] / vbar_max) * (BOTTOM_H - 2)))
        y0 = y_bottom
        y1 = max(MAIN_H+1, y_bottom - h)
        g.line((x0, y0, x0, y1), fill=CLR_VOL)
        if x1 != x0:
            g.line((x1, y0, x1, y1), fill=CLR_VOL)

    return Image.alpha_composite(img, overlay)

def _compute_tells(mids: List[float], dmids: List[float]) -> Dict[str, float]:
    N = len(mids) or 1
    px1 = _PRICE_PER_PX_VIS
    stall = sum(1 for d in dmids if abs(d) <= px1) / N
    comp_px = (max(mids)-min(mids)) / max(1e-12, _PRICE_PER_PX_VIS)
    up_px = max(0.0, max(dmids)) / max(1e-12, _PRICE_PER_PX_VIS)
    dn_px = max(0.0, -min(dmids)) / max(1e-12, _PRICE_PER_PX_VIS)

    flips, last = 0, 0
    for d in dmids:
        s = 1 if d>0 else (-1 if d<0 else 0)
        if s and last and s != last: flips += 1
        if s: last = s
    zigzag = flips / max(1, N-1)

    # curvature “kinks”
    kinks = 0
    for i in range(1, N-1):
        dd = (dmids[i] - dmids[i-1]) / max(1e-12, _PRICE_PER_PX_VIS)
        if abs(dd) >= 2.0: kinks += 1

    dm_px = [d / max(1e-12, _PRICE_PER_PX_VIS) for d in dmids]
    mean = sum(dm_px)/N
    var = sum((x-mean)**2 for x in dm_px)/max(1, N-1)
    std = math.sqrt(var)
    entropy = 1.0/(1.0 + std)

    # Keep non-price metrics as is (already low precision); prices quantized below if needed
    return dict(
        stall_pct=float(round(stall,4)),
        compress_px=float(round(comp_px,3)),
        burst_up_px=float(round(up_px,3)),
        burst_dn_px=float(round(dn_px,3)),
        zigzag=float(round(zigzag,4)),
        kink=int(kinks),
        entropy=float(round(entropy,4)),
    )

def _zip_add(zpath: Path, arcname: str, src_path: Path):
    mode = "a" if zpath.exists() and zpath.stat().st_size>0 else "w"
    with zipfile.ZipFile(zpath, mode=mode, compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        if arcname not in zf.namelist():
            zf.write(str(src_path), arcname=arcname)

# ----------------- Minute aggregator -----------------
class MinuteAgg:
    """
    Maintains 120 0.5s bins for the active minute. We finalize either when:
      - producer reports a new m1_close_ts (!= last), OR
      - utc-minute key (YYYYMMDDHHMM) changes (fallback).
    """
    def __init__(self):
        self.curr_key: Optional[int] = None   # YYYYMMDDHHMM
        self.curr_m1_close_ts: int = 0
        self.mids = [math.nan]*SEC_PER_BLOCK
        self.prev_close_for_baseline: Optional[float] = None

    @staticmethod
    def _key_from_ms(epoch_ms: int) -> int:
        dt = datetime.fromtimestamp(epoch_ms/1000.0, tz=timezone.utc)
        return int(dt.strftime("%Y%m%d%H%M"))

    @staticmethod
    def _idx_from_ms(epoch_ms: int) -> int:
        dt = datetime.fromtimestamp(epoch_ms/1000.0, tz=timezone.utc)
        return min(SEC_PER_BLOCK-1, dt.second*2 + (1 if dt.microsecond >= 500000 else 0))

    def _filled(self) -> List[float]:
        out = list(self.mids)
        last = None
        for i in range(SEC_PER_BLOCK):
            if math.isnan(out[i]):
                if last is not None: out[i] = last
            else:
                last = out[i]
        first = next((x for x in out if not math.isnan(x)), 0.0)
        out = [first if math.isnan(x) else x for x in out]
        return out

    def add(self, epoch_ms: int, mid: float, m1_close_ts: int) -> Optional[Tuple[int, List[float], List[float], int]]:
        key = self._key_from_ms(epoch_ms)
        idx = self._idx_from_ms(epoch_ms)

        if self.curr_key is None:
            self.curr_key = key
            self.curr_m1_close_ts = m1_close_ts

        minute_changed = (key != self.curr_key)
        m1_roll = (m1_close_ts > 0 and m1_close_ts != self.curr_m1_close_ts)

        if minute_changed or m1_roll:
            filled = self._filled()
            dmids = np.diff(filled, prepend=filled[0]).tolist()
            out_key = self.curr_key
            last_close = filled[-1]
            self.prev_close_for_baseline = last_close
            # reset
            self.curr_key = key
            self.curr_m1_close_ts = m1_close_ts
            self.mids = [math.nan]*SEC_PER_BLOCK
            self.mids[idx] = mid
            return out_key, filled, dmids, (m1_close_ts if m1_roll else 0)

        self.mids[idx] = mid
        return None

# ----------------- Worker loop -----------------
class _Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="trap_tape_live")
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        reader = LiveFeedReader()
        agg = MinuteAgg()
        prev_seq = None
        try:
            while not self._stop.is_set():
                snap = reader.read()
                if prev_seq is not None and snap.seq == prev_seq:
                    time.sleep(SLEEP_S); continue
                prev_seq = snap.seq

                epoch_ms = int(snap.mt5_ms) if snap.mt5_ms else int(time.time()*1000)
                mid = snap.mid if snap.mid > 0.0 else ((snap.bid + snap.ask)*0.5 if (snap.bid>0 and snap.ask>0) else 0.0)
                if mid <= 0.0:
                    time.sleep(SLEEP_S); continue

                rolled = agg.add(epoch_ms, mid, int(snap.m1_close_ts))
                if not rolled:
                    time.sleep(SLEEP_S); continue

                out_key, mids, dmids, m1_flag = rolled
                dt = datetime.strptime(str(out_key), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                label = _weekday_label(dt)
                if not label:
                    continue  # weekend

                zip_path, ndj_path, tells_path, tmp_dir = _weekday_paths(label)

                # baseline calc (quantize at the end)
                if CLAMP_FLOW and agg.prev_close_for_baseline is not None:
                    baseline = agg.prev_close_for_baseline - (MAIN_H - 1 - Y_REF) * _PRICE_PER_PX_VIS
                else:
                    baseline = min(mids) if mids else 0.0

                # render
                img = _render_block(mids, dmids, baseline)
                tkey_png = int(dt.strftime("%Y%m%d%H%M"))  # minute-resolution file id
                fname = f"{tkey_png}.png"
                out_png = tmp_dir / fname
                img.save(str(out_png), format="PNG")
                _zip_add(zip_path, fname, out_png)
                try:
                    out_png.unlink(missing_ok=True)
                except Exception:
                    pass

                # quantize price fields to 5dp
                open_, high_, low_, close_ = q5(mids[0]), q5(max(mids)), q5(min(mids)), q5(mids[-1])
                mid_price = close_
                price_per_px_q = q5(_PRICE_PER_PX_VIS)
                baseline_q = q5(baseline)

                # NDJSON (prices strictly 5dp)
                ndj_line = json.dumps({
                    "id": tkey_png,
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:00"),
                    "tod": dt.strftime("%H:%M:%S"),
                    "dow": dt.isoweekday(),
                    "pair": PAIR,
                    "open": open_,
                    "high": high_,
                    "low":  low_,
                    "close": close_,
                    "mid_price": mid_price,
                    "png": fname,
                    "has_velocity": True,
                    "has_spread_overlay": False,
                    "price_per_px": price_per_px_q,
                    "clamp_mode": ("prev_close@y_ref" if CLAMP_FLOW else "local_min"),
                    "y_ref": (Y_REF if CLAMP_FLOW else None),
                    "baseline_px0": baseline_q,
                    "m1_roll_seen": bool(m1_flag),
                }, ensure_ascii=False)
                _append_lines(ndj_path, [ndj_line])

                # tells sidecar (non-price metrics kept as floats; okay)
                tells = _compute_tells(mids, dmids)
                tells_line = json.dumps({
                    "id": tkey_png,
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:00"),
                    **tells
                }, ensure_ascii=False)
                _append_lines(tells_path, [tells_line])

                # light heartbeat (stdout)
                print(f"✔ {label} +1 → {fname}", flush=True)

                time.sleep(SLEEP_S)
        finally:
            reader.close()

# ----------------- Public controls -----------------
__worker: Optional[_Worker] = None
__lock = threading.Lock()

def start():
    global __worker
    with __lock:
        if __worker and __worker.is_alive():
            return
        __worker = _Worker()
        __worker.start()

def stop():
    global __worker
    with __lock:
        if __worker:
            __worker.stop()
            __worker.join(timeout=2.0)
            __worker = None

def is_running() -> bool:
    with __lock:
        return bool(__worker and __worker.is_alive())

# Clean shutdown on interpreter exit
atexit.register(stop)

# Auto-start unless explicitly disabled
if os.getenv("TRAP_TAPE_DISABLE_AUTOSTART", "0") != "1":
    start()

def start_taper():
    return start()
