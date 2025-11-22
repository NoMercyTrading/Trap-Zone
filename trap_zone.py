#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# 0) IMPORTS
# ============================================================

import ctypes
import struct
import sys
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pymysql
import io
import contextlib
import warnings

import tape
from brain import run_brain  # Desk 6 brain module

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================
# 1) PATHS & HUD
# ============================================================

TRAP_JSON_DIR       = r"E:/eurusd/json"
TRAP_ACTIVE_PATH    = os.path.join(TRAP_JSON_DIR, "trap_active.json")
TRAP_ZONE_PATH      = os.path.join(TRAP_JSON_DIR, "trap_zone.json")
CDX_WIDGET_PATH     = os.path.join(TRAP_JSON_DIR, "cdx_widget.json")
BLOCK3_STATUS_PATH  = os.path.join(TRAP_JSON_DIR, "block3_status.json")  # 👈 match this name

HUD = {
    "desk1": {},
    "desk2": {},
    "desk3": {},
    "desk4": {},
    "desk5": {},
    "desk6": {},
    "desk7": {},
    "desk8": {},
    "meta": {
        "cycle_id": None,
        "last_update": None,
    },
}

PIP = 0.00010
SEP = "────────────────────────────────────────────────────────"

DB_CFG = dict(
    host="localhost",
    user="ai",
    password="8662533",
    database="EURUSD",
    charset="utf8mb4",
    autocommit=True,
)

# ============================================================
# 2) SHARED MEMORY FEED (SINGLE SOURCE OF TRUTH)
# ============================================================

MAP_NAME = r"Local\EURUSD_LIVE_FEED_V1"

# V3 layout (with m1/m5 timestamps)
STRUCT_V3 = struct.Struct("<ddddQQddddQddddQL")
# V2 layout (no per-bar timestamps)
STRUCT_V2 = struct.Struct("<ddddQQddddddddL")
# V1 layout (very old)
STRUCT_V1 = struct.Struct("<dddLL")

# Win32 API
k32 = ctypes.WinDLL("kernel32", use_last_error=True)

OpenFileMappingW = k32.OpenFileMappingW
OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
OpenFileMappingW.restype = ctypes.c_void_p

MapViewOfFile = k32.MapViewOfFile
MapViewOfFile.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_size_t,
]
MapViewOfFile.restype = ctypes.c_void_p

UnmapViewOfFile = k32.UnmapViewOfFile
UnmapViewOfFile.argtypes = [ctypes.c_void_p]
UnmapViewOfFile.restype = ctypes.c_int

CloseHandle = k32.CloseHandle
CloseHandle.argtypes = [ctypes.c_void_p]
CloseHandle.restype = ctypes.c_int

FILE_MAP_READ = 0x0004


def open_mapping():
    """Open shared memory and detect layout V3 → V2 → V1."""
    h_map = OpenFileMappingW(FILE_MAP_READ, False, MAP_NAME)
    if not h_map:
        err = ctypes.get_last_error()
        raise OSError(err, f"OpenFileMappingW failed (code={err})")

    # Try V3
    ptr = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, STRUCT_V3.size)
    if ptr:
        return h_map, ptr, STRUCT_V3, "V3"

    # Try V2
    ptr = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, STRUCT_V2.size)
    if ptr:
        return h_map, ptr, STRUCT_V2, "V2"

    # Try V1
    ptr = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, STRUCT_V1.size)
    if ptr:
        return h_map, ptr, STRUCT_V1, "V1"

    CloseHandle(h_map)
    err = ctypes.get_last_error()
    raise OSError(err, "MapViewOfFile failed for all layouts")


def fmt_ts_ms(ms: int) -> str:
    """
    Format feed timestamp (ms or sec) into 'YYYY-mm-dd HH:MM:SS'
    using the raw MT4 server clock with ZERO timezone conversion.
    """
    if not ms:
        return "n/a"

    try:
        # Normalize to seconds
        if ms < 10**11:   # seconds
            ts = float(ms)
        else:             # milliseconds
            ts = ms / 1000.0
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


def parse_snapshot(ptr, S, version: str) -> Optional[Dict[str, Any]]:
    """Read one snapshot from shared memory (NO system time)."""
    raw = ctypes.string_at(ptr, S.size)
    data = S.unpack(raw)

    if version == "V3":
        (
            bid, ask, mid, spread,
            tick_ms, mt5_ms,
            m1_o, m1_h, m1_l, m1_c, m1_ts,
            m5_o, m5_h, m5_l, m5_c, m5_ts,
            seq
        ) = data

        snap = {
            "version": "V3",
            "seq": int(seq),
            "bid": float(bid),
            "ask": float(ask),
            "mid": float(mid),
            "spread": float(spread),
            "tick_ms": int(tick_ms),
            "mt5_ms": int(mt5_ms),
            "server_time_str": fmt_ts_ms(mt5_ms) if mt5_ms else "n/a",
            "m1": None,
            "m5": None,
        }

        if m1_ts:
            snap["m1"] = {
                "o": float(m1_o),
                "h": float(m1_h),
                "l": float(m1_l),
                "c": float(m1_c),
                "ts": int(m1_ts),
                "time_str": fmt_ts_ms(m1_ts),
            }
        if m5_ts:
            snap["m5"] = {
                "o": float(m5_o),
                "h": float(m5_h),
                "l": float(m5_l),
                "c": float(m5_c),
                "ts": int(m5_ts),
                "time_str": fmt_ts_ms(m5_ts),
            }
        return snap

    if version == "V2":
        (
            bid, ask, mid, spread,
            tick_ms, mt5_ms,
            m1_o, m1_h, m1_l, m1_c,
            m5_o, m5_h, m5_l, m5_c,
            seq
        ) = data

        return {
            "version": "V2",
            "seq": int(seq),
            "bid": float(bid),
            "ask": float(ask),
            "mid": float(mid),
            "spread": float(spread),
            "tick_ms": int(tick_ms),
            "mt5_ms": int(mt5_ms),
            "server_time_str": fmt_ts_ms(mt5_ms) if mt5_ms else "n/a",
            "m1": {
                "o": float(m1_o),
                "h": float(m1_h),
                "l": float(m1_l),
                "c": float(m1_c),
                "ts": 0,
                "time_str": "n/a",
            },
            "m5": {
                "o": float(m5_o),
                "h": float(m5_h),
                "l": float(m5_l),
                "c": float(m5_c),
                "ts": 0,
                "time_str": "n/a",
            },
        }

    # V1
    bid, ask, mid, tick_ms, seq = data
    spread = ask - bid if (bid and ask) else 0.0
    return {
        "version": "V1",
        "seq": int(seq),
        "bid": float(bid),
        "ask": float(ask),
        "mid": float(mid),
        "spread": float(spread),
        "tick_ms": int(tick_ms),
        "mt5_ms": 0,
        "server_time_str": "n/a",
        "m1": None,
        "m5": None,
    }

# ============================================================
# 3) TIME HELPERS (ONE CLOCK: FEED ONLY)
# ============================================================

# Single source of truth:
# - TIME_REF is ALWAYS whatever came from shared memory (or forecast/etc if you set it).
# - No system clock, no UTC/local conversion.
# - If TIME_REF is empty, the system "does not know" the time.

TIME_REF: str = ""  # raw 'YYYY-mm-dd HH:MM:SS' from feed, or "" if unknown


def set_server_time_from_memory(t: str):
    """
    Set the one true clock from feed (string 'YYYY-mm-dd HH:MM:SS').

    If t is falsy/empty, we treat time as unknown.
    """
    global TIME_REF
    TIME_REF = t.strip() if t else ""


def get_server_time() -> str:
    """
    Return the canonical server time string.

    If empty string, time is unknown. No fake default dates.
    """
    return TIME_REF


def parse_ts(s) -> int:
    """
    Convert a server time string into a comparable integer.

    Contract:
      - No timezone / UTC / local conversion.
      - We only care about ordering and "freshness" comparisons.
      - Implementation: strip to digits and pack as YYYYmmddHHMMSS → int.
        If we can't parse, return 0 (unknown).
    """
    if s is None:
        return 0

    # Allow existing callers to pass datetime/int/float but STILL
    # avoid system timezone magic.
    if isinstance(s, datetime):
        # 'YYYYmmddHHMMSS' from strftime, then digits→int
        ds = s.strftime("%Y%m%d%H%M%S")
        try:
            return int(ds)
        except Exception:
            return 0

    if isinstance(s, (int, float)):
        # Already some monotone number; just coerce to int.
        return int(s)

    s = str(s).strip()
    if not s:
        return 0

    # Keep digits only (YYYY-mm-dd HH:MM:SS → YYYYmmddHHMMSS)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return 0

    # Normalize length to 14 (YYYYmmddHHMMSS) by padding/truncating.
    if len(digits) > 14:
        digits = digits[:14]
    elif len(digits) < 14:
        digits = digits.ljust(14, "0")

    try:
        return int(digits)
    except Exception:
        return 0


def server_now_ts() -> int:
    """
    Monotone integer from the ONE clock (memory).

    0 means: time unknown.
    """
    return parse_ts(get_server_time())

# ============================================================
# 4) CONSOLE HELPERS (HUD / CLEAR)
# ============================================================

STD_OUTPUT_HANDLE = -11

if os.name == "nt":
    import ctypes.wintypes as wt
    _h_stdout = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

    class COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.c_short),
                    ("Y", ctypes.c_short)]

    def cursor_home():
        """Move cursor to top-left without clearing (Windows API)."""
        ctypes.windll.kernel32.SetConsoleCursorPosition(_h_stdout, COORD(0, 0))
else:
    def cursor_home():
        """Move cursor to top-left (ANSI)."""
        sys.stdout.write("\x1b[H")
        sys.stdout.flush()


def clear_screen():
    """Full screen clear."""
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


def enable_ansi():
    """Enable ANSI escape codes on Windows console (for static HUD)."""
    try:
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass

# ============================================================
# 5) IN-MEMORY "DB" (RUNTIME STATE)
# ============================================================

@dataclass
class MemoryDB:
    ea_settings: Dict[str, Any] = field(default_factory=dict)

    # LIVE FEED (shared memory)
    mem_tick: List[Dict[str, Any]] = field(default_factory=list)   # bid/ask/mid/spread
    mem_m1:   List[Dict[str, Any]] = field(default_factory=list)   # last M1 candle
    mem_m5:   List[Dict[str, Any]] = field(default_factory=list)   # last M5 candle
    master_time: List[Dict[str, Any]] = field(default_factory=list)

    # TABLE-DERIVED DATA
    md_ngz:           Dict[str, float]      = field(default_factory=dict)
    heat_map_pivots:  List[Dict[str, Any]]  = field(default_factory=list)
    pending_orders:   List[Dict[str, Any]]  = field(default_factory=list)
    orders:           List[Dict[str, Any]]  = field(default_factory=list)
    alerts:           List[Dict[str, Any]]  = field(default_factory=list)

    # INTERNAL STATE
    ppmap: Dict[int, Dict[str, Any]] = field(default_factory=dict)


db: MemoryDB = MemoryDB()
cdx: Dict[str, Any] = {}
pivot_lock_cache: Dict[int, Dict[str, Any]] = {}


def fmt5(x: float) -> str:
    return f"{float(x):.5f}"

# ============================================================
# 6) SQL / FORECAST / LOCK HELPERS
# ============================================================

def load_heat_map_pivots(db_: MemoryDB):
    """Load pivots from SQL into memory (for Desk 1 filters)."""
    try:
        conn = pymysql.connect(**DB_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    pivot_price,
                    pivot_type,
                    COALESCE(pivot_role, 0) AS pivot_role,
                    COALESCE(score, 0)       AS score
                FROM heat_map_pivots
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall() or []
            db_.heat_map_pivots = rows
            print(f"🗺 Loaded {len(rows)} pivots from SQL heat_map_pivots.")
    except Exception as e:
        print(f"⚠️ Could not load pivots: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_pending_orders(db_: MemoryDB):
    """Load active pending orders from SQL into memory."""
    try:
        conn = pymysql.connect(**DB_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT id, pivot_id, pivot_price, bias, trigger_line,
                       upper_threshold, lower_threshold, status,
                       trap_state, pivot_state, case_closed, created_at
                FROM pending_orders
                WHERE case_closed = 0
                ORDER BY id DESC
            """)
            rows = cur.fetchall()
            db_.pending_orders = rows or []
            print(f"📦 Loaded {len(rows)} active pending orders from SQL.")
    except Exception as e:
        print(f"⚠️ Could not load pending_orders: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def calc_zone(pp: float, bias: int, pip: float = PIP) -> Tuple[float, float, float]:
    if bias == 1:
        lower = pp - 5 * pip
        upper = pp + 3 * pip
        trigger = pp + 3 * pip
    else:
        lower = pp - 3 * pip
        upper = pp + 5 * pip
        trigger = pp - 3 * pip
    return upper, lower, trigger


def load_forecast_json(path: str = "E:/eurusd/candles/vision_forecast.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, "json_not_found"
    except Exception:
        return None, "json_decode_error"

    candles = (
        data.get("ensemble", {}).get("ensemble_predicted_candles")
        or data.get("ensemble", {}).get("predicted_candles")
        or data.get("forecast_candle", {}).get("predicted_candles")
        or data.get("forecast_spread", {}).get("predicted_candles")
        or []
    )

    f_time = str(data.get("forecast_time") or data.get("anchor_time") or "")
    f_ts = parse_ts(f_time) if f_time else 0

    return {"candles": candles, "time_str": f_time, "ts": f_ts}, "ok"


def forecast_is_fresh(f_ts: int, last_tick_ts: int) -> bool:
    if not f_ts:
        return False
    if last_tick_ts and f_ts <= last_tick_ts:
        return False
    now_ts = server_now_ts()
    if not now_ts:
        return True
    return (now_ts - f_ts) <= 300  # 5 minutes, same clock


def forecast_breakout_probe(
    bias: int,
    upper: float,
    lower: float,
    candles: List[Dict[str, Any]],
    lookahead: int = 5,
    buffer_pips: float = 1.0,
    pip: float = PIP,
) -> Tuple[int, int, float]:
    buf = buffer_pips * pip
    lim = min(lookahead, max(0, len(candles)))
    hits = 0
    for i in range(lim):
        pc = candles[i] or {}
        high = float(pc.get("high") or pc.get("mid_price") or pc.get("close", 0.0))
        low = float(pc.get("low") or pc.get("mid_price") or pc.get("close", 0.0))
        close = float(pc.get("close") or pc.get("mid_price") or high)
        if bias == 2:
            if high >= upper + buf or close >= upper + buf:
                hits += 1
        elif bias == 1:
            if low <= lower - buf or close <= lower - buf:
                hits += 1
    return hits, lim, buf


def write_lock_packet(pid, pp, bias, locked, upper, lower, trigger, since):
    pivot_lock_cache.clear()
    pivot_lock_cache[int(pid)] = {
        "bias": int(bias),
        "role": int(bias),
        "locked_since": since,
        "locked": int(locked),
        "pivot_price": round(float(pp), 5),
        "zone_top": round(float(upper), 5),
        "zone_bottom": round(float(lower), 5),
        "trigger_line": round(float(trigger), 5),
    }


def write_lock_packet_safe(pid, pp, bias, locked, upper, lower, trigger, since):
    if pivot_lock_cache:
        cur_id = next(iter(pivot_lock_cache))
        row = pivot_lock_cache[cur_id]
        if int(row.get("locked", 0)) == 1:
            if locked == 0 and cur_id > 0:
                row["locked"] = 0
                row["locked_since"] = since
            return
    write_lock_packet(pid, pp, bias, locked, upper, lower, trigger, since)


def ppmap_set(db_: MemoryDB, pending_id: int, pivot_id: int, play: str):
    db_.ppmap[int(pending_id)] = {
        "pivot_id": int(pivot_id),
        "play": play,
        "ts": get_server_time(),
    }


def ppmap_take(db_: MemoryDB, pending_id: int):
    return db_.ppmap.pop(int(pending_id), None)

# ============================================================
# 7) UPDATE DB FROM SNAPSHOT (LIVE FEED → INTERNAL STATE)
# ============================================================

def update_db_from_snapshot(db_: MemoryDB, snap: Dict[str, Any]):
    """
    Live feed snapshot → MemoryDB.
    This is the ONLY place that is allowed to touch the master clock.
    """
    # ---- 1) clock from feed (or nothing) ----
    server_str = snap.get("server_time_str") or ""
    if server_str and server_str != "n/a":
        set_server_time_from_memory(server_str)

    # may be "" if feed hasn't given a time yet
    now = get_server_time()

    # ---- 2) tick data (bid/ask/mid/spread) ----
    bid = float(snap.get("bid") or 0.0)
    ask = float(snap.get("ask") or 0.0)

    spr = snap.get("spread")
    if spr is None and bid and ask:
        spr = ask - bid
    spr = float(spr or 0.0)

    if snap.get("mid") is not None:
        mid = float(snap.get("mid"))
    else:
        mid = float((bid + ask) / 2.0) if (bid and ask) else 0.0

    db_.master_time = [{"mt5_time": now}]
    db_.mem_tick = [{
        "bid": bid,
        "ask": ask,
        "spread": spr,
        "mid": mid,
        "timestamp": now,   # "" means unknown clock
    }]

    # ---- 3) M1 candle (if present) ----
    m1 = snap.get("m1")
    if m1:
        t1_raw = m1.get("time_str") or ""
        t1 = t1_raw if (t1_raw and t1_raw != "n/a") else now

        db_.mem_m1 = [{
            "time":  t1,
            "open":  m1["o"],
            "high":  m1["h"],
            "low":   m1["l"],
            "close": m1["c"],
        }]
    else:
        db_.mem_m1 = []

    # ---- 4) M5 candle (if present) ----
    m5 = snap.get("m5")
    if m5:
        t5_raw = m5.get("time_str") or ""
        t5 = t5_raw if (t5_raw and t5_raw != "n/a") else now

        db_.mem_m5 = [{
            "time":  t5,
            "open":  m5["o"],
            "high":  m5["h"],
            "low":   m5["l"],
            "close": m5["c"],
        }]
    else:
        db_.mem_m5 = []

    # ---- 5) default EA settings (once) ----
    if not db_.ea_settings:
        db_.ea_settings = {
            "Master": 1,   # 1=ON, 2=OFF
            "Bias":   2,   # 1=BiasTable, 2=ManualClosest
            "Lots":   0.01,
        }

# ============================================================
# 8) DESK 0 – ONE SHOT LOADER (MEMORY + TABLES)
# ============================================================

def desk0_load_state(db_: MemoryDB):
    """
    DESK 0 – Load live feed + SQL tables into MemoryDB.
    Call this once at the start of the run.
    """
    # ---- 1) Live feed from shared memory ----
    try:
        h_map, ptr, S, version = open_mapping()
        snap = parse_snapshot(ptr, S, version)
        if snap:
            update_db_from_snapshot(db_, snap)
        else:
            print("⚠️ DESK0: snapshot is None (shared memory empty?)")
        UnmapViewOfFile(ptr)
        CloseHandle(h_map)
    except Exception as e:
        print(f"⚠️ DESK0: could not read shared memory: {e}")
        db_.mem_tick = [{
            "bid": 0.0,
            "ask": 0.0,
            "spread": 0.0,
            "mid": 0.0,
            "timestamp": get_server_time(),
        }]
        db_.mem_m1 = []
        db_.mem_m5 = []

    # ---- 2) SQL tables ----
    load_heat_map_pivots(db_)
    load_pending_orders(db_)
    # (optional) load orders if you need them for this run
    try:
        conn = pymysql.connect(**DB_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 200")
            db_.orders = cur.fetchall() or []
        conn.close()
    except Exception:
        pass

    print("DESK0: state loaded (live feed + tables).")


# ============================================================
# DESK 0 – UNIVERSAL RULES BLOCK (MEM-ONLY MARKET, SQL RULES)
# ============================================================

def desk0_universal_rules(db_: MemoryDB):
    global cdx

    # HUD wiring
    cdx.setdefault("hud", {})
    log_lines: List[str] = []

    def log(msg: str):
        log_lines.append(str(msg))

    log("🏢 DESK 0 – UNIVERSAL RULES BLOCK")
    log(SEP)

    cdx["rules_pass"] = 2
    reasons: List[str] = []

    # --------------------------------------------------------
    # 1) RULES: SQL ONLY (Master flag + open/pending checks)
    # --------------------------------------------------------
    try:
        conn = pymysql.connect(**DB_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            # MASTER toggle
            cur.execute("""
                SELECT setting_value
                FROM ea_settings
                WHERE setting_key='Master'
                LIMIT 1
            """)
            row = cur.fetchone()
            master = int(row["setting_value"]) if row and row["setting_value"] not in (None, "") else 2

            if master == 2:
                reasons.append("MASTER=2 → OFF")
            else:
                # any open orders?
                cur.execute("SELECT COUNT(*) AS n FROM orders WHERE is_closed=0")
                has_open = int(cur.fetchone()["n"])

                # any active pendings?
                cur.execute("""
                    SELECT COUNT(*) AS n
                    FROM pending_orders
                    WHERE case_closed=0
                      AND status IN ('waiting','pending')
                """)
                has_pending = int(cur.fetchone()["n"])

                if has_open > 0:
                    reasons.append("Open trades exist → OFF")
                elif has_pending > 0:
                    reasons.append("Pending orders exist → OFF")
                else:
                    cdx["rules_pass"] = 1
                    reasons.append("All rules passed → ON")

    except Exception as e:
        reasons.append(f"Desk0 SQL error → OFF ({e})")
        cdx["rules_pass"] = 2
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # 1b) SYNC: pending_orders memory mirror ← DB (ONE SOURCE)
    # --------------------------------------------------------
    try:
        conn = pymysql.connect(**DB_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT id, pivot_id, pivot_price, bias,
                       upper_threshold, lower_threshold, trigger_line,
                       trap_state, pivot_state, status, case_closed, created_at
                FROM pending_orders
                WHERE case_closed = 0
                ORDER BY id ASC
            """)
            rows = cur.fetchall() or []

        old_map = {
            int(p["id"]): p
            for p in db_.pending_orders
            if isinstance(p, dict) and "id" in p
        }

        new_list: List[Dict[str, Any]] = []
        for r in rows:
            try:
                pid = int(r["id"])
            except Exception:
                continue

            base = old_map.get(pid, {}).copy()
            base.update(
                id=pid,
                pivot_id=int(r["pivot_id"]) if r["pivot_id"] is not None else 0,
                pivot_price=float(r["pivot_price"] or 0.0),
                bias=int(r["bias"] or 0),
                upper_threshold=float(r["upper_threshold"] or 0.0),
                lower_threshold=float(r["lower_threshold"] or 0.0),
                trigger_line=float(r["trigger_line"] or 0.0),
                trap_state=int(r["trap_state"] or 0),
                pivot_state=int(r["pivot_state"] or 0),
                status=str(r["status"] or ""),
                case_closed=int(r["case_closed"] or 0),
                created_at=str(r["created_at"] or ""),
            )
            new_list.append(base)

        db_.pending_orders = new_list
        log(f"🔄 DESK 0: synced pending_orders from DB → {len(new_list)} row(s)")
    except Exception as e:
        log(f"⚠️ DESK 0: pending_orders sync failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # 2) MARKET SNAPSHOT: SHARED MEMORY ONLY (mem_tick)
    # --------------------------------------------------------
    if db_.mem_tick:
        last = db_.mem_tick[-1]

        bid = float(last.get("bid") or 0.0)
        ask = float(last.get("ask") or 0.0)

        spr = last.get("spread")
        if spr is None and bid and ask:
            spr = ask - bid
        spr = float(spr or 0.0)

        if last.get("mid") is not None:
            mid = float(last["mid"])
        else:
            mid = (bid + ask) / 2.0 if (bid and ask) else 0.0

        ts = str(last.get("timestamp") or get_server_time())

        cdx["md_cp"] = {
            "bid": bid,
            "ask": ask,
            "spread": spr,
            "mid": round(mid, 5),
            "timestamp": ts,
        }

        log(f"📡 md_cp (MEM): bid={fmt5(bid)} ask={fmt5(ask)} mid={fmt5(mid)} @ {ts}")
    else:
        cdx["md_cp"] = None
        log("⚠️ md_cp memory empty (no tick from shared memory)")

    # --------------------------------------------------------
    # 3) CLOSED CANDLES (M1): MEMORY ONLY
    # --------------------------------------------------------
    if db_.mem_m1:
        cdx["md_m1"] = db_.mem_m1[-12:]
        log(f"🕯 md_m1 (MEM) candles: {len(cdx['md_m1'])}")
    else:
        cdx["md_m1"] = []
        log("⚠️ md_m1 memory empty")

    # --------------------------------------------------------
    # 4) MT5 SERVER TIME: ONE CLOCK, FROM FEED ONLY
    # --------------------------------------------------------
    mt5_time = get_server_time()
    if not mt5_time and cdx["md_cp"] and cdx["md_cp"].get("timestamp"):
        mt5_time = str(cdx["md_cp"]["timestamp"])

    if mt5_time:
        cdx["mt5_time"] = mt5_time
        log(f"🕒 mt5_time (MEM): {mt5_time}")
    else:
        log("⚠️ mt5_time not set")

    log(f"State   : {cdx['rules_pass']} (1=ON, 2=OFF)")
    log("Reason  : " + " | ".join(reasons))
    log(f"✅ DESK 0 COMPLETE")
    log(SEP)

    # Save HUD text in cdx
    cdx["hud"]["desk0"] = "\n".join(log_lines)

    # --------------------------------------------------------
    # 5) PANEL JSON SNAPSHOT (for PHP / UI HUD)
    # --------------------------------------------------------
    try:
        panel_dir = r"E:\EURUSD\trap_zone\panel"
        os.makedirs(panel_dir, exist_ok=True)

        has_pending = any(
            p for p in db_.pending_orders
            if p.get("case_closed", 0) == 0
            and str(p.get("status", "")).lower() in ("waiting", "pending")
        )

        payload = {
            "timestamp": get_server_time(),
            "rules_pass": int(cdx.get("rules_pass", 2)),
            "reasons": reasons,
            "md_cp": cdx.get("md_cp"),
            "mt5_time": cdx.get("mt5_time"),
            "has_pending": int(has_pending),
            "hud": "\n".join(log_lines),      # full text block for PHP HUD
            "hud_lines": log_lines,           # array version if you want per-line rendering
        }

        tmp_path = os.path.join(panel_dir, "desk0_rules.json.tmp")
        final_path = os.path.join(panel_dir, "desk0_rules.json")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        os.replace(tmp_path, final_path)
    except Exception as e:
        # last resort: we still log to HUD list, but no print
        log_lines.append(f"⚠️ DESK 0: panel JSON write failed: {e}")
        cdx["hud"]["desk0"] = "\n".join(log_lines)


# ============================================================
# DESK 1 – DIRECTION FILTER (ABOVE/BELOW cp_mid)
# ============================================================

def desk1_direction_filter(db_: MemoryDB):
    """
    DESK 1 – DIRECTION FILTER

    - Uses cp_mid from cdx['md_cp']['mid'] (from Desk0 / shared memory).
    - Loads ALL pivots from db_.heat_map_pivots.
    - Finds the single closest pivot to cp_mid.
    - If that pivot is ABOVE cp_mid → side='above', pivot_bias=2 (short).
      If BELOW cp_mid → side='below', pivot_bias=1 (long).
    - Filters pivots so only the chosen side is kept.
    - Writes:
        cdx['pivot_bias']
        cdx['pivot_choice_side']
        cdx['pivot_choice_1']
        cdx['pivot_pool']          (side-filtered pivots)
        cdx['pivot_count_total']
        cdx['pivot_count_after']
        cdx['hud']['desk1']
    - Panel JSON: E:\EURUSD\trap_zone\panel\desk1_direction.json
    """
    global cdx

    cdx.setdefault("hud", {})
    log_lines: List[str] = []

    def log(msg: str):
        log_lines.append(str(msg))

    log("🧭 DESK 1 – DIRECTION FILTER (above/below cp_mid)")
    log(SEP)

    # ----- get cp_mid from Desk0 snapshot -----
    md = cdx.get("md_cp") or {}
    try:
        cp_mid = float(md.get("mid"))
    except (TypeError, ValueError):
        cp_mid = None

    if cp_mid is None or cp_mid == 0.0:
        log("⚠️ no valid cp_mid in cdx['md_cp']['mid']")
        cdx["pivot_pool"] = []
        cdx["pivot_bias"] = 0
        cdx["pivot_choice_side"] = "none"
        cdx["pivot_choice_1"] = None
        cdx["pivot_count_total"] = 0
        cdx["pivot_count_after"] = 0
        cdx["hud"]["desk1"] = "\n".join(log_lines)

        # small panel JSON
        try:
            panel_dir = r"E:\EURUSD\trap_zone\panel"
            os.makedirs(panel_dir, exist_ok=True)
            payload = {
                "timestamp": get_server_time(),
                "cp_mid": None,
                "pivot_bias": 0,
                "side": "none",
                "counts": {"total": 0, "after": 0},
                "choice": None,
                "hud": "\n".join(log_lines),
                "hud_lines": log_lines,
                "pivots": [],
            }
            tmp_path = os.path.join(panel_dir, "desk1_direction.json.tmp")
            final_path = os.path.join(panel_dir, "desk1_direction.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            log_lines.append(f"⚠️ DESK 1: panel JSON write failed: {e}")
            cdx["hud"]["desk1"] = "\n".join(log_lines)
        return

    log(f"cp_mid = {fmt5(cp_mid)}")

    # ----- build raw pivot list from db_.heat_map_pivots -----
    pivots: List[Dict[str, Any]] = []

    for r in db_.heat_map_pivots:
        try:
            pid = int(r.get("id") or 0) or None
            px = float(r.get("pivot_price") or r.get("price", 0.0) or 0.0)
        except Exception:
            continue

        if not px:
            continue

        raw_role = r.get("pivot_role")
        try:
            role = int(raw_role)
            if role not in (0, 1, 2):
                role = 0
        except Exception:
            role = 0

        score = int(r.get("score") or 0)
        pivots.append({
            "id": pid,
            "price": px,
            "role": role,      # 0=neutral, 1=upper, 2=lower
            "score": score,
            "removed_by": None,
        })

    total = len(pivots)
    cdx["pivot_count_total"] = total
    log(f"• pivots loaded: {total}")

    if not pivots:
        log("⚠️ no pivots loaded from heat_map_pivots")
        cdx["pivot_pool"] = []
        cdx["pivot_bias"] = 0
        cdx["pivot_choice_side"] = "none"
        cdx["pivot_choice_1"] = None
        cdx["pivot_count_after"] = 0
        cdx["hud"]["desk1"] = "\n".join(log_lines)
        return

    # ----- find nearest pivot to cp_mid -----
    best_pivot = None
    best_dist = None

    for p in pivots:
        d = abs(p["price"] - cp_mid)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_pivot = p

    if best_pivot is None:
        log("⚠️ no valid pivot after nearest-search")
        cdx["pivot_pool"] = []
        cdx["pivot_bias"] = 0
        cdx["pivot_choice_side"] = "none"
        cdx["pivot_choice_1"] = None
        cdx["pivot_count_after"] = 0
        cdx["hud"]["desk1"] = "\n".join(log_lines)
        return

    # ----- derive side + bias from nearest pivot -----
    if best_pivot["price"] < cp_mid:
        side = "below"
        pivot_bias = 1   # long
    elif best_pivot["price"] > cp_mid:
        side = "above"
        pivot_bias = 2   # short
    else:
        side = "flat"
        pivot_bias = 0

    log(
        f"nearest pivot: #{best_pivot['id']} @ {fmt5(best_pivot['price'])} "
        f"(dist={fmt5(best_dist or 0.0)}, side={side}, bias={pivot_bias})"
    )

    # ----- filter pivots to chosen side -----
    if side == "below":
        filtered = [p for p in pivots if p["price"] <= cp_mid]
    elif side == "above":
        filtered = [p for p in pivots if p["price"] >= cp_mid]
    else:
        filtered = list(pivots)  # flat → keep all

    after = len(filtered)

    cdx["pivot_pool"] = filtered
    cdx["pivot_bias"] = pivot_bias
    cdx["pivot_choice_side"] = side
    cdx["pivot_choice_1"] = {
        "id": best_pivot["id"],
        "price": best_pivot["price"],
    }
    cdx["pivot_choice_2"] = None
    cdx["pivot_count_after"] = after

    log(f"• pivots kept on side='{side}': {after}")

    if filtered:
        sample = []
        for p in filtered[:8]:
            s = f"#{p['id']}@{fmt5(p['price'])}"
            if p.get("role"):
                s += f"(R{p['role']})"
            sample.append(s)
        log("• sample: " + ", ".join(sample))

    log(SEP)

    # HUD text
    cdx["hud"]["desk1"] = "\n".join(log_lines)

    # ----- PANEL JSON SNAPSHOT -----
    try:
        panel_dir = r"E:\EURUSD\trap_zone\panel"
        os.makedirs(panel_dir, exist_ok=True)

        payload = {
            "timestamp": get_server_time(),
            "cp_mid": cp_mid,
            "pivot_bias": pivot_bias,
            "side": side,
            "counts": {
                "total": total,
                "after": after,
            },
            "choice": {
                "id": best_pivot["id"],
                "price": best_pivot["price"],
            },
            "hud": cdx["hud"].get("desk1", ""),
            "hud_lines": log_lines,
            "pivots": [
                {
                    "id": p.get("id"),
                    "price": float(p.get("price", 0.0)),
                    "role": int(p.get("role", 0)),
                    "score": int(p.get("score", 0)),
                    "removed_by": p.get("removed_by"),
                }
                for p in filtered
            ],
        }

        tmp_path = os.path.join(panel_dir, "desk1_direction.json.tmp")
        final_path = os.path.join(panel_dir, "desk1_direction.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, final_path)
    except Exception as e:
        log_lines.append(f"⚠️ DESK 1: panel JSON write failed: {e}")
        cdx["hud"]["desk1"] = "\n".join(log_lines)

# ============================================================
# DESK 2 – NGZ + FORECAST FILTER + REUSE BAN + ROLE/DEDUPE
#     (runs on cdx['pivot_pool'] from Desk 1)
# ============================================================

def desk2_pivot_filter(db_: MemoryDB):
    global cdx

    cdx.setdefault("hud", {})
    hud_lines: List[str] = []

    def hud_print(msg: str):
        hud_lines.append(str(msg))

    def build_summary(
        rules_pass: int,
        ngz_active: int,
        ngz_low: float,
        ngz_high: float,
        total: int,
        after: int,
        removed_ngz: int,
        reuse_removed: int,
        role_blocked: int,
        removed_fc: int,
        dup_removed: int,
        mid_live: Optional[float],
        audit_list: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.append("Desk 2 — Pivots (NGZ + forecast + reuse)")
        lines.append(f"Rules: {'PASS ✔️' if rules_pass == 1 else 'OFF ⛔'}")

        if ngz_active and ngz_low is not None and ngz_high is not None:
            lines.append(f"NGZ: ACTIVE [{fmt5(ngz_low)} - {fmt5(ngz_high)}]")
        else:
            lines.append("NGZ: INACTIVE")

        lines.append("Counts:")
        lines.append(f"Total: {total}")
        lines.append(f"After filters: {after}")
        lines.append(f"Removed NGZ: {removed_ngz}")
        lines.append(f"Reuse-removed: {reuse_removed}")
        lines.append(f"Role-blocked: {role_blocked}")
        lines.append(f"Forecast-removed: {removed_fc}")
        lines.append(f"Dedupe-removed: {dup_removed}")

        if mid_live is not None:
            where = "n/a"
            if ngz_active and ngz_low is not None and ngz_high is not None:
                if ngz_low <= mid_live <= ngz_high:
                    where = "inside NGZ"
                else:
                    where = "outside NGZ"
            lines.append(f"Mid Live: {fmt5(mid_live)} ({where})")

        kept = [p for p in audit_list if p.get("removed_by") is None]
        removed = [p for p in audit_list if p.get("removed_by") is not None]

        if kept:
            lines.append(f"Pivots kept ({len(kept)}):")
            for p in kept[:12]:
                rid = p.get("id")
                px = p.get("price", 0.0)
                role = int(p.get("role", 0) or 0)
                role_txt = f"(role {role})" if role in (1, 2) else ""
                lines.append(f"#{rid} @ {fmt5(px)} {role_txt}".rstrip())

        if removed:
            def tag(reason: Optional[str]) -> str:
                m = {
                    "ngz": "ngz",
                    "reuse": "reuse",
                    "role": "role",
                    "forecast_hits": "fc_hits",
                    "forecast_noband": "fc_noband",
                    "dedupe": "dedupe",
                    "rules_off": "rules_off",
                }
                return m.get(reason or "", reason or "unknown")

            lines.append(f"Pivots removed ({len(removed)}):")
            for p in removed[:12]:
                rid = p.get("id")
                px = p.get("price", 0.0)
                reason = tag(p.get("removed_by"))
                lines.append(f"#{rid} @ {fmt5(px)} [{reason}]")

        return "\n".join(lines)

    hud_print(f"🧾 DESK 2 – NGZ + FORECAST FILTER (DB-backed reuse)")
    hud_print(SEP)

    cdx.setdefault("pivot_pool", [])
    cdx["pivot_count_total"] = cdx.get("pivot_count_total", 0)
    cdx["pivot_count_after"] = 0
    cdx["ngz_low"] = None
    cdx["ngz_high"] = None
    cdx["ngz_active"] = 0

    starved = (cdx.get("rules_pass") != 1)

    WHITELIST_IDS = [1, 2, 11, 12]
    PIP = 0.00010
    EPS = 1.0 * PIP

    removed_ngz = kept_wl = 0
    reuse_removed = 0
    dup_removed = 0
    role_blocked = 0
    removed_fc = 0
    why_hits = 0
    why_noband = 0

    trade_day = get_server_time().split(" ")[0]
    hud_print(f"🕒 Trading day: {trade_day}")

    # ---------------- NGZ (memory first, SQL fallback) ----------------
    if "no_go_low" in db_.md_ngz and "no_go_high" in db_.md_ngz:
        low = float(db_.md_ngz["no_go_low"])
        high = float(db_.md_ngz["no_go_high"])
        if low > high:
            low, high = high, low
        cdx["ngz_low"] = low
        cdx["ngz_high"] = high
        cdx["ngz_active"] = 1
        hud_print(f"✓ NGZ (memory): [{fmt5(low)} … {fmt5(high)}]")
    else:
        try:
            conn = pymysql.connect(**DB_CFG)
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute("SELECT no_go_low,no_go_high FROM md_ngz WHERE id=1 LIMIT 1")
                row = cur.fetchone()
                if row and row["no_go_low"] is not None and row["no_go_high"] is not None:
                    low = float(row["no_go_low"])
                    high = float(row["no_go_high"])
                    if low > high:
                        low, high = high, low
                    cdx["ngz_low"] = low
                    cdx["ngz_high"] = high
                    cdx["ngz_active"] = 1
                    hud_print(f"✓ NGZ (sql): [{fmt5(low)} … {fmt5(high)}]")
                else:
                    hud_print("⚠️ md_ngz id=1 missing/bad → NGZ disabled")
        except Exception as e:
            hud_print(f"⚠️ NGZ SQL error → disabled: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ---------------- Load pivots from cdx['pivot_pool'] ----------------
    pivots: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []

    base_pool = cdx.get("pivot_pool") or []
    for p in base_pool:
        try:
            pid = int(p.get("id") or 0) or None
            px = float(p.get("price") or 0.0)
        except Exception:
            continue
        if not px:
            continue

        try:
            role = int(p.get("role", 0))
            if role not in (0, 1, 2):
                role = 0
        except Exception:
            role = 0

        score = int(p.get("score") or 0)
        row = {
            "id": pid,
            "price": px,
            "role": role,
            "score": score,
            "removed_by": p.get("removed_by"),
        }
        pivots.append(row)
        audit.append(row)

    cdx["pivot_count_total"] = len(pivots)
    hud_print(f"• Pivots in (from Desk1 side-filter): {len(pivots)}")

    # ---------------- Reuse-ban from SQL (today, price-only) ----------- 
    today_prices: List[float] = []
    try:
        conn = pymysql.connect(**DB_CFG)
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT pivot_price
                FROM pending_orders
                WHERE DATE(created_at) = %s
                """,
                (trade_day,),
            )
            rows = cur.fetchall() or []
            for r in rows:
                try:
                    pp = float(r.get("pivot_price") or 0.0)
                    if pp:
                        today_prices.append(pp)
                except Exception:
                    continue
        if today_prices:
            hud_print(
                f"📌 reuse-ban (SQL pending_orders {trade_day}): "
                f"{len(today_prices)} price(s) (±1 pip)"
            )
        else:
            hud_print(f"📌 reuse-ban: no rows in pending_orders for {trade_day}")
    except Exception as e:
        hud_print(f"⚠️ reuse-ban SQL failed → fallback to memory snapshot: {e}")
        today_prices = [
            float(p.get("pivot_price", 0.0))
            for p in db_.pending_orders
            if str(p.get("created_at", "")).startswith(trade_day)
            and int(p.get("case_closed", 0)) == 0
            and str(p.get("status", "")).lower() in ("waiting", "pending")
        ]
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # ---------------- Hard starve when rules OFF ----------------
    if starved:
        cdx["pivot_pool"] = []
        cdx["pivot_count_after"] = 0
        hud_print("⛔ rules_pass!=1 → starving pivots")
        hud_print(f"{SEP}")
        hud_print(f"CDX OUT → total: {cdx['pivot_count_total']}, after: 0")

        for p in audit:
            if p.get("removed_by") is None:
                p["removed_by"] = "rules_off"

        md = cdx.get("md_cp") or {}
        mid_live = md.get("mid")

        summary = build_summary(
            rules_pass=int(cdx.get("rules_pass", 2)),
            ngz_active=int(cdx.get("ngz_active", 0)),
            ngz_low=cdx.get("ngz_low"),
            ngz_high=cdx.get("ngz_high"),
            total=cdx["pivot_count_total"],
            after=0,
            removed_ngz=0,
            reuse_removed=0,
            role_blocked=0,
            removed_fc=0,
            dup_removed=0,
            mid_live=mid_live,
            audit_list=audit,
        )
        cdx["hud"]["desk2"] = summary

        # panel JSON (starved)
        try:
            panel_dir = r"E:\EURUSD\trap_zone\panel"
            os.makedirs(panel_dir, exist_ok=True)

            pivot_list = [
                {
                    "id": p.get("id"),
                    "price": float(p.get("price", 0.0)),
                    "role": int(p.get("role", 0)),
                    "score": int(p.get("score", 0)),
                    "removed_by": p.get("removed_by"),
                }
                for p in audit
            ]

            payload = {
                "timestamp": get_server_time(),
                "rules_pass": int(cdx.get("rules_pass", 2)),
                "ngz": {
                    "active": int(cdx.get("ngz_active", 0)),
                    "low": cdx.get("ngz_low"),
                    "high": cdx.get("ngz_high"),
                },
                "counts": {
                    "total": cdx["pivot_count_total"],
                    "after": 0,
                    "removed_ngz": 0,
                    "reuse_removed": 0,
                    "role_blocked": 0,
                    "forecast_removed": 0,
                    "dedupe_removed": 0,
                },
                "mid_live": mid_live,
                "hud": summary,
                "hud_lines": hud_lines,
                "pivots": pivot_list,
            }

            tmp_path = os.path.join(panel_dir, "desk2_pivots.json.tmp")
            final_path = os.path.join(panel_dir, "desk2_pivots.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            hud_print(f"⚠️ DESK 2: panel JSON write failed (starved): {e}")
            cdx["hud"]["desk2"] = summary

        return

    # ---------------- Stage A: NGZ filter (whitelist ok) ----------------
    filtered: List[Dict[str, Any]] = []
    if cdx["ngz_active"] == 1:
        low, high = cdx["ngz_low"], cdx["ngz_high"]
        for p in pivots:
            pid, px = p["id"], p["price"]
            if pid is not None and pid in WHITELIST_IDS:
                filtered.append(p)
                kept_wl += 1
                continue
            if low <= px <= high:
                removed_ngz += 1
                if p.get("removed_by") is None:
                    p["removed_by"] = "ngz"
            else:
                filtered.append(p)
        hud_print(f"• NGZ removed: {removed_ngz} | whitelist kept: {kept_wl}")
    else:
        filtered = pivots
        hud_print("• NGZ inactive → no removals")

    # ---------------- Stage B: reuse-ban ----------------
    def has_reuse(pp: float) -> bool:
        return any(abs(pp - q) <= EPS for q in today_prices)

    filtered2: List[Dict[str, Any]] = []
    whitelist_reuse_skipped = 0

    for p in filtered:
        pid = p["id"]

        if pid is not None and pid in WHITELIST_IDS:
            filtered2.append(p)
            whitelist_reuse_skipped += 1
            continue

        if has_reuse(p["price"]):
            reuse_removed += 1
            if p.get("removed_by") is None:
                p["removed_by"] = "reuse"
            continue

        filtered2.append(p)

    if reuse_removed or whitelist_reuse_skipped:
        hud_print(
            f"• reuse-ban (DB-backed): removed={reuse_removed}, "
            f"whitelist bypassed={whitelist_reuse_skipped}"
        )
    else:
        hud_print("• reuse-ban (DB-backed): removed=0")

    # ---------------- De-dupe by price ----------------
    def dedupe_by_price(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        nonlocal dup_removed
        by_price: Dict[float, Dict[str, Any]] = {}
        for p in rows:
            key = round(p["price"], 5)
            cur = by_price.get(key)
            if cur is None:
                by_price[key] = p
            else:
                keep_new = (
                    p.get("score", 0) > cur.get("score", 0)
                    or (
                        p.get("score", 0) == cur.get("score", 0)
                        and (p.get("id") or 1e9) < (cur.get("id") or 1e9)
                    )
                )
                if keep_new:
                    if cur.get("removed_by") is None:
                        cur["removed_by"] = "dedupe"
                    by_price[key] = p
                else:
                    if p.get("removed_by") is None:
                        p["removed_by"] = "dedupe"
        dup_removed = len(rows) - len(by_price)
        return list(by_price.values())

    filtered2 = dedupe_by_price(filtered2)
    if dup_removed:
        hud_print(f"• de-dupe by price: removed {dup_removed} duplicate pivot_price rows")

    # ---------------- Stage C0: pivot_role gating ----------------
    md = cdx.get("md_cp") or {}
    mid_live = md.get("mid")
    filtered_role: List[Dict[str, Any]] = []

    if mid_live is not None:
        for p in filtered2:
            pp = p["price"]
            role = int(p.get("role") or 0)

            if role == 1 and not (pp >= mid_live):
                role_blocked += 1
                if p.get("removed_by") is None:
                    p["removed_by"] = "role"
                continue

            if role == 2 and not (pp <= mid_live):
                role_blocked += 1
                if p.get("removed_by") is None:
                    p["removed_by"] = "role"
                continue

            filtered_role.append(p)

        hud_print(
            f"• role gating (role1=above, role2=below) "
            f"blocked={role_blocked}, kept={len(filtered_role)}"
        )
    else:
        filtered_role = filtered2
        hud_print("• role gating skipped (mid missing)")

    # ---------------- Stage C1: forecast filter ----------------
    last_tick_ts = parse_ts(md.get("timestamp", "")) if md.get("timestamp") else 0
    F, fk = load_forecast_json()
    candles = F["candles"] if (isinstance(F, dict) and "candles" in F) else []
    fresh = isinstance(F, dict) and forecast_is_fresh(int(F.get("ts", 0)), last_tick_ts)

    filtered3: List[Dict[str, Any]] = []

    if fresh and candles and mid_live is not None:
        for p in filtered_role:
            pp = p["price"]
            role = int(p.get("role") or 0)

            if role in (1, 2):
                bias_test = role
            else:
                bias_test = 1 if pp <= mid_live else 2

            upper, lower, trigger = calc_zone(pp, bias_test, PIP)

            hits, _, _ = forecast_breakout_probe(bias_test, upper, lower, candles)
            blow = hits >= 2

            band_lo, band_hi = min(pp, trigger), max(pp, trigger)
            touch = False
            for c in candles[:5]:
                hi = float(c.get("high") or c.get("mid_price") or c.get("close", pp))
                lo = float(c.get("low") or c.get("mid_price") or c.get("close", pp))
                if hi >= band_lo and lo <= band_hi:
                    touch = True
                    break
            no_band = not touch

            if not blow and not no_band:
                filtered3.append(p)
            else:
                removed_fc += 1
                if blow:
                    why_hits += 1
                    if p.get("removed_by") is None:
                        p["removed_by"] = "forecast_hits"
                elif no_band:
                    why_noband += 1
                    if p.get("removed_by") is None:
                        p["removed_by"] = "forecast_noband"

        hud_print(
            f"🔮 Forecast filter → removed={removed_fc} "
            f"(blow={why_hits}, noband={why_noband}), kept={len(filtered3)}"
        )
    else:
        filtered3 = filtered_role
        hud_print("🔮 Forecast not fresh/available OR mid missing → skipped")

    # ---------------- Finalize pivot_pool ----------------
    cdx["pivot_pool"] = filtered3
    cdx["pivot_count_after"] = len(filtered3)

    if filtered3:
        sample = []
        for p in filtered3[:5]:
            s = f"#{p['id']}@{fmt5(p['price'])}"
            if p.get("role"):
                s += f"(R{p['role']})"
            sample.append(s)
        hud_print("• sample: " + ", ".join(sample))

    hud_print(SEP)
    hud_print(
        f"CDX OUT → total: {cdx['pivot_count_total']}, after: {cdx['pivot_count_after']}, "
        f"ngz_active: {cdx['ngz_active']}, low: {cdx['ngz_low']}, high: {cdx['ngz_high']}"
    )

    # ---------- Build and store summary HUD ----------
    summary = build_summary(
        rules_pass=int(cdx.get("rules_pass", 2)),
        ngz_active=int(cdx.get("ngz_active", 0)),
        ngz_low=cdx.get("ngz_low"),
        ngz_high=cdx.get("ngz_high"),
        total=cdx["pivot_count_total"],
        after=cdx["pivot_count_after"],
        removed_ngz=removed_ngz,
        reuse_removed=reuse_removed,
        role_blocked=role_blocked,
        removed_fc=removed_fc,
        dup_removed=dup_removed,
        mid_live=mid_live,
        audit_list=audit,
    )
    cdx["hud"]["desk2"] = summary

    # ---------------- PANEL JSON SNAPSHOT ----------------
    try:
        panel_dir = r"E:\EURUSD\trap_zone\panel"
        os.makedirs(panel_dir, exist_ok=True)

        pivot_list = [
            {
                "id": p.get("id"),
                "price": float(p.get("price", 0.0)),
                "role": int(p.get("role", 0)),
                "score": int(p.get("score", 0)),
                "removed_by": p.get("removed_by"),
            }
            for p in audit
        ]

        payload = {
            "timestamp": get_server_time(),
            "rules_pass": int(cdx.get("rules_pass", 2)),
            "ngz": {
                "active": int(cdx.get("ngz_active", 0)),
                "low": cdx.get("ngz_low"),
                "high": cdx.get("ngz_high"),
            },
            "counts": {
                "total": cdx["pivot_count_total"],
                "after": cdx["pivot_count_after"],
                "removed_ngz": removed_ngz,
                "reuse_removed": reuse_removed,
                "role_blocked": role_blocked,
                "forecast_removed": removed_fc,
                "dedupe_removed": dup_removed,
            },
            "mid_live": mid_live,
            "hud": summary,
            "hud_lines": hud_lines,
            "pivots": pivot_list,
        }

        tmp_path = os.path.join(panel_dir, "desk2_pivots.json.tmp")
        final_path = os.path.join(panel_dir, "desk2_pivots.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, final_path)
    except Exception as e:
        hud_lines.append(f"⚠️ DESK 2: panel JSON write failed: {e}")
        cdx["hud"]["desk2"] = summary



# ============================================================
# DESK 3 – PIVOT LOCK (seat/lock using LIVE mid)
# ============================================================

def desk3_pivot_lock(db_: MemoryDB):
    global cdx, pivot_lock_cache

    cdx.setdefault("hud", {})
    hud_lines: List[str] = []

    def log(msg: str):
        hud_lines.append(str(msg))

    def flush_hud():
        cdx["hud"]["desk3"] = "\n".join(hud_lines) if hud_lines else ""

    master_time = get_server_time()
    rules_pass  = int(cdx.get("rules_pass", 2))

    # prefer cp_mid; if missing, fall back to md_cp.mid
    cp_mid = cdx.get("cp_mid")
    md     = cdx.get("md_cp") or {}
    if cp_mid is None:
        try:
            cp_mid = float(md.get("mid") or 0.0) or None
        except Exception:
            cp_mid = None

    bias_in = cdx.get("pivot_bias")  # 1=long, 2=short

    primary = None
    backup  = None
    last_close = None

    # Ensure in-memory cache always has a sentinel row
    if not pivot_lock_cache:
        write_lock_packet(0, 0.0, 3, 0, 0.0, 0.0, 0.0, master_time)

    # ----- lock helpers -----
    def read_lock():
        if not pivot_lock_cache:
            return None
        pid = next(iter(pivot_lock_cache))
        row = pivot_lock_cache.get(pid, {})
        pp  = float(row.get("pivot_price", 0.0))
        top = float(row.get("zone_top", 0.0))
        bot = float(row.get("zone_bottom", 0.0))
        trg = float(row.get("trigger_line", 0.0))
        locked = int(row.get("locked", 0))
        bias   = int(row.get("bias", row.get("role", 0)))

        # sentinel or junk => treat as no lock
        if pid == 0 or (pp == 0.0 and top == 0.0 and bot == 0.0):
            return None

        return {
            "pid":   int(pid),
            "pp":    pp,
            "top":   top,
            "bot":   bot,
            "trg":   trg,
            "locked": locked,
            "bias":   bias,
            "since": row.get("locked_since", "n/a"),
        }

    def write_panel_json(last_close_value=None):
        # make sure HUD text is up to date before writing
        flush_hud()
        try:
            panel_dir = r"E:\EURUSD\trap_zone\panel"
            os.makedirs(panel_dir, exist_ok=True)

            lock_snapshot = read_lock()
            md_local  = cdx.get("md_cp") or {}
            mid_live  = md_local.get("mid")
            md_ts     = md_local.get("timestamp")

            payload = {
                "timestamp":     master_time,
                "rules_pass":    rules_pass,
                "mid_live":      mid_live,
                "mid_timestamp": md_ts,
                "cp_mid":        cp_mid,
                "last_close":    last_close_value,
                "pivot_bias":    cdx.get("pivot_bias"),
                "pivot_locked":  int(cdx.get("pivot_locked", 0)),
                "pivot": {
                    "id":        cdx.get("pivot_selected_id"),
                    "price":     cdx.get("pivot_selected_px"),
                    "zone_top":  cdx.get("pivot_selected_top"),
                    "zone_bottom": cdx.get("pivot_selected_bot"),
                    "trigger":   cdx.get("pivot_selected_trg"),
                },
                "lock_row": lock_snapshot,
                "hud":      cdx["hud"].get("desk3", ""),
            }

            tmp_path   = os.path.join(panel_dir, "desk3_lock.json.tmp")
            final_path = os.path.join(panel_dir, "desk3_lock.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            log(f"⚠️ DESK 3: panel JSON write failed: {e}")

    # ----- initial lock snapshot -----
    lock = read_lock()
    if lock:
        log(
            f"LOCK cache id={lock['pid']}@{fmt5(lock['pp'])} "
            f"Z=[{fmt5(lock['bot'])}…{fmt5(lock['top'])}] "
            f"trg={fmt5(lock['trg'])} bias={lock['bias']} since={lock['since']}"
        )
    else:
        log("LOCK cache: none (sentinel/empty)")

    # ============================================================
    # 1) RULES OFF HANDLING
    # ============================================================
    if rules_pass == 2:
        has_pending = any(
            p for p in db_.pending_orders
            if p.get("case_closed", 0) == 0
            and str(p.get("status", "")).lower() in ("waiting", "pending")
        )

        if not has_pending:
            # full OFF: reset to sentinel, clear CDX
            write_lock_packet(0, 0.0, 3, 0, 0.0, 0.0, 0.0, master_time)
            cdx["pivot_locked"]       = 0
            cdx["pivot_selected_id"]  = None
            cdx["pivot_selected_px"]  = None
            cdx["pivot_selected_top"] = None
            cdx["pivot_selected_bot"] = None
            cdx["pivot_selected_trg"] = None
            cdx["pivot_bias"]         = 3
            log("RULES OFF + no pending → reset to sentinel, lock cleared")
        else:
            # keep existing lock state if present
            if lock and lock["locked"] == 1:
                cdx["pivot_locked"]       = 1
                cdx["pivot_selected_id"]  = lock["pid"]
                cdx["pivot_selected_px"]  = lock["pp"]
                cdx["pivot_selected_top"] = lock["top"]
                cdx["pivot_selected_bot"] = lock["bot"]
                cdx["pivot_selected_trg"] = lock["trg"]
                cdx["pivot_bias"]         = lock["bias"]
                log("RULES OFF + pending + lock → preserve active lock")
            else:
                log("RULES OFF + pending → no new seat, no reset")

        log(
            f"OUT: locked={cdx.get('pivot_locked',0)} "
            f"bias={cdx.get('pivot_bias','n/a')} "
            f"id={cdx.get('pivot_selected_id','n/a')} "
            f"px={fmt5(cdx['pivot_selected_px']) if cdx.get('pivot_selected_px') else 'n/a'}"
        )
        flush_hud()
        write_panel_json(last_close)
        return

    # ============================================================
    # 2) UNLOCK LOGIC (RULES ON)
    # ============================================================

    # Last CLOSED candle from MEMORY (DLL feed), not SQL
    if db_.mem_m1:
        try:
            last_close = float(db_.mem_m1[-1].get("close", 0.0))
        except Exception:
            last_close = None

    if lock and lock["locked"] == 1 and last_close is not None:
        in_trap = lock["bot"] <= last_close <= lock["top"]
        trig_lo, trig_hi = min(lock["trg"], lock["pp"]), max(lock["trg"], lock["pp"])
        in_trig = trig_lo <= last_close <= trig_hi
        in_either = in_trap or in_trig

        log(
            f"UNLOCK check: close={fmt5(last_close)} "
            f"trap=[{fmt5(lock['bot'])}…{fmt5(lock['top'])}] "
            f"trig=[{fmt5(trig_lo)}…{fmt5(trig_hi)}] "
            f"{'INSIDE' if in_either else 'OUTSIDE'}"
        )

        if not in_either:
            # unlock same pivot, keep its prices
            use_bias = lock["bias"] if lock["bias"] in (1, 2) else (bias_in if bias_in in (1, 2) else 0)
            write_lock_packet_safe(
                lock["pid"],
                lock["pp"],
                use_bias,
                0,
                lock["top"],
                lock["bot"],
                lock["trg"],
                master_time,
            )
            cdx["pivot_locked"] = 0
            log("UNLOCK: close outside BOTH trap and trigger → unlocked")
            lock = None
        else:
            # still locked → mirror to CDX and stop, no reseat
            cdx["pivot_locked"]       = 1
            cdx["pivot_selected_id"]  = lock["pid"]
            cdx["pivot_selected_px"]  = lock["pp"]
            cdx["pivot_selected_top"] = lock["top"]
            cdx["pivot_selected_bot"] = lock["bot"]
            cdx["pivot_selected_trg"] = lock["trg"]
            cdx["pivot_bias"]         = lock["bias"]
            log("LOCK retained (inside trap/trigger) → no reseat this cycle")
            log(
                f"OUT: locked=1 bias={cdx.get('pivot_bias')} "
                f"id={cdx.get('pivot_selected_id')} "
                f"px={fmt5(cdx['pivot_selected_px'])}"
            )
            flush_hud()
            write_panel_json(last_close)
            return

    # ============================================================
    # 3) CHOOSE BEST / BACKUP PIVOT (WHEN UNLOCKED)
    # ============================================================

    pool = cdx.get("pivot_pool") or []
    if cp_mid is not None and isinstance(pool, list) and pool:
        def dist(p):
            try:
                return abs(float(p.get("price", 0.0)) - cp_mid)
            except Exception:
                return 1e9

        sorted_pool = sorted(pool, key=dist)

        if sorted_pool:
            p0 = sorted_pool[0]
            primary = {
                "id":    p0.get("id"),
                "price": float(p0.get("price", 0.0)),
            }
        if len(sorted_pool) > 1:
            p1 = sorted_pool[1]
            backup = {
                "id":    p1.get("id"),
                "price": float(p1.get("price", 0.0)),
            }

        if primary:
            log(
                f"BEST pivot: #{primary['id']}@{fmt5(primary['price'])} "
                f"(cp_mid={fmt5(cp_mid)})"
            )
        if backup:
            log(f"BACKUP pivot: #{backup['id']}@{fmt5(backup['price'])}")
    else:
        log("NO candidates: no cp_mid or empty pivot_pool")

    # ============================================================
    # 4) SEATING LOGIC (WHEN UNLOCKED)
    # ============================================================

    def seat(cand, bias):
        if not cand or "id" not in cand or "price" not in cand:
            return False
        if bias not in (1, 2):
            return False

        existing = read_lock()
        # once locked, never change pivot here
        if existing and existing["locked"] == 1:
            log("SEAT: already locked → no reseat")
            return False

        sel_id = int(cand["id"])
        sel_pp = float(cand["price"])

        upper, lower, trg = calc_zone(sel_pp, bias, PIP)
        will_lock = 1 if (cp_mid is not None and lower <= cp_mid <= upper) else 0

        write_lock_packet_safe(sel_id, sel_pp, bias, will_lock, upper, lower, trg, master_time)

        cdx["pivot_locked"]       = will_lock
        cdx["pivot_selected_id"]  = sel_id
        cdx["pivot_selected_px"]  = sel_pp
        cdx["pivot_selected_top"] = upper
        cdx["pivot_selected_bot"] = lower
        cdx["pivot_selected_trg"] = trg
        cdx["pivot_bias"]         = bias

        log(
            f"SEAT pivot #{sel_id}@{fmt5(sel_pp)} "
            f"Z=[{fmt5(lower)}…{fmt5(upper)}] bias={bias} locked={will_lock}"
        )
        return True

    current_lock = read_lock()
    unlocked = (current_lock is None) or (int(current_lock["locked"]) == 0)

    if unlocked and bias_in in (1, 2):
        seated = False
        if primary:
            seated = seat(primary, bias_in)
        if not seated and backup:
            log("SEAT: primary failed → try backup")
            seated = seat(backup, bias_in)
        if not seated:
            log("SEAT: no candidate seated")
    else:
        if not unlocked:
            log("SEAT skipped: lock still active")
        elif bias_in not in (1, 2):
            log("SEAT skipped: invalid bias")

    log(
        f"OUT: locked={cdx.get('pivot_locked',0)} "
        f"bias={cdx.get('pivot_bias','n/a')} "
        f"id={cdx.get('pivot_selected_id','n/a')} "
        f"px={fmt5(cdx['pivot_selected_px']) if cdx.get('pivot_selected_px') else 'n/a'}"
    )

    flush_hud()
    write_panel_json(last_close)


# ============================================================
# DESK 4 – PENDING BUILDER (lean, mem-first)
# ============================================================

def desk4_pending_builder(db_: MemoryDB):
    """
    DESK 4 – PENDING BUILDER
      - Uses live lock (pivot_lock_cache + cdx/md_cp/mem_m1) to decide play.
      - TWO triggers only:
          1) bounce  = live bid/ask straddles pivot (center)
          2) reclaim = M1 close near pivot (±5 pips in trade direction)
      - Writes new pending into MySQL pending_orders.
      - Mirrors to db_.pending_orders (memory mirror).
      - Respects existing in-memory pending to avoid double-build.
    """
    global cdx, pivot_lock_cache

    # ---------- HUD + PANEL STATE ----------
    cdx.setdefault("hud", {})
    hud_lines: List[str] = []

    def set_hud(status: str, line2: str = ""):
        hud_lines.clear()
        hud_lines.append(f"Desk4: {status}")
        if line2:
            hud_lines.append(line2)

    def flush_hud():
        cdx["hud"]["desk4"] = "\n".join(hud_lines) if hud_lines else ""

    panel_dir = r"E:\EURUSD\trap_zone\panel"
    panel_state = {
        "status": "init",        # init/skip/idle/error/built
        "detail": "",
        "new_pending_id": None,
    }

    # lock & market vars (for panel JSON)
    pid = 0
    locked = 0
    bias = 0
    pp = 0.0           # center pivot
    z_top = 0.0
    z_bot = 0.0
    trg = 0.0
    entry_price = 0.0  # actual pending price (edge of zone in bias direction)

    bid = ask = spr = 0.0
    m1c = m1h = m1l = m1o = 0.0

    pivot_straddle = False
    close_reclaim = False
    desired: Optional[str] = None

    def write_panel_json():
        """Best-effort panel snapshot → desk4_pending.json"""
        flush_hud()
        try:
            os.makedirs(panel_dir, exist_ok=True)
            payload = {
                "timestamp": get_server_time(),
                "rules_pass": cdx.get("rules_pass"),
                "lock": {
                    "id": pid,
                    "pivot_price": entry_price,  # entry, not center
                    "bias": bias,
                    "locked": locked,
                    "zone_top": z_top,
                    "zone_bottom": z_bot,
                    "trigger": trg,
                },
                "signals": {
                    "pivot_straddle": bool(pivot_straddle),
                    "close_reclaim": bool(close_reclaim),
                    "desired": desired,
                },
                "market": {
                    "bid": bid,
                    "ask": ask,
                    "spread": spr,
                    "m1_close": m1c,
                    "m1_high": m1h,
                    "m1_low": m1l,
                    "m1_open": m1o,
                },
                "pending_state": panel_state,
                "hud": cdx["hud"].get("desk4", ""),
            }
            tmp_path   = os.path.join(panel_dir, "desk4_pending.json.tmp")
            final_path = os.path.join(panel_dir, "desk4_pending.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            # HUD only, no prints
            set_hud("error", f"panel JSON failed: {e}")
            flush_hud()

    # ---------- PRECONDITIONS ----------
    if cdx.get("rules_pass") != 1:
        set_hud("OFF", "rules_pass!=1")
        panel_state["status"] = "skip"
        panel_state["detail"] = "rules_pass!=1"
        write_panel_json()
        return

    if not pivot_lock_cache:
        set_hud("no lock", "pivot_lock_cache empty")
        panel_state["status"] = "skip"
        panel_state["detail"] = "no lock cache"
        write_panel_json()
        return

    # single active lock row (enforced upstream by Desk 3)
    pid = next(iter(pivot_lock_cache))
    L = pivot_lock_cache[pid]
    locked = int(L.get("locked", 0))
    bias = int(L.get("bias", L.get("role", 0)))
    pp = float(L.get("pivot_price", 0.0))      # center pivot
    z_top = float(L.get("zone_top", 0.0))
    z_bot = float(L.get("zone_bottom", 0.0))
    trg = float(L.get("trigger_line", 0.0))

    if locked != 1 or bias not in (1, 2) or pp == 0.0:
        set_hud("invalid lock", f"locked={locked} bias={bias} pp={pp}")
        panel_state["status"] = "skip"
        panel_state["detail"] = f"invalid lock (locked={locked}, bias={bias}, pp={pp})"
        write_panel_json()
        return

    # ENTRY PRICE: zone edge in direction of trade
    if bias == 1:   # LONG → lower edge
        entry_price = z_bot if z_bot else pp - 5 * PIP
    else:           # SHORT → upper edge
        entry_price = z_top if z_top else pp + 5 * PIP

    # ---------- MARKET SNAPSHOT FROM MEMORY ----------
    md = cdx.get("md_cp") or {}
    bid = float(md.get("bid") or 0.0)
    ask = float(md.get("ask") or 0.0)
    spr = float(md.get("spread") or (ask - bid) if (bid and ask) else 0.0)

    have_live = bool(bid and ask)
    lo_live = min(bid, ask) if have_live else 0.0
    hi_live = max(bid, ask) if have_live else 0.0

    # last closed M1 from MEMORY (DLL feed), not SQL
    if db_.mem_m1:
        last = db_.mem_m1[-1]
        m1c = float(last.get("close", pp))
        m1h = float(last.get("high", m1c))
        m1l = float(last.get("low", m1c))
        m1o = float(last.get("open", m1c))
    else:
        m1c = m1h = m1l = m1o = pp

    # ---------- SIGNAL LOGIC (2 TRIGGERS ONLY) ----------
    # 1) BOUNCE: live band straddles the center pivot
    pivot_straddle = have_live and (lo_live <= pp <= hi_live)

    # 2) RECLAIM: M1 close near center pivot in direction of trade, up to ±5 pips
    close_reclaim = False
    if bias == 2:  # short
        if m1c <= pp and m1c >= pp - 5 * PIP:
            close_reclaim = True
    elif bias == 1:  # long
        if m1c >= pp and m1c <= pp + 5 * PIP:
            close_reclaim = True

    # per-pivot play ledger: if pivot dead today, block
    if is_pivot_dead_today(db_, pid):
        set_hud("pivot dead", f"id={pid}")
        panel_state["status"] = "skip"
        panel_state["detail"] = f"pivot dead today (id={pid})"
        write_panel_json()
        return

    # decide play: bounce priority > trap
    if pivot_straddle:
        desired = "bounce"
    elif close_reclaim:
        desired = "trap"
    else:
        set_hud("idle", "waiting for bounce/reclaim")
        panel_state["status"] = "idle"
        panel_state["detail"] = "waiting for bounce/reclaim"
        write_panel_json()
        return

    # don't reuse same play for that pivot/day
    pk = plays_key(pid)
    ledger = db_.plays.get(pk, {"bounce": 0, "trap": 0, "dead": 0})
    if desired == "bounce" and ledger["bounce"]:
        set_hud("play spent", "bounce already used")
        panel_state["status"] = "skip"
        panel_state["detail"] = "bounce already used"
        write_panel_json()
        return
    if desired == "trap" and ledger["trap"]:
        set_hud("play spent", "trap already used")
        panel_state["status"] = "skip"
        panel_state["detail"] = "trap already used"
        write_panel_json()
        return

    # ---------- MEMORY PENDING CHECK (ONLY SOURCE) ----------
    if any(
        p for p in db_.pending_orders
        if p.get("case_closed", 0) == 0
        and str(p.get("status", "")).lower() in ("waiting", "pending")
    ):
        set_hud("pending exists", "memory pending active")
        panel_state["status"] = "skip"
        panel_state["detail"] = "memory pending active"
        write_panel_json()
        return

    # ---------- DB CONNECT (FOR INSERT ONLY) ----------
    try:
        conn = pymysql.connect(**DB_CFG)
    except Exception as e:
        set_hud("error", "db connect failed")
        panel_state["status"] = "error"
        panel_state["detail"] = f"DB connect failed: {e}"
        write_panel_json()
        return

    created_at = get_server_time()

    # detect optional play column name (lightweight schema check)
    play_col = None
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM pending_orders")
            cols = {row[0].lower(): row[0] for row in cur.fetchall()}
        for key in ("play_used", "play_type", "play"):
            if key in cols:
                play_col = cols[key]
                break
    except Exception:
        play_col = None  # fine, we'll insert without play column

    # ---------- INSERT PENDING ----------
    try:
        with conn.cursor() as cur:
            if play_col:
                sql = f"""
                    INSERT INTO pending_orders (
                        pivot_id, pivot_price, bias,
                        upper_threshold, lower_threshold, trigger_line,
                        trap_state, pivot_state, status, case_closed,
                        {play_col}, created_at
                    ) VALUES (
                        %s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s
                    )
                """
                cur.execute(
                    sql,
                    (
                        pid, entry_price, bias,
                        z_top, z_bot, trg,
                        1, 1, "waiting", 0,
                        desired, created_at,
                    ),
                )
            else:
                sql = """
                    INSERT INTO pending_orders (
                        pivot_id, pivot_price, bias,
                        upper_threshold, lower_threshold, trigger_line,
                        trap_state, pivot_state, status, case_closed,
                        created_at
                    ) VALUES (
                        %s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,%s,
                        %s
                    )
                """
                cur.execute(
                    sql,
                    (
                        pid, entry_price, bias,
                        z_top, z_bot, trg,
                        1, 1, "waiting", 0,
                        created_at,
                    ),
                )
            conn.commit()
            new_id = cur.lastrowid or 0
    except Exception as e:
        set_hud("error", "INSERT failed")
        panel_state["status"] = "error"
        panel_state["detail"] = f"INSERT failed: {e}"
        write_panel_json()
        conn.close()
        return

    # ---------- MIRROR INTO MEMORY ----------
    pending = {
        "id": int(new_id),
        "pivot_id": pid,
        "pivot_price": entry_price,  # entry, not center
        "bias": bias,
        "upper_threshold": z_top,
        "lower_threshold": z_bot,
        "trigger_line": trg,
        "trap_state": 1,
        "pivot_state": 1,
        "status": "waiting",
        "case_closed": 0,
        "play": desired,
        "created_at": created_at,
    }
    db_.pending_orders.append(pending)

    ppmap_set(db_, new_id, pid, desired)
    _row = mark_play_tried(db_, pid, desired)

    direction = "BUY" if bias == 1 else "SELL"
    msg = f"{direction} {fmt5(entry_price)} · {desired} (id={new_id})"

    set_hud("PENDING", msg)
    panel_state["status"] = "built"
    panel_state["detail"] = msg
    panel_state["new_pending_id"] = int(new_id)

    # ---------- OPTIONAL ALERT ----------
    try:
        with conn.cursor() as cur:
            alert_txt = f"🚨 {direction} Pending @ {fmt5(entry_price)} (id={new_id}) · {desired}"
            cur.execute("SHOW COLUMNS FROM alerts")
            acols = [r[0].lower() for r in cur.fetchall()]
            if "timestamp" in acols:
                cur.execute(
                    "INSERT INTO alerts (alert_text, timestamp) VALUES (%s,%s)",
                    (alert_txt, created_at),
                )
            else:
                cur.execute(
                    "INSERT INTO alerts (alert_text, created_at) VALUES (%s,%s)",
                    (alert_txt, created_at),
                )
            conn.commit()
    except Exception as e:
        # log to HUD, but don't break
        set_hud("PENDING", msg + " · alert_failed")

    conn.close()
    write_panel_json()



def desk5_pending_invalidation(db_: MemoryDB):
    """
    DESK 5 – PENDING INVALIDATION (CLOSE-ONLY on CLOSED BARS)
    - One clock: get_server_time() (MT4/MT5 from shared memory).
    - pending_orders: SQL is source of truth, mirrored into db_.pending_orders.
    - Uses CLOSED candles from memory (cdx['md_m1'] or db_.mem_m1) to decide.
    - Invalidates only when a CLOSED bar prints OUTSIDE BOTH trap zone AND trigger band.
    - On invalidation: updates DB, updates memory, clears pivot lock for that pivot.
    """
    global cdx, pivot_lock_cache

    cdx["pending_checked"] = True
    cdx["pending_valid"] = 0

    # ---------- HUD helper ----------
    cdx.setdefault("hud", {})

    def hud(status: str, line2: str = ""):
        """Lightweight HUD snapshot for Desk 5."""
        try:
            line1 = f"Desk5: {status}"
            if line2:
                cdx["hud"]["desk5"] = line1 + "\n" + line2
            else:
                cdx["hud"]["desk5"] = line1
        except Exception:
            pass  # best-effort only

    PIP = 0.00010  # still here if you want it later

    # ---------- PANEL JSON SETUP ----------
    panel_dir = r"E:\EURUSD\trap_zone\panel"

    # defaults so writer is safe from any branch
    panel_state = {
        "status": "init",          # init/hold/closed/none/error
        "detail": "",
        "pending_id": None,
        "breach_time": None,
        "breach_close": None,
    }

    pending_id = 0
    pivot_id = 0
    pivot = 0.0
    bias = 0
    upper = lower = trigger = 0.0
    created_at = ""
    created_ts = 0

    last_bar_close = None
    breach_bar = None
    trig_lo = trig_hi = 0.0

    def write_panel_json():
        """Best-effort panel snapshot → desk5_invalidation.json"""
        try:
            os.makedirs(panel_dir, exist_ok=True)

            payload = {
                "timestamp": get_server_time(),
                "rules_pass": cdx.get("rules_pass"),
                "pending": {
                    "id": pending_id,
                    "pivot_id": pivot_id,
                    "pivot_price": pivot,
                    "bias": bias,
                    "upper_threshold": upper,
                    "lower_threshold": lower,
                    "trigger_line": trigger,
                    "created_at": created_at,
                },
                "bands": {
                    "trap_low": lower,
                    "trap_high": upper,
                    "trig_low": trig_lo,
                    "trig_high": trig_hi,
                },
                "bars": {
                    "last_bar_close": last_bar_close,
                },
                "panel_state": panel_state,
                "hud": cdx["hud"].get("desk5", ""),
            }

            tmp_path = os.path.join(panel_dir, "desk5_invalidation.json.tmp")
            final_path = os.path.join(panel_dir, "desk5_invalidation.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, final_path)
        except Exception as e:
            print(f"⚠️ DESK5 panel JSON write failed: {e}")

    # ---------- tiny helpers (pymysql, DB_CFG) ----------
    def get_db():
        return pymysql.connect(**DB_CFG)

    def hydrate_pending_from_db(conn):
        """Sync latest active pending from DB into memory if memory is empty/behind."""
        has_mem = any(
            p for p in db_.pending_orders
            if p.get("case_closed", 0) == 0
            and str(p.get("status", "")).lower() in ("waiting", "pending")
        )
        if has_mem:
            return
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, pivot_id, pivot_price, bias,
                           upper_threshold, lower_threshold, trigger_line,
                           trap_state, pivot_state, status, case_closed, created_at
                    FROM pending_orders
                    WHERE case_closed = 0
                      AND status IN ('waiting','pending')
                    ORDER BY id DESC
                    LIMIT 5
                    """
                )
                rows = cur.fetchall()
            for r in rows:
                db_.pending_orders.append(
                    {
                        "id": int(r["id"]),
                        "pivot_id": int(r["pivot_id"]),
                        "pivot_price": float(r["pivot_price"]),
                        "bias": int(r["bias"]),
                        "upper_threshold": float(r["upper_threshold"]),
                        "lower_threshold": float(r["lower_threshold"]),
                        "trigger_line": float(r["trigger_line"]),
                        "trap_state": int(r["trap_state"]),
                        "pivot_state": int(r["pivot_state"]),
                        "status": str(r["status"]),
                        "case_closed": int(r["case_closed"]),
                        "created_at": str(r["created_at"]),
                    }
                )
        except Exception as e:
            print(f"DESK5 hydrate_pending_from_db failed: {e}")

    # ---------- current price snapshot (from memory) ----------
    md = cdx.get("md_cp") or {}
    bid = float(md.get("bid") or 0.0)
    ask = float(md.get("ask") or 0.0)
    mid = float(md.get("mid") or (bid + ask) / 2.0 if (bid and ask) else 0.0)

    # ---------- DB: get active pending ----------
    try:
        conn = get_db()
    except Exception as e:
        print(f"DESK5 DB connect failed → cannot validate pending: {e}")
        hud("error", "db connect failed")
        panel_state["status"] = "error"
        panel_state["detail"] = f"db connect failed: {e}"
        write_panel_json()
        return

    # Ensure memory mirror is aware of DB pendings
    hydrate_pending_from_db(conn)

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT id, pivot_id, pivot_price, bias,
                       upper_threshold, lower_threshold, trigger_line,
                       trap_state, pivot_state, created_at, status, case_closed
                FROM pending_orders
                WHERE case_closed = 0
                  AND status IN ('waiting','pending')
                ORDER BY id DESC
                LIMIT 1
                """
            )
            order = cur.fetchone()
    except Exception as e:
        print(f"DESK5 pending_orders SELECT failed: {e}")
        hud("error", "SELECT failed")
        panel_state["status"] = "error"
        panel_state["detail"] = f"SELECT failed: {e}"
        write_panel_json()
        conn.close()
        return

    if not order:
        hud("no pending")
        panel_state["status"] = "none"
        panel_state["detail"] = "no active pending"
        write_panel_json()
        conn.close()
        return

    pending_id = int(order["id"])
    pivot_id = int(order["pivot_id"])
    pivot = float(order["pivot_price"])
    bias = int(order["bias"])
    upper = float(order["upper_threshold"])
    lower = float(order["lower_threshold"])
    trigger = float(order["trigger_line"])
    created_at = str(order["created_at"])
    created_ts = parse_ts(created_at)

    trig_lo = min(trigger, pivot)
    trig_hi = max(trigger, pivot)

    # ---------- CLOSED BAR LOGIC (memory only) ----------
    # Primary source: Desk 0 slice in cdx["md_m1"]; fallback to raw mem_m1.
    bars = cdx.get("md_m1") or getattr(db_, "mem_m1", []) or []
    if not bars:
        hud("HOLD", "no closed bars yet")
        cdx["pending_valid"] = 1
        panel_state["status"] = "hold"
        panel_state["detail"] = "no closed bars yet"
        panel_state["pending_id"] = pending_id
        write_panel_json()
        conn.close()
        return

    # Only bars strictly AFTER created_at (same MT4/MT5 clock)
    norm_bars = []
    for b in bars:
        t_str = str(b.get("time") or b.get("timestamp") or "")
        t = parse_ts(t_str)
        if not t or t <= created_ts:
            continue
        norm_bars.append((t, b))

    norm_bars.sort(key=lambda x: x[0])

    if not norm_bars:
        hud("HOLD", "waiting for first bar after create")
        cdx["pending_valid"] = 1
        panel_state["status"] = "hold"
        panel_state["detail"] = "waiting for first bar after create"
        panel_state["pending_id"] = pending_id
        write_panel_json()
        conn.close()
        return

    # ---------- Breach check ----------
    breach_bar = None
    for _, b in norm_bars:
        c = float(b.get("close") or 0.0)
        in_trap = (lower <= c <= upper)
        in_trig = (trig_lo <= c <= trig_hi)

        # Invalidate only when CLOSED bar is OUTSIDE BOTH trap and trigger band
        if not (in_trap or in_trig):
            breach_bar = b
            break

    if breach_bar:
        reason = "close_outside_trap_and_trigger"
        bt = str(breach_bar.get("time") or breach_bar.get("timestamp") or "n/a")
        bc = float(breach_bar.get("close") or 0.0)

        # panel fields for breach
        last_bar_close = bc
        panel_state["status"] = "closed"
        panel_state["detail"] = reason
        panel_state["pending_id"] = pending_id
        panel_state["breach_time"] = bt
        panel_state["breach_close"] = bc

        # ---------- Update DB ----------
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE pending_orders
                    SET status='closed',
                        trap_state=2,
                        pivot_state=2,
                        case_closed=1,
                        reason=%s
                    WHERE id=%s
                      AND case_closed=0
                      AND status IN ('waiting','pending')
                    """,
                    (reason, pending_id),
                )
                conn.commit()
        except Exception as e:
            print(f"DESK5 UPDATE pending_orders failed: {e}")
            hud("error", "UPDATE failed")
            panel_state["status"] = "error"
            panel_state["detail"] = f"UPDATE failed: {e}"
            write_panel_json()
            conn.close()
            return

        # ---------- Update memory ----------
        for p in db_.pending_orders:
            if int(p.get("id", -1)) == pending_id:
                p.update(
                    {
                        "status": "closed",
                        "trap_state": 2,
                        "pivot_state": 2,
                        "case_closed": 1,
                        "reason": reason,
                    }
                )
                break

        # Clear pivot lock for this pivot (memory only)
        if pivot_id in pivot_lock_cache:
            del pivot_lock_cache[pivot_id]
        else:
            # If single-lock mode with mismatched key, nuke if that row's pivot_id matches.
            if pivot_lock_cache:
                k = next(iter(pivot_lock_cache))
                row = pivot_lock_cache[k]
                if int(row.get("pivot_id", k)) == pivot_id:
                    pivot_lock_cache.clear()

        cdx["pending_valid"] = 0
        hud(
            "CLOSED",
            f"id={pending_id} close={fmt5(bc)}\nreason=outside trap+trigger"
        )
    else:
        # No breach → still valid
        last_bar = norm_bars[-1][1]
        lb_c = float(last_bar.get("close") or 0.0)
        last_bar_close = lb_c
        cdx["pending_valid"] = 1
        hud("HOLD", f"id={pending_id} close={fmt5(lb_c)} in zone")
        panel_state["status"] = "hold"
        panel_state["detail"] = "in zone"
        panel_state["pending_id"] = pending_id

    conn.close()
    write_panel_json()


# ============================================================
# DESK 7 – ORDER EXECUTOR (SIMPLE + SELF-CONTAINED)
# ============================================================

def desk7_order_executor(db_: MemoryDB):
    """
    DESK 7 – ORDER EXECUTOR (lean: trust brain + single dedupe in TX)

    Contract (current 6C file):
      desk6c_decision.json:
        {
          "stage": "6C",
          "time": "...",
          "corridor": {...},
          "fusion_result": {...},
          "decision": {
              "fire": 0/1,
              "pend_id": <id>,
              "reason": "...",
              ...
          }
        }

    Fallback: if old schema exists with root keys "trade" / "pending_id",
    we still support that.

    Writes E:\\EURUSD\\trap_zone\\panel\\desk7_executor.json
    with status snapshot for the diagnostic panel.
    """
    import os, json
    from pathlib import Path
    from datetime import datetime
    import pymysql

    global cdx

    # ---------- tiny helpers ----------
    def fmt5(x):
        try:
            return f"{float(x):.5f}"
        except Exception:
            return "0.00000"

    cdx.setdefault("hud", {})
    cdx["order_fired"] = 0
    cdx["order_reason"] = None

    def hud(status: str, line2: str = ""):
        try:
            line1 = f"Desk7: {status}"
            cdx["hud"]["desk7"] = line1 + ("\n" + line2 if line2 else "")
        except Exception:
            pass

    PANEL_DIR     = Path(r"E:\EURUSD\trap_zone\panel")
    DECISION_PATH = PANEL_DIR / "desk6c_decision.json"
    EXEC_PATH     = PANEL_DIR / "desk7_executor.json"

    def write_panel(data: dict):
        """Write a small JSON snapshot for the panel (best-effort)."""
        try:
            data = dict(data)  # copy
            data.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            tmp = EXEC_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, EXEC_PATH)
        except Exception:
            # panel is best-effort; never kill the executor
            pass

    # ---------- 1) read brain decision ----------
    try:
        if not DECISION_PATH.exists():
            hud("WAIT", "no decision file")
            cdx["order_reason"] = "no_decision_file"
            write_panel(
                {
                    "status": "wait",
                    "reason": "no_decision_file",
                    "trade_flag": 0,
                    "pending_id": 0,
                }
            )
            return

        with open(DECISION_PATH, "r", encoding="utf-8") as f:
            J = json.load(f)
    except Exception as e:
        hud("ERROR", "bad decision.json")
        cdx["order_reason"] = f"decision_read_error: {e}"
        write_panel(
            {
                "status": "error",
                "reason": "decision_read_error",
                "detail": str(e)[:200],
                "trade_flag": 0,
                "pending_id": 0,
            }
        )
        return

    # Support both old ("trade"/"pending_id") and new ("decision.fire"/"decision.pend_id") schemas
    try:
        dec = J.get("decision", {}) or {}
        trade_flag = int(dec.get("fire", J.get("trade", 0)) or 0)
        pending_id = int(dec.get("pend_id", J.get("pending_id", 0)) or 0)
        dec_reason = dec.get("reason") or J.get("reason") or ""
    except Exception:
        trade_flag = 0
        pending_id = 0
        dec_reason = ""

    if trade_flag != 1 or pending_id <= 0:
        hud("WAIT", f"trade={trade_flag} pid={pending_id}")
        cdx["order_reason"] = "trade_flag_off_or_bad_pid"
        write_panel(
            {
                "status": "wait",
                "reason": "trade_flag_off_or_bad_pid",
                "trade_flag": trade_flag,
                "pending_id": pending_id,
                "decision_reason": dec_reason,
            }
        )
        return

    # ---------- 2) DB connect ----------
    try:
        cfg = DB_CFG.copy()
        cfg["autocommit"] = False
        cfg["cursorclass"] = pymysql.cursors.DictCursor
        conn = pymysql.connect(**cfg)
    except Exception as e:
        hud("DB FAIL", str(e)[:40])
        cdx["order_reason"] = "db_connect_failed"
        write_panel(
            {
                "status": "error",
                "reason": "db_connect_failed",
                "trade_flag": trade_flag,
                "pending_id": pending_id,
                "db_error": str(e)[:200],
            }
        )
        return

    try:
        cur = conn.cursor()

        # ---------- 3) TX: lock pending + dedupe + fire ----------
        try:
            conn.begin()

            # 3a) lock that pending row and ensure it's still active
            cur.execute(
                """
                SELECT id, pivot_id, pivot_price, bias, case_closed, status
                FROM pending_orders
                WHERE id=%s
                FOR UPDATE
                """,
                (pending_id,),
            )
            p = cur.fetchone()

            if (
                not p
                or int(p.get("case_closed", 1)) != 0
                or p.get("status") not in ("waiting", "pending")
            ):
                raise RuntimeError("pending_not_active")

            pivot_id = int(p.get("pivot_id") or 0)
            pivot    = float(p.get("pivot_price") or 0.0)
            bias     = int(p.get("bias") or 0)

            if bias not in (1, 2):
                raise RuntimeError(f"invalid_bias:{bias}")

            # 3b) single dedupe check inside TX
            cur.execute(
                "SELECT COUNT(*) AS c FROM orders WHERE pending_id=%s FOR UPDATE",
                (pending_id,),
            )
            row2 = cur.fetchone() or {}
            if int(row2.get("c", 0)) > 0:
                # Order already exists for this pending -> skip quietly
                conn.rollback()
                hud("SKIP", "order_exists_for_pending")
                cdx["order_reason"] = "order_exists_for_pending"
                write_panel(
                    {
                        "status": "skip",
                        "reason": "order_exists_for_pending",
                        "trade_flag": trade_flag,
                        "pending_id": pending_id,
                        "pivot_id": pivot_id,
                        "pivot_price": pivot,
                        "bias": bias,
                    }
                )
                conn.close()
                return

            # 3c) compute order params
            direction = "buy" if bias == 1 else "sell"
            entry     = pivot + 0.00030 if bias == 1 else pivot - 0.00030

            # lot size from ea_settings (optional)
            lot = 0.01
            try:
                cur.execute(
                    "SELECT setting_value FROM ea_settings WHERE setting_key='Lots' LIMIT 1"
                )
                r = cur.fetchone()
                if r and r.get("setting_value") not in (None, ""):
                    lot = float(r["setting_value"])
            except Exception:
                pass

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 3d) insert order
            cur.execute(
                """
                INSERT INTO orders (
                    symbol, entry_price, lot_size,
                    pivot_id, pivot_price,
                    direction, is_opened, is_closed,
                    time, magic_number, pending_id
                ) VALUES (
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s,%s
                )
                """,
                (
                    "EURUSD",
                    entry,
                    lot,
                    pivot_id,
                    pivot,
                    direction,
                    0,
                    0,
                    now_str,
                    1,
                    pending_id,
                ),
            )
            order_id = cur.lastrowid

            # 3e) close pending
            cur.execute(
                """
                UPDATE pending_orders
                SET status='triggered',
                    trap_state=2,
                    pivot_state=2,
                    case_closed=1
                WHERE id=%s
                  AND case_closed=0
                  AND status IN ('waiting','pending')
                """,
                (pending_id,),
            )
            if cur.rowcount != 1:
                raise RuntimeError("pending_update_failed")

            # 3f) optional alert row
            try:
                alert_txt = f"🎯 {direction. upper()} @ {fmt5(entry)} | Pivot={fmt5(pivot)}"
                cur.execute(
                    "INSERT INTO alerts (alert_text, timestamp) VALUES (%s, NOW())",
                    (alert_txt,),
                )
            except Exception:
                # alert is nice to have; don't break tx
                pass

            conn.commit()

        except Exception as txe:
            try:
                conn.rollback()
            except Exception:
                pass
            hud("ERROR", "tx_failed")
            cdx["order_reason"] = f"tx_failed: {txe}"
            write_panel(
                {
                    "status": "error",
                    "reason": "tx_failed",
                    "trade_flag": trade_flag,
                    "pending_id": pending_id,
                    "detail": str(txe)[:200],
                }
            )
            conn.close()
            return

        # ---------- 4) mirror to memory ----------
        try:
            db_.orders.append(
                {
                    "id": order_id,
                    "symbol": "EURUSD",
                    "entry_price": entry,
                    "lot_size": lot,
                    "pivot_id": pivot_id,
                    "pivot_price": pivot,
                    "direction": direction,
                    "is_opened": 0,
                    "is_closed": 0,
                    "time": now_str,
                    "magic_number": 1,
                    "pending_id": pending_id,
                }
            )
        except Exception:
            pass

        try:
            for po in db_.pending_orders:
                if int(po.get("id", -1)) == pending_id:
                    po.update(
                        {
                            "status": "triggered",
                            "trap_state": 2,
                            "pivot_state": 2,
                            "case_closed": 1,
                        }
                    )
                    break
        except Exception:
            pass

        cdx["order_fired"] = 1
        cdx["order_reason"] = "ok"
        hud("FIRED", f"{direction.upper()} @ {fmt5(entry)} id={order_id}")

        write_panel(
            {
                "status": "fired",
                "reason": "ok",
                "trade_flag": trade_flag,
                "pending_id": pending_id,
                "order_id": order_id,
                "direction": direction,
                "entry_price": entry,
                "entry_price_fmt": fmt5(entry),
                "pivot_id": pivot_id,
                "pivot_price": pivot,
                "pivot_price_fmt": fmt5(pivot),
                "bias": bias,
                "lot_size": lot,
            }
        )

    finally:
        try:
            conn.close()
        except Exception:
            pass


# ============================================================
# RUN ONE FULL CYCLE (ALL DESKS)
# ============================================================

# somewhere near top of file:
# cdx is your shared state dict
cdx: Dict[str, Any] = {}

def run_cycle(db_: MemoryDB):
    global cdx

    # make sure shared HUD/state bucket exists
    if not isinstance(cdx, dict):
        cdx = {}
    cdx.setdefault("hud", {})

    # ----- DESK 0–5: build / maintain pending -----
    desk0_universal_rules(db_)
    desk1_pivot_filter(db_)          # ❌ no such function
    desk2_direction_selector(db_)    # ❌ no such function
    desk3_pivot_lock(db_)
    desk4_pending_builder(db_)
    desk5_pending_invalidation(db_)

    # ----- DESK 6: brain -----
    decision = run_brain(db_, cdx)
    cdx["last_decision"] = decision

    # ----- DESK 7 -----
    desk7_order_executor(db_)




# ============================================================
# JSON EXPORT – FRONTEND SYNC ENGINE (with drift_fusion)
# ============================================================
def render_hud(snap, db_: MemoryDB):
    # Width per column. Adjust if your terminal is tiny.
    COL_W = 54

    # If you want to peek at the globals Desk 6/7 use:
    global trade_setup_flag, trade_setup_pending

    # ---------- helpers ----------
    def fmt_or_dash(v):
        return fmt5(v) if v not in (None, 0) else "--"

    # ---------- LEFT COLUMN: PRICE + PIVOT ----------
    raw_bid = snap.get("bid")
    raw_ask = snap.get("ask")
    raw_mid = snap.get("mid")
    raw_spr = snap.get("spread")

    bid = fmt_or_dash(raw_bid)
    ask = fmt_or_dash(raw_ask)
    mid = fmt_or_dash(raw_mid)
    spr = fmt_or_dash(raw_spr)

    # treat seq=0 + no prices as "feed idle"
    seq = snap.get("seq", 0) or 0
    feed_idle = (seq == 0 and not raw_bid and not raw_ask)

    # mask goofy epoch time
    raw_time = get_server_time()
    if raw_time.startswith("1970-01-01"):
        disp_time = "offline (no server time)"
    else:
        disp_time = raw_time

    left = []
    left.append("=== MARKET ASSASSIN – LIVE HUD ===")
    left.append(f"FeedMap : {MAP_NAME}")
    left.append(f"Layout  : {snap.get('version','?')}   Seq: {seq}")
    left.append(f"Time    : {disp_time}")
    left.append("")

    if feed_idle:
        left.append("Price   : feed idle / no ticks")
        left.append("         (market closed or EA OFF)")
    else:
        left.append(f"Price   : Bid {bid}  Ask {ask}")
        left.append(f"         Mid {mid}  Spr {spr}")

    # M1 / M5
    if snap.get("m1"):
        m1 = snap["m1"]
        left.append(
            "M1      : "
            f"O {fmt_or_dash(m1.get('o'))} "
            f"H {fmt_or_dash(m1.get('h'))} "
            f"L {fmt_or_dash(m1.get('l'))} "
            f"C {fmt_or_dash(m1.get('c'))}"
        )
        left.append(f"         @ {m1.get('time_str','--')}")
    if snap.get("m5"):
        m5 = snap["m5"]
        left.append(
            "M5      : "
            f"O {fmt_or_dash(m5.get('o'))} "
            f"H {fmt_or_dash(m5.get('h'))} "
            f"L {fmt_or_dash(m5.get('l'))} "
            f"C {fmt_or_dash(m5.get('c'))}"
        )
        left.append(f"         @ {m5.get('time_str','--')}")

    # Lock / zone snapshot
    lock_id = cdx.get("pivot_selected_id")
    lock_px = cdx.get("pivot_selected_px")
    z_top = cdx.get("pivot_selected_top")
    z_bot = cdx.get("pivot_selected_bot")
    locked = cdx.get("pivot_locked", 0)

    left.append("")
    if lock_id and lock_px:
        left.append(
            f"Lock    : {'🔒' if locked else '🟦'} "
            f"#{lock_id} @ {fmt5(lock_px)}"
        )
        if z_bot and z_top:
            left.append(
                f"Zone    : [{fmt5(z_bot)} … {fmt5(z_top)}]"
            )
    else:
        left.append("Lock    : 🟦 none")
    left.append("")

    # ---------- RIGHT COLUMN: DESK / SYSTEM STATUS ----------
    rules       = cdx.get("rules_pass", 2)
    bias        = cdx.get("pivot_bias")
    trade_setup = cdx.get("trade_setup", 0)  # informational
    fired       = cdx.get("order_fired", 0)
    reason_raw  = (cdx.get("order_reason", "") or "").strip()

    # Active pendings from memory
    active_pendings = [
        p for p in db_.pending_orders
        if int(p.get("case_closed", 0)) == 0 and p.get("status") in ("waiting", "pending")
    ]
    has_pending = bool(active_pendings)

    # Take latest pending by id, if any
    current_pending_id = 0
    if active_pendings:
        try:
            current_pending_id = max(
                active_pendings, key=lambda p: int(p.get("id", 0))
            ).get("id", 0)
            current_pending_id = int(current_pending_id or 0)
        except Exception:
            current_pending_id = 0

    # Safely read external Desk 6 flag
    try:
        ts_flag = int(trade_setup_flag)
    except Exception:
        ts_flag = 0
    try:
        ts_pid = int(trade_setup_pending)
    except Exception:
        ts_pid = 0

    # ---- DESK 6 STATUS (based on external flag) ----
    if not has_pending:
        desk6_status = "⚪ no pending"
    else:
        if ts_flag == 1 and ts_pid == current_pending_id:
            desk6_status = "✅ SETUP LOCKED"
        elif ts_flag == 1 and ts_pid and ts_pid != current_pending_id:
            desk6_status = "⚠ flag for old pending"
        else:
            desk6_status = "… waiting"

    # ---- DESK 7 STATUS (aligned with Desk 7 logic) ----
    if fired:
        desk7_status = "🚀 FIRED"
    else:
        if not has_pending:
            desk7_status = "⚪ no pending"
        elif reason_raw in ("order_exists_for_pending", "order_exists_for_pending_tx"):
            desk7_status = "✅ already fired"
        elif reason_raw in ("db_connect_failed", "db_error", "tx_failed"):
            desk7_status = "❌ error"
        elif reason_raw in ("trade_flag_not_confirmed",):
            desk7_status = "⏳ waiting flag"
        else:
            desk7_status = "⏳ idle"

    # Human-readable reason text
    def pretty_reason(reason: str, has_pending_: bool) -> str:
        if not reason or reason == "ok":
            return ""
        if reason == "no_active_pending" and not has_pending_:
            return "waiting for new pending (Desk 4)"
        if reason == "trade_flag_not_confirmed":
            return "pending ready, waiting for Desk 6 flag"
        if reason in ("order_exists_for_pending", "order_exists_for_pending_tx"):
            return "order already exists for this pending"
        if reason == "trap_pivot_inactive":
            return "trap/pivot not active; cannot fire yet"
        if reason in ("db_connect_failed", "db_error"):
            return "database error (see logs)"
        if reason == "tx_failed":
            return "transaction failed (rolled back)"
        if reason == "conditions_not_met":
            return "conditions not met for firing"
        return reason  # fallback raw code if we don't know it

    reason_pretty = pretty_reason(reason_raw, has_pending)

    # active order (if any) for HUD
    last_order = db_.orders[-1] if db_.orders else None
    last_alert = db_.alerts[-1] if db_.alerts else None

    right = []
    right.append("=== SYSTEM STATUS =====================")
    right.append(
        f"DESK0 Rules : {'✅ ON' if rules == 1 else '⛔ OFF'}"
    )
    right.append(
        f"DESK1-3 Pivot: Bias={bias if bias in (1,2) else '--'}  "
        f"Lock={'🔒' if locked else '🟦'}"
    )
    right.append(
        f"DESK4 Pending: "
        f"{'🟢 ACTIVE' if has_pending else '⚪ none'}"
    )
    right.append(
        f"DESK6 Drift  : {desk6_status}"
    )
    right.append(
        f"DESK7 Order  : {desk7_status}"
    )

    if reason_pretty:
        right.append(f"Reason       : {reason_pretty}")

    right.append("")

    if last_order:
        direction = last_order.get("direction", "").upper()
        ep = fmt_or_dash(last_order.get("entry_price", 0.0))
        pid = last_order.get("pivot_id")
        right.append(
            f"Last Order   : {direction} @ {ep} (pivot #{pid})"
        )

    if last_alert:
        txt = last_alert.get("alert_text", "")
        ts = last_alert.get("timestamp", "")
        right.append(f"Last Alert   : {txt}")
        if ts:
            right.append(f"               @ {ts}")

    # ---- PER-DESK HUD EXCERPT (from cdx['hud']) ----
    hud_block = []
    hud_store = cdx.get("hud")
    if isinstance(hud_store, dict):
        desk_labels = {
            "desk1": "D1",
            "desk4": "D4",
            "desk6": "D6",
            "desk7": "D7",
        }
        for key, label in desk_labels.items():
            txt = hud_store.get(key)
            if not txt:
                continue
            lines = str(txt).splitlines()
            # strip blank lines so we don't get naked "D1:" rows
            tail = [ln for ln in lines if ln.strip()][-2:]
            for line in tail:
                hud_block.append(f"{label}: {line}")

    if hud_block:
        right.append("")
        right.append("HUD Detail  :")
        for ln in hud_block[:6]:
            right.append(f"  {ln[:COL_W-3]}")

    # ---------- RENDER TWO COLUMNS ----------
    max_rows = max(len(left), len(right))
    while len(left) < max_rows:
        left.append("")
    while len(right) < max_rows:
        right.append("")

    total_width = COL_W * 2 + 3
    print("═" * total_width)
    for i in range(max_rows):
        l = left[i][:COL_W]
        r = right[i][:COL_W]
        print(f"{l:<{COL_W}} │ {r:<{COL_W}}")
    print("═" * total_width)

def export_all_json(db_: MemoryDB):
    """
    Master JSON exporter:
      - trap_active.json       (latest pending snapshot, for chart overlay)
      - trap_zone.json         (current lock/zone snapshot, for chart overlay)
      - trap_zone_widget.json  (condensed system/trap status for UI cards)

    Uses in-memory cdx + db_ + pivot_lock_cache to produce a coherent frontend state.
    """
    global cdx, pivot_lock_cache

    os.makedirs(TRAP_JSON_DIR, exist_ok=True)
    now = get_server_time()

    # ---------- safe writer ----------
    def write_json(path: str, payload: dict):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e2:
                print(f"💥 export_all_json: failed to write {os.path.basename(path)} → {e2}")

    # =========================================================
    # 1) trap_active.json – current pending order snapshot
    # =========================================================
    pending = next(
        (
            p for p in sorted(db_.pending_orders, key=lambda x: x.get("id", 0), reverse=True)
            if p.get("case_closed", 0) == 0
            and str(p.get("status", "")).lower() in ("waiting", "pending", "triggered")
        ),
        None,
    )

    trap_active = {
        "timestamp": now,
        "pending": 1 if pending else 0,
        "order": None,
    }

    if pending:
        pivot = round(float(pending.get("pivot_price", 0.0)), 5)
        bias = int(pending.get("bias", 0))
        trigger = round(float(pending.get("trigger_line", 0.0)), 5)
        zone_top = round(float(pending.get("upper_threshold", 0.0)), 5)
        zone_bot = round(float(pending.get("lower_threshold", 0.0)), 5)

        trap_active.update(
            {
                "bias": bias,
                "trigger_price": trigger,
                "zone_top": zone_top,
                "zone_bottom": zone_bot,
                "selected": {"pivot_price": pivot, "zone_price": trigger},
                "order": {
                    "id": int(pending.get("id", 0)),
                    "pivot_id": int(pending.get("pivot_id", 0)),
                    "pivot_price": pivot,
                    "trigger": trigger,
                    "bias": bias,
                    "status": str(pending.get("status", "")),
                    "trap_state": int(pending.get("trap_state", 0)),
                    "pivot_state": int(pending.get("pivot_state", 0)),
                    "created_at": str(pending.get("created_at", "")),
                },
            }
        )
    else:
        trap_active.update(
            {
                "trap_state": 0,
                "pivot_state": 0,
                "bias": 0,
                "trigger_price": 0.0,
                "zone_top": 0.0,
                "zone_bottom": 0.0,
                "selected": None,
            }
        )

    write_json(TRAP_ACTIVE_PATH, trap_active)

    # =========================================================
    # 2) trap_zone.json – lock/zone snapshot (for chart lines)
    # =========================================================
    rules_pass = int(cdx.get("rules_pass", 2) or 2)

    trap_zone = {
        "timestamp": now,
        "pending": 0,
        "status": "no trap zone",
    }

    # derive active lock from pivot_lock_cache (ignore sentinel)
    if pivot_lock_cache:
        pid = next(iter(pivot_lock_cache))
        row = pivot_lock_cache.get(pid, {})
        pivot = float(row.get("pivot_price", 0.0) or 0.0)
        z_top = float(row.get("zone_top", 0.0) or 0.0)
        z_bot = float(row.get("zone_bottom", 0.0) or 0.0)

        # treat all-zero/sentinel as no zone
        if not (pid == 0 and pivot == 0.0 and z_top == 0.0 and z_bot == 0.0):
            trig = float(row.get("trigger_line", 0.0) or 0.0)
            bias = int(row.get("bias", row.get("role", 0)) or 0)

            trap_zone = {
                "timestamp": now,
                "pending": 1,
                "pivot_id": int(pid),
                "pivot_price": round(pivot, 5),
                "bias": bias,
                "zone_top": round(z_top, 5),
                "zone_bottom": round(z_bot, 5),
                "trigger_price": round(trig, 5),
                "trigger_line": round(trig, 5),
                "locked": int(row.get("locked", 0)),
                "locked_since": str(row.get("locked_since", "")),
                "status": "active",
            }

    # RULES OFF → neutralise zone for UI overlays
    if rules_pass != 1:
        trap_zone.update(
            {
                "pivot_price": 0.0,
                "zone_top": 0.0,
                "zone_bottom": 0.0,
                "trigger_price": 0.0,
                "trigger_line": 0.0,
                "pending": 0,
                "status": "rules_off",
            }
        )

    write_json(TRAP_ZONE_PATH, trap_zone)

    # =========================================================
    # 3) trap_zone_widget.json – single source for status cards
    # =========================================================

    # prices snapshot from cdx
    md = cdx.get("md_cp") or {}
    prices = {
        "bid": float(md.get("bid") or 0.0),
        "ask": float(md.get("ask") or 0.0),
        "mid": float(md.get("mid") or 0.0),
    }

    # pending summary for widget
    widget_pending = None
    if pending:
        widget_pending = {
            "id": int(pending.get("id", 0)),
            "pivot_id": int(pending.get("pivot_id", 0)),
            "pivot_price": round(float(pending.get("pivot_price", 0.0) or 0.0), 5),
            "trigger": round(float(pending.get("trigger_line", 0.0) or 0.0), 5),
            "bias": int(pending.get("bias", 0) or 0),
            "status": str(pending.get("status", "")),
            "trap_state": int(pending.get("trap_state", 0) or 0),
            "pivot_state": int(pending.get("pivot_state", 0) or 0),
            "created_at": str(pending.get("created_at", "")),
        }

    # lock snapshot for widget (same sentinel guard)
    lock = None
    if pivot_lock_cache:
        pid = next(iter(pivot_lock_cache))
        row = pivot_lock_cache.get(pid, {})
        pivot = float(row.get("pivot_price", 0.0) or 0.0)
        z_top = float(row.get("zone_top", 0.0) or 0.0)
        z_bot = float(row.get("zone_bottom", 0.0) or 0.0)
        trig = float(row.get("trigger_line", 0.0) or 0.0)

        if not (pid == 0 and pivot == 0.0 and z_top == 0.0 and z_bot == 0.0):
            lock = {
                "pivot_id": int(pid),
                "pivot_price": round(pivot, 5),
                "bias": int(row.get("bias", row.get("role", 0)) or 0),
                "zone_top": round(z_top, 5),
                "zone_bottom": round(z_bot, 5),
                "trigger_line": round(trig, 5),
                "locked": int(row.get("locked", 0)),
                "locked_since": str(row.get("locked_since", "")),
            }

    # drift / fusion snapshot (from Desk 6)
    fusion_raw = cdx.get("drift_fusion")
    if not isinstance(fusion_raw, dict):
        fusion_raw = {}

    drift = {
        "active": int(fusion_raw.get("active", 0) or 0),
        "decision": str(fusion_raw.get("reason") or fusion_raw.get("decision") or ""),
        "score_pct": int(fusion_raw.get("score_pct", 0) or 0),
        "corridor_ok": int(fusion_raw.get("corridor_ok", 0) or 0),
        "drift_ok": int(fusion_raw.get("drift_ok", 0) or 0),
    }

    # order / trigger info (from Desk 6 + 7 flags)
    order_fired = int(cdx.get("order_fired", 0) or 0)
    order_reason = str(cdx.get("order_reason") or "")

    # high-level booleans for pipeline card
    rules_on = (rules_pass == 1)
    has_pending = bool(widget_pending)
    rules_safe_lock = (not rules_on and has_pending)
    rules_hard_off = (not rules_on and not has_pending)

    locked_flag = bool(lock and lock.get("locked") == 1)
    zone_on = bool(lock)
    drift_ok = bool(drift["drift_ok"])
    armed = bool(rules_on and locked_flag and has_pending and drift_ok)

    flow_state = {
        "rules_on": int(rules_on),
        "rules_safe_lock": int(rules_safe_lock),
        "rules_off": int(rules_hard_off),
        "zone_on": int(zone_on),
        "locked": int(locked_flag),
        "has_pending": int(has_pending),
        "drift_ok": int(drift_ok),
        "armed": int(armed),
    }

    widget = {
        "timestamp": now,
        "rules": {
            "pass": rules_pass,
            "mode": (
                "ON" if rules_on else
                "SAFE_LOCK" if rules_safe_lock else
                "OFF"
            ),
        },
        "prices": prices,
        "zone": {
            "has_zone": int(zone_on),
            "locked": int(locked_flag),
            "lock": lock,
        },
        "pending": {
            "has_pending": int(has_pending),
            "data": widget_pending,
        },
        "drift": drift,
        "order": {
            "armed": int(armed),
            "last_fired": order_fired,
            "reason": order_reason,
        },
        "flow_state": flow_state,
    }

    write_json(os.path.join(TRAP_JSON_DIR, "trap_zone_widget.json"), widget)




# ============================================================
# MAIN LOOP – STATIC TERMINAL DASHBOARD
# ============================================================
import io
import contextlib

def run_cycle_quiet(db_: MemoryDB):
    global cdx
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        desk0_universal_rules(db_)
        desk1_direction_filter(db_)    # ✅ was desk1_pivot_filter
        desk2_pivot_filter(db_)        # ✅ was desk2_direction_selector
        desk3_pivot_lock(db_)
        desk4_pending_builder(db_)
        desk5_pending_invalidation(db_)
        desk7_order_executor(db_)
        export_all_json(db_)
    cdx["last_cycle_log"] = buf.getvalue()



def main():
    enable_ansi()
    os.makedirs(r"E:\EURUSD\trap_zone\panel", exist_ok=True)

    try:
        load_heat_map_pivots(db)
        load_pending_orders(db)
    except Exception:
        pass

    try:
        tape.start_taper()
    except Exception:
        pass

    try:
        h_map, ptr, S, version = open_mapping()
    except Exception:
        return

    last_seq = None
    clear_screen()

    try:
        while True:
            try:
                snap = parse_snapshot(ptr, S, version)
            except Exception:
                break

            if not snap:
                time.sleep(0.05)
                continue

            # ---- live feed → MemoryDB ----
            update_db_from_snapshot(db, snap)

            # ---- legacy alias: cdx["md_cp"] from mem_tick (no db.md_cp) ----
            tick = (db.mem_tick[0] if getattr(db, "mem_tick", None) else {}) or {}
            if tick:
                cdx["md_cp"] = {
                    "bid": float(tick.get("bid") or 0.0),
                    "ask": float(tick.get("ask") or 0.0),
                    "mid": float(tick.get("mid") or 0.0),
                    "spread": float(tick.get("spread") or 0.0),
                    "timestamp": tick.get("timestamp") or get_server_time(),
                }
            else:
                cdx.pop("md_cp", None)

            if last_seq is not None and snap["seq"] == last_seq:
                time.sleep(0.1)
                continue
            last_seq = snap["seq"]

            try:
                run_brain(db, cdx)
            except Exception as e:
                try:
                    err_dir = r"E:\EURUSD\trap_zone\panel"
                    os.makedirs(err_dir, exist_ok=True)
                    err_path = os.path.join(err_dir, "brain_error.log")
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} :: {repr(e)}\n")
                except Exception:
                    pass

            run_cycle_quiet(db)

            try:
                cursor_home()
                render_hud(snap, db)
                sys.stdout.flush()
            except Exception:
                pass

            time.sleep(0.25)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if ptr:
                UnmapViewOfFile(ptr)
        except Exception:
            pass
        try:
            if h_map:
                CloseHandle(h_map)
        except Exception:
            pass


if __name__ == "__main__":
    main()



