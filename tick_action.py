#!/usr/bin/env python3
# sidecar_hud.py — Live static HUD + Desk6 sidecar (PNG-aware)
# Location: E:\EURUSD\trap_zone\sidecar_hud.py

import os, sys, time, json, zipfile, signal, math
from collections import deque, Counter
from datetime import datetime, timezone
from pathlib import Path

# ───────────────────────── Config ─────────────────────────
VISION_DIR    = Path(r"E:\EURUSD\trap_zone\vision_model").resolve()
STATE_PATH    = VISION_DIR / "sidecar_state.json"   # <= Desk 6 will read this
REFRESH_HZ    = float(os.getenv("SIDECAR_HZ", "2.0"))      # UI refresh (Hz)
WINDOW_N      = int(os.getenv("SIDECAR_WINDOW", "5"))      # regime smoothing window
MAX_LATENCY_S = float(os.getenv("SIDECAR_MAX_LAT", "180")) # data stale guard (kept, not used for repaint)

# Regime thresholds (tunable)
TH_STALL_PCT = float(os.getenv("TH_STALL_PCT", "0.85"))
TH_COIL_COMP = float(os.getenv("TH_COIL_COMP", "12.0"))
TH_BURST_PX  = float(os.getenv("TH_BURST_PX", "8.0"))
TH_KINK_COIL = int(os.getenv("TH_KINK_COIL", "6"))
TH_KINK_BRST = int(os.getenv("TH_KINK_BRST", "12"))

WEEK = {1:"mon", 2:"tue", 3:"wed", 4:"thu", 5:"fri"}
EMO = dict(
    ok="✅", warn="⚠️", bad="❌", info="ℹ️", zip="📦", png="🖼️", clock="⏱️",
    bolt="⚡", shell="🛡️", coil="🌀", pause="⏸️", up="🟢", dn="🔴",
    meh="🟡", pulse="💓", file="🗂️", link="🔗", tape="📼"
)

# Optional PNG analysis (Pillow)
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ─────────────────────── Helpers ─────────────────────────
def stable_fp(ndj, tells, z_ok, png_ok, id_ok, png_conf):
    """Minimal fingerprint to avoid redraw spam (no latency/ETA)."""
    return (
        ndj.get("id") if ndj else None,
        ndj.get("m1_roll_seen") if ndj else None,
        (tells.get("stall_pct") if tells else None,
         tells.get("compress_px") if tells else None,
         tells.get("burst_up_px") if tells else None,
         tells.get("burst_dn_px") if tells else None,
         tells.get("kink") if tells else None),
        bool(z_ok), bool(png_ok), bool(id_ok),
        int(png_conf or 0)
    )

def weekday_label(ts_utc: float) -> str:
    return WEEK.get(datetime.utcfromtimestamp(ts_utc).isoweekday(), "mon")

def ndjson_paths(label: str):
    return (VISION_DIR / f"market_tape_{label}.ndjson",
            VISION_DIR / f"market_tape_{label}.tells.ndjson",
            VISION_DIR / f"market_tape_{label}.zip")

def tail_last_json(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        back = min(8192, size)
        fh.seek(max(0, size - back))
        chunk = fh.read().decode("utf-8", "ignore")
    for ln in reversed([ln for ln in chunk.splitlines() if ln.strip()]):
        try:
            return json.loads(ln)
        except Exception:
            continue
    return None

def zip_has_png(zf: zipfile.ZipFile, fname: str) -> bool:
    try:
        return fname in zf.namelist()
    except Exception:
        return False

def read_png_from_zip(zpath: Path, fname: str):
    if not _HAS_PIL:
        return None
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            with zf.open(fname) as fh:
                img = Image.open(fh).convert("L")  # grayscale
                return img
    except Exception:
        return None

def clear():  # redraw in place, no scroll
    # Windows: use cls (more reliable than raw ANSI in stock cmd/PowerShell)
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


def color(tag, text):
    C = {"ok":"\x1b[92m","warn":"\x1b[93m","bad":"\x1b[91m","info":"\x1b[96m","dim":"\x1b[90m","rst":"\x1b[0m"}
    return f"{C.get(tag,'')}{text}{C['rst']}"

def fmt5(x):
    try:
        return f"{float(x):.5f}"
    except:
        return "0.00000"

def minute_eta():
    now = datetime.utcnow()
    return 60 - now.second

def ts_to_epoch_s(ts_str: str):
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None

# ─────────────── PNG micro-texture probe (fast) ───────────────
def png_probe(img, y_ref: int|None = None):
    """
    Returns lightweight metrics extracted from the rendered PNG:
      - grad_mean / grad_std : global gradient magnitude stats
      - edge_ratio          : % pixels above gradient threshold
      - center_slope        : linear slope of center-band brightness (proxy drift)
      - band_entropy        : Shannon entropy of center band brightness (0..8)
      - png_conf            : 0..100 heuristic confidence of readable texture
    """
    if img is None:  # no PIL or missing file
        return dict(grad_mean=None, grad_std=None, edge_ratio=None,
                    center_slope=None, band_entropy=None, png_conf=0)

    W, H = img.size
    px = img.load()

    # Simple Sobel-like gradient (integer math, no numpy)
    def gmag(x, y):
        # Clamp neighbors
        def p(xx, yy):
            xx = 0 if xx < 0 else (W-1 if xx >= W else xx)
            yy = 0 if yy < 0 else (H-1 if yy >= H else yy)
            return px[xx, yy]
        gx = (p(x+1,y-1) + 2*p(x+1,y) + p(x+1,y+1)) - (p(x-1,y-1) + 2*p(x-1,y) + p(x-1,y+1))
        gy = (p(x-1,y+1) + 2*p(x,y+1) + p(x+1,y+1)) - (p(x-1,y-1) + 2*p(x,y-1) + p(x+1,y-1))
        return abs(gx) + abs(gy)  # L1 norm, fast

    sample_step = 2  # subsample for speed
    mags = []
    for y in range(1, H-1, sample_step):
        for x in range(1, W-1, sample_step):
            mags.append(gmag(x,y))

    if not mags:
        return dict(grad_mean=None, grad_std=None, edge_ratio=None,
                    center_slope=None, band_entropy=None, png_conf=0)

    # Stats
    m = sum(mags)/len(mags)
    var = sum((v-m)*(v-m) for v in mags)/max(1, len(mags)-1)
    s = math.sqrt(var)
    thr = m + s  # "edge" if > mean+std
    edge_ratio = sum(1 for v in mags if v > thr)/len(mags)

    # Center band analysis (±8px around y_ref, or middle of chart)
    if y_ref is None: y_ref = H//2
    y0 = max(0, y_ref-8); y1 = min(H-1, y_ref+8)
    col_means = []
    for x in range(W):
        acc = 0; c = 0
        for y in range(y0, y1+1):
            acc += px[x,y]; c += 1
        col_means.append(acc / max(1,c))
    # linear regression slope (brightness vs x)
    n = len(col_means)
    sx = (n-1)*n/2.0
    sx2 = (n-1)*n*(2*n-1)/6.0
    sy = sum(col_means)
    sxy = sum(i*col_means[i] for i in range(n))
    den = n*sx2 - sx*sx
    center_slope = ((n*sxy - sx*sy)/den) if den else 0.0

    # Shannon entropy of center band brightness (8-bit)
    hist = [0]*256
    for v in col_means:
        hist[int(max(0, min(255, round(v))))] += 1
    total = float(sum(hist)) or 1.0
    band_entropy = -sum((h/total)*math.log2(h/total) for h in hist if h)  # 0..8

    # Heuristic "confidence" (texture present + not washed out)
    png_conf = int(round(100 * max(0.0, min(1.0,
        0.6*min(1.0, s/150.0) + 0.3*min(1.0, edge_ratio/0.25) + 0.1*min(1.0, band_entropy/7.0)
    ))))

    return dict(
        grad_mean=round(m,1),
        grad_std=round(s,1),
        edge_ratio=round(edge_ratio,3),
        center_slope=round(center_slope,5),
        band_entropy=round(band_entropy,3),
        png_conf=png_conf
    )

# ───────────────────── Regime Logic ─────────────────────
def regime_label(t):
    stall = t.get("stall_pct", 0.0) if t else 0.0
    comp  = t.get("compress_px", 0.0) if t else 0.0
    up    = t.get("burst_up_px", 0.0) if t else 0.0
    dn    = t.get("burst_dn_px", 0.0) if t else 0.0
    kink  = t.get("kink", 0) if t else 0
    if max(up, dn) >= TH_BURST_PX and kink >= TH_KINK_BRST and comp <= 30.0:
        return "burst_up" if up >= dn else "burst_dn"
    if stall >= TH_STALL_PCT and comp <= 30.0:
        return "stall"
    if comp <= TH_COIL_COMP and kink <= TH_KINK_COIL:
        return "coil"
    return "normal"

def regime_score_counts(regs):
    if not regs: return ("normal", 0.0)
    c = Counter(regs)
    top, n = c.most_common(1)[0]
    return top, n / max(1, len(regs))

def regime_badge(regime):
    return {
        "burst_up": f"{EMO['bolt']}{EMO['up']} BURST↑",
        "burst_dn": f"{EMO['bolt']}{EMO['dn']} BURST↓",
        "stall":    f"{EMO['pause']} STALL",
        "coil":     f"{EMO['coil']} COIL",
        "normal":   f"{EMO['meh']} NORMAL",
    }.get(regime, f"{EMO['meh']} NORMAL")

def verdict_color(regime):
    return {"burst_up":"ok","burst_dn":"ok","stall":"warn","coil":"info","normal":"dim"}[regime]

# ─────────────────────── Main Loop ──────────────────────
def run():
    stop = False
    def _sig(*_):
        nonlocal stop; stop = True
    for s in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(s, _sig)
        except: pass

    last_id = None
    miss_png_streak = 0
    tick_ok_streak  = 0
    regs_window     = deque(maxlen=WINDOW_N)
    last_write_s    = 0

    # cache last stable fingerprint + panel text
    run._last_fp    = getattr(run, "_last_fp", None)
    run._last_panel = getattr(run, "_last_panel", None)

    while not stop:
        now = time.time()
        label = weekday_label(now)
        ndj_path, tells_path, zip_path = ndjson_paths(label)

        ndj   = tail_last_json(ndj_path)
        tells = tail_last_json(tells_path)

        # ZIP/PNG checks
        png_ok = False
        z_ok = zip_path.exists() and zip_path.stat().st_size > 0
        if z_ok and ndj and ndj.get("png"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    png_ok = zip_has_png(zf, ndj["png"])
            except Exception:
                z_ok = False
        miss_png_streak = 0 if png_ok else (miss_png_streak + 1)

        # Heartbeat (ID advancing)
        id_ok = False
        if ndj and isinstance(ndj.get("id"), int):
            if last_id is None or ndj["id"] > last_id:
                id_ok = True
                tick_ok_streak += 1
            last_id = ndj["id"]

        # Latency (kept for sidecar JSON; not part of repaint fingerprint)
        lat_s = None
        if ndj and ndj.get("timestamp"):
            t = ts_to_epoch_s(ndj["timestamp"])
            if t: lat_s = max(0, int(time.time()) - t)

        # Regime smoothing
        reg_now = regime_label(tells) if tells else "normal"
        regs_window.append(reg_now)
        regime, conf = regime_score_counts(list(regs_window))

        # PNG probe (optional)
        png_metrics = {}
        if png_ok and ndj and ndj.get("png"):
            try:
                img = read_png_from_zip(zip_path, ndj["png"])
                y_ref = ndj.get("y_ref") if ndj else None
                png_metrics = png_probe(img, y_ref=y_ref)
            except Exception:
                png_metrics = {}
        png_conf = png_metrics.get("png_conf", 0)

        # ---- Sidecar state (Desk6 will read this JSON) ----
        sidecar = {
            "ts": int(time.time()),
            "ok": bool(ndj and z_ok and png_ok and id_ok),
            "id": (ndj.get("id") if ndj else None),
            "tod": (ndj.get("tod") if ndj else None),
            "latency_s": lat_s,
            "eta_s": minute_eta(),
            "m1_roll_seen": bool(ndj.get("m1_roll_seen") if ndj else False),
            "png_in_zip": bool(png_ok),
            "zip_ok": bool(z_ok),
            "id_advance_ok": bool(id_ok),
            "verdict": regime,
            "confidence": round(conf, 3),
            "tells": {
                "stall_pct": tells.get("stall_pct") if tells else None,
                "compress_px": tells.get("compress_px") if tells else None,
                "burst_up_px": tells.get("burst_up_px") if tells else None,
                "burst_dn_px": tells.get("burst_dn_px") if tells else None,
                "zigzag": tells.get("zigzag") if tells else None,
                "kink": tells.get("kink") if tells else None,
                "entropy": tells.get("entropy") if tells else None,
            },
            "prices": {
                "open":  float(ndj.get("open"))  if ndj else None,
                "high":  float(ndj.get("high"))  if ndj else None,
                "low":   float(ndj.get("low"))   if ndj else None,
                "close": float(ndj.get("close")) if ndj else None,
                "mid":   float(ndj.get("mid_price")) if ndj else None,
            },
            "png": (ndj.get("png") if ndj else None),
            "png_metrics": png_metrics,
            "png_conf": png_conf,
            "streaks": {"png_miss": miss_png_streak, "tick_ok": tick_ok_streak},
        }

        # Persist sidecar (rate-limited)
        if sidecar != getattr(run, "_last_state", None) or (time.time() - last_write_s) >= 2.0:
            try:
                STATE_PATH.write_text(json.dumps(sidecar, ensure_ascii=False))
            except Exception:
                pass
            run._last_state = sidecar
            last_write_s = time.time()

        # -------- Stable fingerprint (NO latency/ETA) --------
        fp = stable_fp(ndj, tells, z_ok, png_ok, id_ok, png_conf)

        # Only rebuild + repaint the panel when the fingerprint changes
        if fp != run._last_fp:
            lines = []
            lines.append(f"=== {EMO['tape']} MARKET ASSASSIN — TAPE SIDECAR (LIVE) ===")
            lines.append(f"{EMO['file']} Folder: {VISION_DIR}")
            lines.append(f"Day: {label.upper()}   NDJSON:{'OK' if ndj_path.exists() else '—'}  Tells:{'OK' if tells_path.exists() else '—'}  {EMO['zip']}:{'OK' if z_ok else '—'}")
            lines.append("────────────────────────────────────────────────────────")

            if ndj:
                lines.append("SNAPSHOT")
                lines.append(f"ID {ndj.get('id')}   TOD {ndj.get('tod')}   DOW {ndj.get('dow')}")
                lines.append(f"Prices  O {fmt5(ndj.get('open'))}  H {fmt5(ndj.get('high'))}  L {fmt5(ndj.get('low'))}  C {fmt5(ndj.get('close'))}")
                lines.append(f"Mid {fmt5(ndj.get('mid_price'))}   roll_seen {ndj.get('m1_roll_seen')}   px/px {ndj.get('price_per_px')}")
                lines.append(f"{EMO['png']} {ndj.get('png')}  in ZIP: {'YES' if png_ok else 'NO'}")
            else:
                lines.append(color("bad", "No NDJSON yet for today. Waiting…"))

            if tells:
                lines.append("TELLS")
                lines.append(f"stall {tells.get('stall_pct')}  comp_px {tells.get('compress_px')}  up {tells.get('burst_up_px')}  dn {tells.get('burst_dn_px')}")
                lines.append(f"zigzag {tells.get('zigzag')}  kink {tells.get('kink')}  entropy {tells.get('entropy')}")
            else:
                lines.append(color("dim","No tells yet."))

            # PNG metrics summary (if available)
            if png_metrics:
                pm = png_metrics
                lines.append("PNG")
                lines.append(f"grad μ {pm['grad_mean']}  σ {pm['grad_std']}  edges {pm['edge_ratio']}")
                lines.append(f"slope {pm['center_slope']}  H(ent) {pm['band_entropy']}  conf {pm['png_conf']}")

            # Status line (stable bits only)
            bits = []
            bits.append(f"{EMO['pulse']} {'ID▲' if id_ok else 'ID?'}")
            bits.append(f"{EMO['zip']} {'OK' if z_ok else 'ERR'}")
            bits.append(f"{EMO['png']} {'OK' if png_ok else f'MISS({miss_png_streak})'}")
            badge = regime_badge(regime)
            bits.append(color(verdict_color(regime), badge + f"  conf={int(round(conf*100))}%"))
            if (png_conf < 20) and png_ok:
                bits.append(color("warn","PNG weak"))

            lines.append("────────────────────────────────────────────────────────")
            lines.append("STATUS  : " + "  ".join(bits))
            lines.append(color("dim", f"(Static HUD. Ctrl+C to exit. Redraw-on-change. Sidecar → {STATE_PATH.name})"))

            panel = "\n".join(lines)
            clear()
            sys.stdout.write(panel + "\n")
            sys.stdout.flush()

            run._last_fp = fp
            run._last_panel = panel

        time.sleep(max(0.05, 1.0/REFRESH_HZ))


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
