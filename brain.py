#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List

# ============================================================
# TRACE STUB
# ============================================================

def trace(msg: str):
    return

# ============================================================
# CONFIG
# ============================================================

PANEL_DIR = Path(r"E:\EURUSD\trap_zone\panel")
try:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

FORECAST_PATH = r"E:/eurusd/candles/vision_forecast.json"
PIP = 0.00010


def get_server_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def panel_write(name: str, payload: dict):
    try:
        path = PANEL_DIR / name
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        pass

# ============================================================
# DESK 6A – PENDING + CORRIDOR
# ============================================================

def desk6a_pending_corridor(db_, cdx: Dict[str, Any]) -> Dict[str, Any]:
    cdx.setdefault("hud", {})

    def hud(msg: str):
        try:
            cdx["hud"]["desk6"] = f"Desk6A: {msg}"
        except Exception:
            pass

    res: Dict[str, Any] = {
        "ok": False,
        "reason": "init",
        "pend_id": 0,
        "bias": 0,
        "pivot": 0.0,
        "trigger": 0.0,
        "band_lo": 0.0,
        "band_hi": 0.0,
        "mid": 0.0,
        "trap_ok": 0,
        "pivot_ok": 0,
        "manual_send_it": 0,
    }

    # ---- latest active pending ----
    try:
        pending = next(
            (
                p for p in sorted(db_.pending_orders, key=lambda x: x.get("id", 0), reverse=True)
                if int(p.get("case_closed", 0)) == 0
                and str(p.get("status", "")).lower() in ("waiting", "pending")
            ),
            None,
        )
    except Exception:
        pending = None

    if not pending:
        res["reason"] = "no_pending"
        panel_write(
            "desk6a_corridor.json",
            {
                "stage": "6A",
                "time": get_server_time(),
                "result": res,
            },
        )
        hud("no pending")
        return res

    pend_id   = int(pending.get("id", 0) or 0)
    pivot     = float(pending.get("pivot_price", 0.0) or 0.0)
    trigger   = float(pending.get("trigger_line", 0.0) or 0.0)
    bias      = int(pending.get("bias", 0) or 0)
    trap_ok   = 1 if int(pending.get("trap_state", 0) or 0) == 1 else 0
    pivot_ok  = 1 if int(pending.get("pivot_state", 0) or 0) == 1 else 0
    manual_it = int(pending.get("manual_send_it", 0) or 0)

    # ---- live prices from shared-memory feed ----
    tick = (db_.mem_tick[0] if getattr(db_, "mem_tick", None) else {}) or {}
    bid = float(tick.get("bid") or 0.0)
    ask = float(tick.get("ask") or 0.0)

    if tick.get("mid") is not None:
        mid = float(tick.get("mid") or 0.0)
    else:
        mid = float((bid + ask) / 2.0) if (bid and ask) else 0.0

    band_lo = min(pivot, trigger)
    band_hi = max(pivot, trigger)
    in_band = (band_lo <= mid <= band_hi) if mid != 0.0 else False

    res.update(
        {
            "pend_id": pend_id,
            "bias": bias,
            "pivot": pivot,
            "trigger": trigger,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "mid": mid,
            "trap_ok": trap_ok,
            "pivot_ok": pivot_ok,
            "manual_send_it": manual_it,
        }
    )

    if bias not in (1, 2):
        res["reason"] = "bad_bias"
    elif not (trap_ok and pivot_ok):
        res["reason"] = "trap_or_pivot_off"
    elif mid == 0.0:
        res["reason"] = "mid_zero"
    elif not in_band:
        res["reason"] = "mid_outside_band"
    else:
        res["ok"] = True
        res["reason"] = "ok"

    panel_write(
        "desk6a_corridor.json",
        {
            "stage": "6A",
            "time": get_server_time(),
            "result": res,
            "pending_min": {
                "id": pend_id,
                "status": pending.get("status"),
                "case_closed": pending.get("case_closed"),
                "pivot_id": pending.get("pivot_id"),
            },
            "md_min": tick,
        },
    )
    hud(f"pend={pend_id} mid={mid:.5f} {res['reason']}")
    return res

# ============================================================
# DESK 6B – FORECAST LOAD + DRIFT FUSION
# ============================================================

def _v2_load(path: str = FORECAST_PATH):
    if not os.path.isfile(path):
        return None, "json_not_found"
    try:
        with open(path, "r", encoding="utf-8") as f:
            J = json.load(f)
    except Exception:
        return None, "json_decode_error"

    if not isinstance(J, dict):
        return None, "json_bad_root"

    def norm_ts(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        s = s.replace("T", " ")
        if s.endswith("Z"):
            s = s[:-1]
        return s

    candles = (
        J.get("ensemble", {}).get("predicted_candles")
        or J.get("ensemble", {}).get("ensemble_predicted_candles")
        or J.get("forecast_candle", {}).get("predicted_candles")
        or J.get("forecast_spread", {}).get("predicted_candles")
        or []
    )

    trade_summary = J.get("trade_summary", {}) or {}
    win_metrics   = J.get("window_metrics", {}) or {}
    tail_info     = win_metrics.get("tail", {}) or {}

    edge_score  = float(win_metrics.get("edge_score", trade_summary.get("edge_score", 0.0)) or 0.0)
    slope       = float(win_metrics.get("slope", 0.0) or 0.0)
    hits_beyond = int(trade_summary.get("hits_beyond_trigger", 0) or 0)
    v2_bias_raw = trade_summary.get("bias")
    bias_conf   = trade_summary.get("bias_confidence")
    net_proj    = trade_summary.get("net_projection_pips")
    exp_prof    = trade_summary.get("expected_profit_pips")
    exp_dd      = trade_summary.get("expected_drawdown_pips")

    sig           = J.get("signal_metrics", {}) or {}
    sig_conf      = sig.get("confidence")
    sig_slope_pip = sig.get("slope_pips")

    f_time_str = (
        trade_summary.get("forecast_time")
        or J.get("forecast_time")
        or J.get("anchor_time")
        or (candles[0].get("timestamp") if candles and isinstance(candles[0], dict) else "")
        or ""
    )
    f_time_str = norm_ts(str(f_time_str))

    for c in candles:
        if not isinstance(c, dict):
            continue
        if "timestamp" in c:
            c["timestamp"] = norm_ts(str(c["timestamp"]))
        for k in ("open", "high", "low", "close", "mid_price", "body_size"):
            if k in c:
                try:
                    c[k] = float(c[k])
                except Exception:
                    pass
        if "direction" in c:
            try:
                c["direction"] = int(c["direction"])
            except Exception:
                pass

    v2_bias = None
    if v2_bias_raw is not None:
        if isinstance(v2_bias_raw, (int, float)):
            vb = int(v2_bias_raw)
            if vb in (1, 2):
                v2_bias = vb
        else:
            s = str(v2_bias_raw).lower()
            if s in ("long", "buy", "bull"):
                v2_bias = 1
            elif s in ("short", "sell", "bear"):
                v2_bias = 2

    tail_dir = 0
    if isinstance(tail_info, dict) and tail_info.get("dir"):
        d = str(tail_info["dir"]).lower()
        if d in ("bull", "up"):
            tail_dir = +1
        elif d in ("bear", "down"):
            tail_dir = -1

    return {
        "raw": J,
        "candles": candles,
        "time_str": f_time_str,
        "v2_bias": v2_bias,
        "edge_score": edge_score,
        "slope": slope,
        "tail_dir": tail_dir,
        "hits_beyond_trg": hits_beyond,
        "bias_confidence": float(bias_conf) if bias_conf is not None else None,
        "net_projection": float(net_proj) if net_proj is not None else None,
        "exp_profit": float(exp_prof) if exp_prof is not None else None,
        "exp_drawdown": float(exp_dd) if exp_dd is not None else None,
        "signal_conf": float(sig_conf) if sig_conf is not None else None,
        "signal_slope_pips": float(sig_slope_pip) if sig_slope_pip is not None else None,
    }, "ok"


def _v2_fusion(
    bias: int,
    pivot: float,
    trigger: float,
    candles: List[Dict[str, Any]],
    V2: Dict[str, Any],
) -> Dict[str, Any]:
    N = min(5, max(0, len(candles)))

    wins = 0
    for i in range(max(0, N - 1)):
        c0 = candles[i]
        c1 = candles[i + 1]
        m0 = c0.get("mid_price") or (
            (c0.get("high") + c0.get("low")) / 2 if c0.get("high") and c0.get("low") else c0.get("close", 0.0)
        )
        # FIXED: use same mid formula as m0 (correct parentheses)
        m1 = c1.get("mid_price") or (
            (c1.get("high") + c1.get("low")) / 2 if c1.get("high") and c1.get("low") else c1.get("close", 0.0)
        )
        ok = (m1 > m0) if bias == 1 else (m1 < m0)
        if ok:
            wins += 1

    trig = trigger
    beyond_cnt = 0
    for i in range(N):
        c = candles[i]
        hi = float(c.get("high") or c.get("mid_price") or c.get("close", pivot))
        lo = float(c.get("low") or c.get("mid_price") or c.get("close", pivot))
        if bias == 1:
            if hi >= trig + 0.5 * PIP:
                beyond_cnt += 1
        else:
            if lo <= trig - 0.5 * PIP:
                beyond_cnt += 1

    edge      = float(V2.get("edge_score") or 0.0)
    slope     = float(V2.get("slope") or 0.0)
    tail_dir  = int(V2.get("tail_dir") or 0)
    hits_v2   = int(V2.get("hits_beyond_trg") or 0)
    bias_v2   = int(V2.get("v2_bias") or 0)
    b_conf    = float(V2.get("bias_confidence") or 0.0)
    sig_conf  = float(V2.get("signal_conf") or 0.0)
    net_proj  = float(V2.get("net_projection") or 0.0)
    exp_p     = float(V2.get("exp_profit") or 0.0)
    exp_dd    = float(V2.get("exp_drawdown") or 0.0)

    wins_ok   = wins >= 3
    beyond_ok = (beyond_cnt >= 1) or (hits_v2 >= 1)
    edge_ok   = edge >= 55.0
    slope_ok  = (slope > 0) if bias == 1 else (slope < 0)
    tail_ok   = True if tail_dir == 0 else (bias == 1 and tail_dir > 0) or (bias == 2 and tail_dir < 0)
    bias_ok_v2 = True if bias_v2 == 0 else (bias_v2 == bias)

    econ_ok = True
    econ_why: List[str] = []
    if any(abs(x) > 0 for x in (net_proj, exp_p, exp_dd)):
        if net_proj < 2.0:
            econ_ok = False
            econ_why.append("netProj<2p")
        if exp_p < 3.0:
            econ_ok = False
            econ_why.append("expProf<3p")
        if exp_dd > 3.0:
            econ_ok = False
            econ_why.append("expDD>3p")
    if 0 < b_conf < 0.55:
        econ_ok = False
        econ_why.append("biasConf<0.55")
    if 0 < sig_conf < 60:
        econ_ok = False
        econ_why.append("sigConf<60")

    score_bits = (
        (1 if wins_ok else 0)
        + (1 if beyond_ok else 0)
        + (1 if edge_ok else 0)
        + (1 if slope_ok else 0)
        + (1 if tail_ok else 0)
        + (1 if bias_ok_v2 else 0)
        + (1 if econ_ok else 0)
    )
    need_bits = 4 if any(abs(x) > 0 for x in (net_proj, exp_p, exp_dd, b_conf, sig_conf)) else 3
    drift_ok = score_bits >= need_bits

    score_ratio = (float(score_bits) / float(need_bits)) if need_bits > 0 else 0.0
    score_pct = int(round(max(0.0, min(1.0, score_ratio)) * 100))

    return {
        "drift_ok": 1 if drift_ok else 0,
        "wins": wins,
        "beyond": beyond_cnt,
        "edge_ok": 1 if edge_ok else 0,
        "slope_ok": 1 if slope_ok else 0,
        "tail_ok": 1 if tail_ok else 0,
        "bias_ok_v2": 1 if bias_ok_v2 else 0,
        "econ_ok": 1 if econ_ok else 0,
        "score_bits": score_bits,
        "need_bits": need_bits,
        "score_ratio": round(score_ratio, 3),
        "score_pct": score_pct,
        "edge": edge,
        "slope": slope,
        "tail_dir": tail_dir,
        "hits_v2": hits_v2,
        "bias_conf": b_conf,
        "sig_conf": sig_conf,
        "net_projection": net_proj,
        "exp_profit": exp_p,
        "exp_drawdown": exp_dd,
        "econ_why": econ_why,
    }


def desk6b_forecast_fusion(corridor: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "ok": False,
        "reason": "init",
        "pend_id": int(corridor.get("pend_id") or 0),
        "forecast_time": "",
        "fusion": None,
    }

    pend_id = res["pend_id"]
    bias    = int(corridor.get("bias") or 0)
    pivot   = float(corridor.get("pivot") or 0.0)
    trigger = float(corridor.get("trigger") or 0.0)

    if pend_id <= 0:
        res["reason"] = "no_pending_from_6A"
        panel_write(
            "desk6b_forecast.json",
            {
                "stage": "6B",
                "time": get_server_time(),
                "result": res,
            },
        )
        return res

    if bias not in (1, 2):
        res["reason"] = "bad_bias_from_6A"
        panel_write(
            "desk6b_forecast.json",
            {
                "stage": "6B",
                "time": get_server_time(),
                "result": res,
            },
        )
        return res

    F, status = _v2_load(FORECAST_PATH)
    if status != "ok" or not isinstance(F, dict):
        res["reason"] = f"forecast_{status}"
        panel_write(
            "desk6b_forecast.json",
            {
                "stage": "6B",
                "time": get_server_time(),
                "result": res,
            },
        )
        return res

    candles = F.get("candles") or []
    if len(candles) < 5:
        res["reason"] = "too_few_candles"
        panel_write(
            "desk6b_forecast.json",
            {
                "stage": "6B",
                "time": get_server_time(),
                "result": res,
                "candles_len": len(candles),
            },
        )
        return res

    # -------- core forecast fusion --------
    fusion = _v2_fusion(bias, pivot, trigger, candles, F)

    # ====================================================
    # SIDECAR TAPE NUDGE (no knobs, simple bias+regime rule)
    # ====================================================
    sidecar = None
    try:
        from pathlib import Path  # already imported at top, but safe
        sidecar_path = Path(r"E:\EURUSD\trap_zone\vision_model\sidecar_state.json")
        if sidecar_path.is_file():
            with open(sidecar_path, "r", encoding="utf-8") as sf:
                sidecar = json.load(sf)
    except Exception:
        sidecar = None

    tape_ok = False
    tape_regime = None
    tape_conf = None

    if isinstance(sidecar, dict):
        tape_ok     = bool(sidecar.get("ok"))
        tape_regime = sidecar.get("verdict")
        tape_conf   = sidecar.get("confidence")
    # write raw tape bits into fusion for HUD/panel
    fusion["tape_ok"]     = 1 if tape_ok else 0
    fusion["tape_regime"] = tape_regime
    fusion["tape_conf"]   = float(tape_conf) if tape_conf is not None else None

    # borderline = missed drift_ok by exactly 1 bit
    score_bits = fusion.get("score_bits")
    need_bits  = fusion.get("need_bits")
    drift_ok   = int(fusion.get("drift_ok") or 0)

    borderline = (
        drift_ok == 0
        and isinstance(score_bits, (int, float))
        and isinstance(need_bits, (int, float))
        and score_bits + 1 == need_bits
    )

    # tape supports direction iff regime is a burst in our bias direction
    tape_supports = False
    if tape_ok and tape_regime in ("burst_up", "burst_dn"):
        if bias == 1 and tape_regime == "burst_up":
            tape_supports = True
        if bias == 2 and tape_regime == "burst_dn":
            tape_supports = True

    if borderline and tape_supports:
        # Nudge: let tape grant the missing bit
        fusion["drift_ok"] = 1
        fusion["tape_nudge"] = 1
        # optionally bump score_bits so the math is consistent
        try:
            fusion["score_bits"] = score_bits + 1
        except Exception:
            pass
    else:
        fusion["tape_nudge"] = 0

    # -------- pack result --------
    res.update(
        {
            "ok": True,
            "reason": "ok",
            "forecast_time": F.get("time_str") or "",
            "fusion": fusion,
        }
    )

    panel_write(
        "desk6b_forecast.json",
        {
            "stage": "6B",
            "time": get_server_time(),
            "result": res,
        },
    )
    return res


# ============================================================
# DESK 6C – FINAL DECISION
# ============================================================

# ============================================================
# DESK 6C – FINAL DECISION
# ============================================================

def desk6c_decision(
    corridor: Dict[str, Any],
    fusion_res: Dict[str, Any],
    cdx: Dict[str, Any],
) -> Dict[str, Any]:

    cdx.setdefault("hud", {})

    def hud(line1: str, line2: str = ""):
        try:
            if line2:
                cdx["hud"]["desk6"] = f"Desk6C: {line1}\n{line2}"
            else:
                cdx["hud"]["desk6"] = f"Desk6C: {line1}"
        except Exception:
            pass

    pend_id = int(corridor.get("pend_id") or fusion_res.get("pend_id") or 0)
    corridor_ok = 1 if corridor.get("ok") else 0
    manual_it   = int(corridor.get("manual_send_it") or 0)
    trap_ok     = int(corridor.get("trap_ok") or 0)
    pivot_ok    = int(corridor.get("pivot_ok") or 0)

    fusion   = fusion_res.get("fusion") or {}
    drift_ok = 1 if (fusion_res.get("ok") and fusion.get("drift_ok") == 1) else 0

    fire   = 0
    reason = "init"

    # ---------- decision logic ----------
    if pend_id <= 0:
        fire = 0
        reason = "no_pending"
        hud("BLOCK", "no pending")

    elif manual_it == 1 and trap_ok and pivot_ok:
        fire = 1
        reason = "manual_send_it"
        drift_ok = 1
        corridor_ok = 1
        hud("FIRE", f"manual id={pend_id}")

    else:
        if corridor_ok and drift_ok:
            fire = 1
            reason = "setup_ok"
            hud("FIRE", f"id={pend_id} score={fusion.get('score_pct', 0)}%")
        else:
            fire = 0
            if not corridor_ok and not fusion_res.get("ok"):
                reason = "corridor_and_forecast_fail"
                hud("BLOCK", "bad corridor + forecast")
            elif not corridor_ok:
                reason = "corridor_fail"
                hud("BLOCK", "corridor_fail")
            elif not drift_ok:
                reason = "drift_fail"
                hud("BLOCK", "drift_fail")
            else:
                reason = "blocked"

    # ---------- publish drift_fusion snapshot for HUD / widget ----------
    cdx["drift_fusion"] = {
        "active": 1 if pend_id > 0 else 0,
        "pending_id": pend_id,
        "bias": int(corridor.get("bias") or 0),
        "corridor_ok": int(corridor_ok),
        "drift_ok": int(drift_ok),
        "band_lo": float(corridor.get("band_lo") or 0.0),
        "band_hi": float(corridor.get("band_hi") or 0.0),
        "mid": float(corridor.get("mid") or 0.0),
        "trigger": float(corridor.get("trigger") or 0.0),
        "forecast_time": fusion_res.get("forecast_time") or "",
        "time": get_server_time(),
        "reason": reason,
        "score_pct": int(fusion.get("score_pct", 0) or 0),
        "fusion": fusion,
    }

    decision = {
        "fire": int(fire),
        "pend_id": pend_id,
        "reason": reason,
        "corridor_ok": int(corridor_ok),
        "drift_ok": int(drift_ok),
    }

    panel_write(
        "desk6c_decision.json",
        {
            "stage": "6C",
            "time": get_server_time(),
            "corridor": corridor,
            "fusion_result": fusion_res,
            "decision": decision,
        },
    )
    return decision


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================

def run_brain(db_, cdx: Dict[str, Any]) -> Dict[str, Any]:
    trace("ENTER run_brain()")

    cdx.setdefault("hud", {})

    # ---- reset flags each cycle ----
    cdx["trade_setup"] = 0
    cdx["trade_setup_flag"] = 0
    cdx["trade_setup_pending"] = 0

    trace("CALL desk6a_pending_corridor")
    corr = desk6a_pending_corridor(db_, cdx)
    trace(f"corridor result: pend_id={corr.get('pend_id')} ok={corr.get('ok')}")

    if corr.get("pend_id", 0) <= 0:
        trace("NO pending detected -> skipping fusion")
        fusion_res = {
            "ok": False,
            "reason": "no_pending_from_6A",
            "pend_id": 0,
            "forecast_time": "",
            "fusion": None,
        }
    else:
        trace("CALL desk6b_forecast_fusion")
        fusion_res = desk6b_forecast_fusion(corr)
        trace(f"6B result: ok={fusion_res.get('ok')} reason={fusion_res.get('reason')}")

    trace("CALL desk6c_decision")
    decision = desk6c_decision(corr, fusion_res, cdx)
    trace(f"decision: fire={decision.get('fire')} reason={decision.get('reason')}")

    # ---- expose simple flags in cdx for HUD / widgets / Desk7 ----
    fire    = int(decision.get("fire") or 0)
    pend_id = int(decision.get("pend_id") or 0)

    cdx["trade_setup"] = fire                 # legacy flag
    cdx["trade_setup_flag"] = fire            # explicit "ready to fire" flag
    cdx["trade_setup_pending"] = pend_id      # which pending to fire

    try:
        snapshot = {
            "time": get_server_time(),
            "corridor": corr,
            "forecast": fusion_res,
            "decision": decision,
        }
        trace("WRITE desk6_snapshot.json")
        panel_write("desk6_snapshot.json", snapshot)
    except Exception as e:
        trace(f"snapshot write FAIL: {repr(e)}")

    trace("EXIT run_brain()\n")
    return decision
