from flask import Flask, jsonify, render_template_string, request
import requests
import sqlite3
import threading
import time
import os
from collections import Counter

app = Flask(__name__)

API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
DB_FILE = os.environ.get("DB_FILE", "wingo_data.db")
db_lock = threading.Lock()
collector_started = False

# 1. Database Initialization & Auto-Fix Existing Buggy Records
def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS draws (
                period TEXT PRIMARY KEY,
                number INTEGER,
                size TEXT,
                color TEXT,
                saved_at INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                period TEXT PRIMARY KEY,
                predicted_size TEXT,
                confidence REAL,
                mode TEXT,
                actual_size TEXT,
                actual_number INTEGER,
                status TEXT,
                created_at INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        c.execute("""
            UPDATE predictions
            SET status = CASE
                WHEN UPPER(TRIM(predicted_size)) = UPPER(TRIM(actual_size)) THEN 'WIN'
                ELSE 'LOSS'
            END
            WHERE actual_size IS NOT NULL AND status IN ('WIN', 'LOSS')
        """)
        conn.commit()
        conn.close()

def classify_number(number):
    size = "Big" if number >= 5 else "Small"
    if number in (0, 5):
        color = "Violet"
    elif number in (1, 3, 7, 9):
        color = "Green"
    else:
        color = "Red"
    return size, color

# 2. 4-Layer Advanced AI Prediction Engine
def predict_from_history(history, current_loss_streak=0):
    total = len(history)
    if total < 3:
        return {
            "suggestion": "BIG",
            "confidence": 50.0,
            "big_prob": 50.0,
            "small_prob": 50.0,
            "mode": "Warming Up",
            "details": {"ngram": 50.0, "num_trans": 50.0, "streak": 50.0, "momentum": 50.0}
        }

    latest = history[-1]
    last_num = latest["number"]
    last_size = latest["size"]

    # --- LAYER 1: Number & Pair Transition Matrix ---
    num_big, num_small = 0, 0
    pair_big, pair_small = 0, 0
    prev_num = history[-2]["number"] if total >= 2 else None

    for i in range(total - 1):
        if history[i]["number"] == last_num:
            if history[i + 1]["size"] == "Big":
                num_big += 1
            else:
                num_small += 1
            if i >= 1 and prev_num is not None and history[i - 1]["number"] == prev_num:
                if history[i + 1]["size"] == "Big":
                    pair_big += 1
                else:
                    pair_small += 1

    num_total = num_big + num_small
    num_big_prob = (num_big / num_total * 100.0) if num_total > 0 else 50.0

    pair_total = pair_big + pair_small
    if pair_total >= 2:
        pair_big_prob = (pair_big / pair_total * 100.0)
        layer1_big = (num_big_prob * 0.6) + (pair_big_prob * 0.4)
    else:
        layer1_big = num_big_prob

    # --- LAYER 2: Multi-Depth N-Gram Pattern Matcher (3, 4, 5, 6-Gram) ---
    sizes_seq = [r["size"][0] for r in history]
    ngram_big_score = 0.0
    ngram_weight_sum = 0.0

    for depth, weight in [(3, 1.0), (4, 1.8), (5, 2.8), (6, 4.0)]:
        if total > depth:
            target_pat = sizes_seq[-depth:]
            m_big, m_small = 0, 0
            for i in range(total - depth):
                if sizes_seq[i:i + depth] == target_pat:
                    if sizes_seq[i + depth] == "B":
                        m_big += 1
                    else:
                        m_small += 1
            m_tot = m_big + m_small
            if m_tot > 0:
                ngram_big_score += (m_big / m_tot * 100.0) * weight
                ngram_weight_sum += weight

    layer2_big = (ngram_big_score / ngram_weight_sum) if ngram_weight_sum > 0 else 50.0

    # --- LAYER 3: Streak Exhaustion & Zig-Zag Break Engine ---
    streak_len = 1
    for i in range(total - 2, -1, -1):
        if history[i]["size"] == last_size:
            streak_len += 1
        else:
            break

    s_cont, s_brk = 0, 0
    temp_s = 1
    for i in range(1, total):
        if history[i - 1]["size"] == last_size and temp_s == streak_len:
            if history[i]["size"] == last_size:
                s_cont += 1
            else:
                s_brk += 1
        if history[i]["size"] == history[i - 1]["size"]:
            temp_s += 1
        else:
            temp_s = 1

    s_tot = s_cont + s_brk
    if s_tot > 0:
        cont_prob = (s_cont / s_tot) * 100.0
        layer3_big = cont_prob if last_size == "Big" else (100.0 - cont_prob)
    else:
        layer3_big = 50.0

    # --- LAYER 4: Recent 16-Round Regime & Equilibrium ---
    recent_window = history[-16:]
    r_big_count = sum(1 for r in recent_window if r["size"] == "Big")
    r_big_ratio = (r_big_count / len(recent_window)) * 100.0

    flips = sum(1 for i in range(1, len(recent_window)) if recent_window[i]["size"] != recent_window[i - 1]["size"])
    flip_rate = flips / (len(recent_window) - 1) if len(recent_window) > 1 else 0.5

    if flip_rate >= 0.65:
        layer4_big = 35.0 if last_size == "Big" else 65.0
    elif flip_rate <= 0.35:
        layer4_big = 65.0 if last_size == "Big" else 35.0
    else:
        layer4_big = 100.0 - r_big_ratio

    # --- COMBINE ALL 4 LAYERS ---
    raw_big_prob = (
        (layer1_big * 0.30) +
        (layer2_big * 0.35) +
        (layer3_big * 0.20) +
        (layer4_big * 0.15)
    )

    mode = "Standard Multi-AI"

    # Self-Correction Shield on Consecutive Losses
    if current_loss_streak == 2:
        raw_big_prob = (layer2_big * 0.50) + (layer3_big * 0.35) + (layer4_big * 0.15)
        mode = "Adaptive Shift (L2 Shield)"
    elif current_loss_streak >= 3:
        raw_big_prob = 100.0 - raw_big_prob
        mode = f"Anti-Trap Reverse (L{current_loss_streak} Shield)"

    final_big = round(max(5.0, min(95.0, raw_big_prob)), 1)
    final_small = round(100.0 - final_big, 1)

    if final_big >= final_small:
        suggestion = "BIG"
        confidence = final_big
    else:
        suggestion = "SMALL"
        confidence = final_small

    return {
        "suggestion": suggestion,
        "confidence": confidence,
        "big_prob": final_big,
        "small_prob": final_small,
        "mode": mode,
        "details": {
            "num_trans": round(layer1_big if suggestion == "BIG" else 100 - layer1_big, 1),
            "ngram": round(layer2_big if suggestion == "BIG" else 100 - layer2_big, 1),
            "streak": round(layer3_big if suggestion == "BIG" else 100 - layer3_big, 1),
            "momentum": round(layer4_big if suggestion == "BIG" else 100 - layer4_big, 1)
        }
    }

# 3. Sync API Data & Verify Predictions
def sync_and_evaluate():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"
    }
    try:
        url = f"{API_URL}?ts={int(time.time() * 1000)}"
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        raw_json = response.json()

        items = []
        if isinstance(raw_json, dict):
            if "data" in raw_json and isinstance(raw_json["data"], dict) and "list" in raw_json["data"]:
                items = raw_json["data"]["list"]
            elif "list" in raw_json:
                items = raw_json["list"]
            elif "data" in raw_json and isinstance(raw_json["data"], list):
                items = raw_json["data"]

        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            now = int(time.time())

            for row in items:
                period = str(row.get("issueNumber") or row.get("period", "")).strip()
                if not period:
                    continue
                try:
                    num = int(row.get("number", 0))
                except (ValueError, TypeError):
                    num = 0
                size, color = classify_number(num)
                c.execute("""
                    INSERT OR IGNORE INTO draws (period, number, size, color, saved_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (period, num, size, color, now))

            c.execute("SELECT value FROM settings WHERE key = 'reset_after_period'")
            row_reset = c.fetchone()
            reset_after_period = row_reset[0] if row_reset else "0"

            c.execute("SELECT period, number, size, color FROM draws ORDER BY period ASC")
            all_draws = [{"period": r[0], "number": r[1], "size": r[2], "color": r[3]} for r in c.fetchall()]

            c.execute("SELECT period, predicted_size, confidence, mode, status FROM predictions")
            pred_map = {
                r[0]: {"predicted_size": r[1], "confidence": r[2], "mode": r[3], "status": r[4]}
                for r in c.fetchall()
            }

            running_loss_streak = 0
            for i, d in enumerate(all_draws):
                p = d["period"]
                if i < 4:
                    continue

                if p <= reset_after_period:
                    running_loss_streak = 0

                if p in pred_map:
                    p_info = pred_map[p]
                    is_win = str(p_info["predicted_size"]).strip().upper() == str(d["size"]).strip().upper()
                    correct_status = "WIN" if is_win else "LOSS"

                    if p_info["status"] != correct_status:
                        c.execute("""
                            UPDATE predictions
                            SET actual_size = ?, actual_number = ?, status = ?
                            WHERE period = ?
                        """, (d["size"], d["number"], correct_status, p))
                        p_info["status"] = correct_status
                    status = correct_status
                else:
                    ai_res = predict_from_history(all_draws[:i], running_loss_streak)
                    is_win = str(ai_res["suggestion"]).strip().upper() == str(d["size"]).strip().upper()
                    status = "WIN" if is_win else "LOSS"
                    c.execute("""
                        INSERT OR IGNORE INTO predictions
                        (period, predicted_size, confidence, mode, actual_size, actual_number, status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (p, ai_res["suggestion"], ai_res["confidence"], ai_res["mode"], d["size"], d["number"], status, now))
                    pred_map[p] = {
                        "predicted_size": ai_res["suggestion"],
                        "confidence": ai_res["confidence"],
                        "mode": ai_res["mode"],
                        "status": status
                    }

                if p > reset_after_period:
                    if status == "LOSS":
                        running_loss_streak += 1
                    elif status == "WIN":
                        running_loss_streak = 0

            if len(all_draws) >= 4 and all_draws[-1]["period"].isdigit():
                next_p = str(int(all_draws[-1]["period"]) + 1)
                next_ai = predict_from_history(all_draws, running_loss_streak)
                if next_p not in pred_map:
                    c.execute("""
                        INSERT OR IGNORE INTO predictions
                        (period, predicted_size, confidence, mode, actual_size, actual_number, status, created_at)
                        VALUES (?, ?, ?, ?, NULL, NULL, 'PENDING', ?)
                    """, (next_p, next_ai["suggestion"], next_ai["confidence"], next_ai["mode"], now))
                elif pred_map[next_p]["status"] == "PENDING":
                    c.execute("""
                        UPDATE predictions
                        SET predicted_size = ?, confidence = ?, mode = ?
                        WHERE period = ? AND status = 'PENDING'
                    """, (next_ai["suggestion"], next_ai["confidence"], next_ai["mode"], next_p))

            conn.commit()
            conn.close()
    except Exception as e:
        print(f"[Sync Error]: {e}")

def background_collector():
    while True:
        sync_and_evaluate()
        time.sleep(3)

def ensure_collector_started():
    global collector_started
    if not collector_started:
        collector_started = True
        init_db()
        sync_and_evaluate()
        t = threading.Thread(target=background_collector, daemon=True)
        t.start()

# Start collector automatically when imported by Gunicorn / Cloud WSGI
ensure_collector_started()

# 4. Complete Database & Number Matrix Analytics
def build_dashboard_payload():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'reset_after_period'")
        row_reset = c.fetchone()
        reset_after_period = row_reset[0] if row_reset else "0"

        c.execute("SELECT period, number, size, color FROM draws ORDER BY period ASC")
        draws = [{"period": r[0], "number": r[1], "size": r[2], "color": r[3]} for r in c.fetchall()]

        c.execute("""
            SELECT period, predicted_size, confidence, mode, actual_size, actual_number, status
            FROM predictions ORDER BY period ASC
        """)
        preds = [
            {
                "period": r[0], "predicted_size": r[1], "confidence": r[2],
                "mode": r[3], "actual_size": r[4], "actual_number": r[5], "status": r[6]
            }
            for r in c.fetchall()
        ]
        conn.close()

    total_draws = len(draws)
    if total_draws == 0:
        return {"total_records": 0}

    resolved = [
        p for p in preds
        if p["status"] in ("WIN", "LOSS") and p["period"] > reset_after_period
    ]
    total_preds = len(resolved)
    total_wins = sum(1 for p in resolved if p["status"] == "WIN")
    total_losses = total_preds - total_wins
    win_rate = round((total_wins / total_preds) * 100.0, 1) if total_preds > 0 else 0.0

    max_loss_streak = 0
    max_win_streak = 0
    temp_loss = 0
    temp_win = 0
    loss_level_counts = Counter()

    for p in resolved:
        if p["status"] == "LOSS":
            temp_loss += 1
            if temp_loss > max_loss_streak:
                max_loss_streak = temp_loss
            temp_win = 0
        else:
            if temp_loss > 0:
                loss_level_counts[temp_loss] += 1
            temp_win += 1
            if temp_win > max_win_streak:
                max_win_streak = temp_win
            temp_loss = 0

    current_loss_streak = temp_loss
    current_win_streak = temp_win

    live_ai = predict_from_history(draws, current_loss_streak)
    latest_draw = draws[-1]
    next_period = str(int(latest_draw["period"]) + 1) if latest_draw["period"].isdigit() else "Next"

    cur_size = latest_draw["size"]
    cur_streak = 1
    for i in range(total_draws - 2, -1, -1):
        if draws[i]["size"] == cur_size:
            cur_streak += 1
        else:
            break

    last_8_seq = " - ".join([f"{r['size'][0]}({r['number']})" for r in draws[-8:]])

    number_matrix = []
    for num in range(10):
        appearances = [i for i, r in enumerate(draws) if r["number"] == num]
        hits = len(appearances)
        missing_gap = (total_draws - 1 - appearances[-1]) if hits > 0 else total_draws

        nxt_big = 0
        nxt_small = 0
        pat_breaks = 0
        nxt_nums = []

        for idx in appearances:
            if idx < total_draws - 1:
                nxt_row = draws[idx + 1]
                if nxt_row["size"] == "Big":
                    nxt_big += 1
                else:
                    nxt_small += 1
                if nxt_row["size"] != draws[idx]["size"]:
                    pat_breaks += 1
                nxt_nums.append(nxt_row["number"])

        trans_total = nxt_big + nxt_small
        if trans_total > 0:
            big_pct = round((nxt_big / trans_total) * 100.0, 1)
            small_pct = round((nxt_small / trans_total) * 100.0, 1)
            break_pct = round((pat_breaks / trans_total) * 100.0, 1)
            top_3 = [str(n) for n, _ in Counter(nxt_nums).most_common(3)]
        else:
            big_pct, small_pct, break_pct = 50.0, 50.0, 50.0
            top_3 = ["-"]

        number_matrix.append({
            "number": num,
            "size": "Big" if num >= 5 else "Small",
            "hits": hits,
            "gap": missing_gap,
            "big_pct": big_pct,
            "small_pct": small_pct,
            "break_pct": break_pct,
            "top_next": ", ".join(top_3),
            "is_latest": (num == latest_draw["number"])
        })

    pred_lookup = {p["period"]: p for p in preds}
    recent_rows = []
    for r in reversed(draws[-25:]):
        p_info = pred_lookup.get(r["period"])
        recent_rows.append({
            "period": r["period"],
            "number": r["number"],
            "size": r["size"],
            "color": r["color"],
            "ai_pred": p_info["predicted_size"] if p_info else "-",
            "ai_conf": p_info["confidence"] if p_info else 0,
            "status": p_info["status"] if p_info else "-"
        })

    return {
        "total_records": total_draws,
        "next_period": next_period,
        "latest_period": latest_draw["period"],
        "latest_number": latest_draw["number"],
        "latest_size": cur_size,
        "cur_streak": f"{cur_streak}x {cur_size}",
        "last_8_seq": last_8_seq,
        "live_ai": live_ai,
        "stats": {
            "total_preds": total_preds,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": win_rate,
            "current_loss_streak": current_loss_streak,
            "max_loss_streak": max_loss_streak,
            "current_win_streak": current_win_streak,
            "max_win_streak": max_win_streak
        },
        "number_matrix": number_matrix,
        "recent": recent_rows
    }

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WinGo 30S AI Pro Terminal</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #0b0f19;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #e2e8f0;
    padding-bottom: 30px;
}
header {
    background: linear-gradient(135deg, #111827, #1e293b);
    color: #f59e0b;
    padding: 14px 16px;
    text-align: center;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #334155;
}
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 10px 12px;
    padding: 10px 14px;
    background: #131b2e;
    border-radius: 10px;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #1e293b;
}
.card {
    margin: 10px 12px;
    background: #131b2e;
    border-radius: 12px;
    padding: 14px;
    border: 1px solid #1e293b;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
.card-title {
    font-size: 12px;
    font-weight: 800;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 8px;
}
.ai-hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0f172a;
    padding: 14px;
    border-radius: 10px;
    border-left: 4px solid #f59e0b;
    margin-bottom: 12px;
}
.pred-big { color: #ef4444; font-size: 26px; font-weight: 900; }
.pred-small { color: #3b82f6; font-size: 26px; font-weight: 900; }
.mode-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: bold;
    background: #1e293b;
    color: #f59e0b;
}
.reset-btn {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.2s;
}
.reset-btn:active {
    transform: scale(0.95);
    background: rgba(239, 68, 68, 0.35);
}
.bar-wrap {
    height: 10px;
    background: #1e293b;
    border-radius: 5px;
    overflow: hidden;
    display: flex;
    margin: 6px 0 12px 0;
}
.bar-big { background: linear-gradient(90deg, #dc2626, #ef4444); height: 100%; }
.bar-small { background: linear-gradient(90deg, #2563eb, #3b82f6); height: 100%; }

.grid-4 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}
.metric-box {
    background: #0f172a;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #1e293b;
    text-align: center;
}
.metric-label { font-size: 11px; color: #94a3b8; margin-bottom: 4px; }
.metric-val { font-size: 18px; font-weight: 800; font-family: monospace; }

.sub-engines {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-top: 10px;
}
.eng-item {
    background: #0f172a;
    padding: 6px;
    border-radius: 6px;
    text-align: center;
    font-size: 11px;
}
.eng-item b { display: block; font-size: 12px; color: #f8fafc; margin-top: 2px; }

table {
    width: 100%;
    border-collapse: collapse;
    text-align: center;
}
th {
    background: #0f172a;
    padding: 10px 4px;
    font-size: 11px;
    color: #94a3b8;
    border-bottom: 1px solid #1e293b;
}
td {
    padding: 9px 4px;
    border-bottom: 1px solid #1e293b;
    font-size: 12px;
}
.active-num-row {
    background: rgba(245, 158, 11, 0.14);
    border-left: 3px solid #f59e0b;
}
.badge {
    display: inline-flex;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    color: white;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 12px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.4);
}
.Green { background: linear-gradient(135deg, #10b981, #059669); }
.Red { background: linear-gradient(135deg, #ef4444, #dc2626); }
.Violet { background: linear-gradient(135deg, #8b5cf6, #6d28d9); }

.size-tag { font-weight: bold; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
.size-Big { color: #fca5a5; background: rgba(239, 68, 68, 0.18); }
.size-Small { color: #93c5fd; background: rgba(59, 130, 246, 0.18); }

.win-pill { color: #10b981; background: rgba(16, 185, 129, 0.15); padding: 3px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; }
.loss-pill { color: #ef4444; background: rgba(239, 68, 68, 0.15); padding: 3px 6px; border-radius: 4px; font-weight: bold; font-size: 11px; }
</style>
</head>
<body>

<header>⚡ WINGO 30S — AI PRO ANALYZER</header>

<div class="top-bar">
    <span style="color:#10b981;">● Saved Draws: <span id="total-db">0</span></span>
    <span id="streak-info" style="color:#f59e0b;">-</span>
    <span id="countdown" style="color:#38bdf8; font-family:monospace; font-size:13px;">Next Draw: 30s</span>
</div>

<!-- 1. NEXT ROUND AI PREDICTION -->
<div class="card">
    <div class="card-title">
        <span>🎯 AI Next Period Signal</span>
        <span class="mode-pill" id="ai-mode">Multi-Engine</span>
    </div>

    <div class="ai-hero">
        <div>
            <div style="font-size:12px; color:#94a3b8;">Target Period</div>
            <div style="font-size:18px; font-weight:800; font-family:monospace; color:#f8fafc;" id="next-period">-</div>
            <div style="font-size:11px; color:#64748b; margin-top:4px;">Last Num: <b style="color:#f59e0b;" id="last-num-txt">-</b></div>
        </div>
        <div style="text-align:right;">
            <div id="pred-signal" class="pred-big">-</div>
            <div style="font-size:12px; font-weight:bold; color:#cbd5e1;" id="pred-conf">0% Confidence</div>
        </div>
    </div>

    <div style="font-size:11px; display:flex; justify-content:space-between; font-weight:bold;">
        <span style="color:#ef4444;">BIG: <span id="bar-big-txt">50%</span></span>
        <span style="color:#3b82f6;">SMALL: <span id="bar-small-txt">50%</span></span>
    </div>
    <div class="bar-wrap">
        <div class="bar-big" id="bar-big" style="width:50%;"></div>
        <div class="bar-small" id="bar-small" style="width:50%;"></div>
    </div>

    <div class="sub-engines">
        <div class="eng-item">Number AI<b id="eng-num">50%</b></div>
        <div class="eng-item">Pattern N-Gram<b id="eng-ngram">50%</b></div>
        <div class="eng-item">Streak Break<b id="eng-streak">50%</b></div>
        <div class="eng-item">Regime Flow<b id="eng-mom">50%</b></div>
    </div>
</div>

<!-- 2. WIN/LOSS & CONSECUTIVE LOSS TRACKER WITH RESET BUTTON -->
<div class="card">
    <div class="card-title">
        <span>📊 AI Accuracy & Loss Streak</span>
        <div style="display:flex; align-items:center; gap:8px;">
            <span id="win-rate-badge" style="color:#10b981;">0% Win Rate</span>
            <button class="reset-btn" onclick="resetOnlyStats()">🔄 Reset Stats</button>
        </div>
    </div>

    <div class="grid-4">
        <div class="metric-box" style="border-color:#ef444444;">
            <div class="metric-label">Current Lagatar Loss</div>
            <div class="metric-val" id="cur-loss" style="color:#ef4444;">0</div>
        </div>
        <div class="metric-box" style="border-color:#f59e0b44;">
            <div class="metric-label">Highest Loss (Max)</div>
            <div class="metric-val" id="max-loss" style="color:#f59e0b;">0</div>
        </div>
        <div class="metric-box" style="border-color:#10b98144;">
            <div class="metric-label">Current Win Streak</div>
            <div class="metric-val" id="cur-win" style="color:#10b981;">0</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Pass / Fail</div>
            <div class="metric-val" style="font-size:15px; color:#f8fafc;">
                <span style="color:#10b981;" id="tot-win">0</span>W / <span style="color:#ef4444;" id="tot-loss">0</span>L
            </div>
        </div>
    </div>
    <div style="margin-top:8px; font-size:11px; color:#94a3b8; text-align:center;" id="seq-bar">
        Recent Sequence: -
    </div>
</div>

<!-- 3. 0 TO 9 DEEP NUMBER ANALYTICS MATRIX -->
<div class="card" style="padding:12px 0 0 0; overflow:hidden;">
    <div class="card-title" style="padding:0 14px 8px 14px; margin-bottom:0;">
        <span>🔢 All Numbers (0–9) Deep Pattern Matrix</span>
        <span style="font-size:10px; color:#f59e0b;">Highlighted = Last Number</span>
    </div>
    <div style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>Num</th>
                    <th>Hits</th>
                    <th>Gap</th>
                    <th>Next Big%</th>
                    <th>Next Small%</th>
                    <th>Break%</th>
                    <th>Top Next Nums</th>
                </tr>
            </thead>
            <tbody id="num-matrix-body">
                <tr><td colspan="7">Loading 0-9 number matrix...</td></tr>
            </tbody>
        </table>
    </div>
</div>

<!-- 4. RECENT DRAWS + AI VERIFICATION LOG -->
<div class="card" style="padding:12px 0 0 0; overflow:hidden;">
    <div class="card-title" style="padding:0 14px 8px 14px; margin-bottom:0;">
        <span>📜 Live Results & AI Pass/Fail History</span>
    </div>
    <table>
        <thead>
            <tr>
                <th>Period</th>
                <th>Num</th>
                <th>Actual</th>
                <th>AI Signal</th>
                <th>Result</th>
            </tr>
        </thead>
        <tbody id="results-body">
            <tr><td colspan="5" style="padding:20px;">Syncing live data...</td></tr>
        </tbody>
    </table>
</div>

<script>
let roundCountdown = 30;
let lastSeenPeriod = null;

async function resetOnlyStats() {
    if (!confirm("Kya aap sirf Win/Loss aur Streak counters ko 0 par reset karna chahte hain? (Aapka Saved Draws data safe rahega)")) {
        return;
    }
    try {
        const res = await fetch("/api/reset-stats", { method: "POST" });
        if (res.ok) {
            await loadDashboard();
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadDashboard() {
    try {
        const res = await fetch("/api/analyze");
        const d = await res.json();
        if (!d.total_records) return;

        if (lastSeenPeriod !== d.latest_period) {
            lastSeenPeriod = d.latest_period;
            roundCountdown = 30;
        }

        document.getElementById("total-db").innerText = d.total_records;
        document.getElementById("streak-info").innerText = "Running: " + d.cur_streak;
        document.getElementById("next-period").innerText = d.next_period;
        document.getElementById("last-num-txt").innerText = d.latest_number + " (" + d.latest_size + ")";
        document.getElementById("seq-bar").innerText = "Recent Flow: " + d.last_8_seq;

        const ai = d.live_ai;
        const sigEl = document.getElementById("pred-signal");
        sigEl.innerText = ai.suggestion;
        sigEl.className = ai.suggestion === "BIG" ? "pred-big" : "pred-small";
        document.getElementById("pred-conf").innerText = ai.confidence + "% Probability";
        document.getElementById("ai-mode").innerText = ai.mode;

        document.getElementById("bar-big-txt").innerText = ai.big_prob + "%";
        document.getElementById("bar-small-txt").innerText = ai.small_prob + "%";
        document.getElementById("bar-big").style.width = ai.big_prob + "%";
        document.getElementById("bar-small").style.width = ai.small_prob + "%";

        document.getElementById("eng-num").innerText = ai.details.num_trans + "%";
        document.getElementById("eng-ngram").innerText = ai.details.ngram + "%";
        document.getElementById("eng-streak").innerText = ai.details.streak + "%";
        document.getElementById("eng-mom").innerText = ai.details.momentum + "%";

        const s = d.stats;
        document.getElementById("win-rate-badge").innerText = s.win_rate + "% (" + s.total_preds + "r)";
        document.getElementById("cur-loss").innerText = s.current_loss_streak;
        document.getElementById("max-loss").innerText = s.max_loss_streak;
        document.getElementById("cur-win").innerText = s.current_win_streak;
        document.getElementById("tot-win").innerText = s.total_wins;
        document.getElementById("tot-loss").innerText = s.total_losses;

        const mBody = document.getElementById("num-matrix-body");
        mBody.innerHTML = d.number_matrix.map(n => {
            const colorCls = (n.number === 0 || n.number === 5) ? "Violet" : ([1,3,7,9].includes(n.number) ? "Green" : "Red");
            return `
                <tr class="${n.is_latest ? 'active-num-row' : ''}">
                    <td><span class="badge ${colorCls}">${n.number}</span></td>
                    <td><b>${n.hits}</b></td>
                    <td style="color:${n.gap >= 15 ? '#f59e0b' : '#94a3b8'};">${n.gap}r</td>
                    <td style="color:#ef4444; font-weight:bold;">${n.big_pct}%</td>
                    <td style="color:#3b82f6; font-weight:bold;">${n.small_pct}%</td>
                    <td style="color:#f59e0b;">${n.break_pct}%</td>
                    <td style="font-family:monospace; color:#cbd5e1;">${n.top_next}</td>
                </tr>
            `;
        }).join("");

        const rBody = document.getElementById("results-body");
        rBody.innerHTML = d.recent.map(r => {
            let statusHtml = '<span style="color:#64748b;">-</span>';
            if (r.status === "WIN") statusHtml = '<span class="win-pill">WIN ✅</span>';
            if (r.status === "LOSS") statusHtml = '<span class="loss-pill">LOSS ❌</span>';
            return `
                <tr>
                    <td style="font-family:monospace; font-size:11px; color:#cbd5e1;">${r.period.slice(-6)}</td>
                    <td><span class="badge ${r.color}">${r.number}</span></td>
                    <td><span class="size-tag size-${r.size}">${r.size}</span></td>
                    <td style="font-weight:bold; color:${r.ai_pred === 'BIG' ? '#ef4444' : '#3b82f6'};">
                        ${r.ai_pred} <span style="font-size:10px; color:#64748b;">(${r.ai_conf}%)</span>
                    </td>
                    <td>${statusHtml}</td>
                </tr>
            `;
        }).join("");

    } catch (err) {
        console.error(err);
    }
}

setInterval(() => {
    if (roundCountdown > 0) {
        roundCountdown--;
    }
    const cdEl = document.getElementById("countdown");
    if (roundCountdown > 0) {
        cdEl.innerText = "Next Draw: " + roundCountdown + "s";
        cdEl.style.color = roundCountdown <= 5 ? "#ef4444" : "#38bdf8";
    } else {
        cdEl.innerText = "Fetching result...";
        cdEl.style.color = "#f59e0b";
    }

    if (roundCountdown % 3 === 0 || roundCountdown <= 2) {
        loadDashboard();
    }
}, 1000);

loadDashboard();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/api/analyze")
def api_analyze():
    sync_and_evaluate()
    return jsonify(build_dashboard_payload())

@app.route("/ping")
def ping():
    sync_and_evaluate()
    return jsonify({"status": "alive", "timestamp": int(time.time())})

@app.route("/api/reset-stats", methods=["POST"])
def api_reset_stats():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT MAX(period) FROM draws")
        row = c.fetchone()
        latest_period = row[0] if (row and row[0]) else "0"
        c.execute("""
            INSERT OR REPLACE INTO settings (key, value)
            VALUES ('reset_after_period', ?)
        """, (latest_period,))
        conn.commit()
        conn.close()
    return jsonify({"ok": True, "reset_after_period": latest_period})

if __name__ == "__main__":
    ensure_collector_started()
    port = int(os.environ.get("PORT", 5000))
    print(f"AI Pro Terminal Running: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
