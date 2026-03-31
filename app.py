import io
import re
import json
import time
import math
import random
import zipfile
import hashlib
import datetime as dt
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import pytz
import requests
from requests.utils import quote
import streamlit as st

# ---------------- HARD-CODED CONFIG ----------------
BASE_URL = "https://centraldashboard-beta.moveinsync.com/centralized-dashboard/locations/locations"
IST = pytz.timezone("Asia/Kolkata")

TOKEN_URL = "https://script.google.com/macros/s/AKfycbwIqPMIbFCRNkcTpN_T2iPBFCG8nXE8cvjLlVTje1LprrDC07pir54EPqPIdk4GX0yxmw/exec"
DEFAULT_BUID = "tepl"
DEFAULT_DEVICE_TYPE = "FIXED_DEVICE"
DEFAULT_SEGMENT_HOURS = 4
DEFAULT_WORKERS = 8
DEFAULT_MIN_INTERVAL_S = 0.20
TOKEN_REFRESH_INTERVAL_SECONDS = 600
REQUEST_TIMEOUT_S = 60
RESULT_CACHE_TTL_SECONDS = 900  # 15 mins
MAX_SPEED_KMPH = 100.0
APP_TITLE = "MoveInSync Location Intelligence"

# ---------------- PAGE ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="📍", layout="wide")

# ---------------- THEME ----------------
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mis-green: #53c21b;
            --mis-green-dark: #2d8f0c;
            --mis-bg: #f3fbef;
            --mis-text: #121316;
            --mis-subtle: #687076;
            --mis-card: #ffffff;
            --mis-border: rgba(0,0,0,0.08);
            --mis-shadow: 0 12px 30px rgba(18,19,22,0.08);
            --mis-radius: 20px;
        }
        .stApp {
            background: linear-gradient(180deg, #f6fcf2 0%, #eef8ea 100%);
            color: var(--mis-text);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .mis-hero {
            background: radial-gradient(circle at top left, rgba(83,194,27,0.18), transparent 38%), linear-gradient(135deg, #eff9ea 0%, #f7fcf5 55%, #edf8e9 100%);
            border: 1px solid rgba(83,194,27,0.14);
            border-radius: 28px;
            padding: 28px 30px;
            box-shadow: var(--mis-shadow);
            margin-bottom: 1rem;
        }
        .mis-badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.2px;
            background: rgba(83,194,27,0.10);
            color: var(--mis-green-dark);
            margin-bottom: 12px;
        }
        .mis-hero h1 {
            margin: 0 0 10px 0;
            font-size: 52px;
            line-height: 1.03;
            font-weight: 800;
            letter-spacing: -1.6px;
            color: var(--mis-text);
        }
        .mis-hero p {
            margin: 0;
            font-size: 18px;
            line-height: 1.5;
            color: #4f5b62;
            max-width: 760px;
        }
        .mis-card {
            background: var(--mis-card);
            border: 1px solid var(--mis-border);
            border-radius: var(--mis-radius);
            padding: 20px 22px;
            box-shadow: var(--mis-shadow);
        }
        .mis-kpi {
            background: white;
            border: 1px solid var(--mis-border);
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 8px 22px rgba(18,19,22,0.06);
        }
        .mis-kpi .label {
            color: var(--mis-subtle);
            font-size: 13px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .mis-kpi .value {
            font-size: 32px;
            line-height: 1;
            font-weight: 800;
            color: var(--mis-text);
        }
        .mis-kpi .sub {
            margin-top: 8px;
            font-size: 12px;
            color: var(--mis-subtle);
        }
        .mis-tabnote {
            color: var(--mis-subtle);
            font-size: 14px;
            margin-bottom: 14px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 18px;
            padding-right: 18px;
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(83,194,27,0.14) !important;
            color: #1e4010 !important;
            border-color: rgba(83,194,27,0.35) !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px !important;
            border: none !important;
            background: linear-gradient(180deg, #58cb1d 0%, #45b117 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(83,194,27,0.28);
            min-height: 46px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(180deg, #4cb418 0%, #3f9b14 100%) !important;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 18px !important;
            border: 1px dashed rgba(83,194,27,0.45) !important;
            background: rgba(255,255,255,0.68) !important;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 8px 22px rgba(18,19,22,0.06);
        }
        .mis-footer {
            font-size: 13px;
            color: var(--mis-subtle);
            margin-top: 8px;
        }
        .mis-login-wrap {
            max-width: 430px;
            margin: 0 auto;
        }
        .mis-login-note {
            text-align: center;
            font-size: 15px;
            color: var(--mis-subtle);
            margin: 0 0 10px 0;
        }
        .mis-loader {
            padding: 18px 20px;
            border-radius: 18px;
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(83,194,27,0.20);
            box-shadow: 0 10px 30px rgba(18,19,22,0.08);
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 12px;
        }
        .mis-loader-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--mis-green);
            box-shadow: 0 0 0 rgba(83,194,27,0.5);
            animation: pulseMis 1.4s infinite;
        }
        .mis-loader-bar {
            flex: 1;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(83,194,27,0.15) 0%, rgba(83,194,27,0.75) 50%, rgba(83,194,27,0.15) 100%);
            background-size: 200% 100%;
            animation: shimmerMis 1.6s linear infinite;
        }
        @keyframes pulseMis {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(83,194,27,0.5); }
            70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(83,194,27,0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(83,194,27,0); }
        }
        @keyframes shimmerMis {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_theme()

# ---------------- SESSION ----------------
for key, default_value in {
    "raw_payload": None,
    "distance_cache": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ---------------- AUTH ----------------
def check_password() -> bool:
    if "APP_PASSWORD" not in st.secrets:
        st.error("Missing APP_PASSWORD in secrets.")
        return False

    if st.session_state.get("authenticated", False):
        return True

    st.markdown(
        """
        <div class="mis-hero">
            <div class="mis-badge">Secure access</div>
            <h1>Think Office Commute.<br>Think of Us.</h1>
            <p>Upload IMEIs, fetch raw location history, calculate distance in-memory, and download operational outputs with a premium MoveInSync-style experience.</p>
        </div>
        <div class="mis-login-wrap">
            <p class="mis-login-note"><b>Enter your workspace password</b> to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Workspace password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Enter workspace", use_container_width=True)

    if submitted:
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False

if not check_password():
    st.stop()

# ---------------- HELPERS ----------------
def parse_ist_datetime(s: str) -> dt.datetime:
    return IST.localize(dt.datetime.strptime(s.strip(), "%d-%m-%Y %H:%M"))

def epoch_ms_to_ist_str(epoch_ms: Optional[int]) -> str:
    if epoch_ms is None or pd.isna(epoch_ms):
        return ""
    utc_dt = dt.datetime.utcfromtimestamp(float(epoch_ms) / 1000.0).replace(tzinfo=pytz.utc)
    return utc_dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")

def read_imeis_from_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    df = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        for sep in [",", "\t", ";", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", encoding=enc)
                if df is not None and df.shape[1] >= 1:
                    break
            except Exception:
                df = None
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        raise RuntimeError("Could not read uploaded CSV.")

    df.columns = [str(c).strip() for c in df.columns]
    imei_col_candidates = ["Fixed Device IMEI", "IMEI", "imei", "FixedDeviceIMEI"]
    imei_col = next((c for c in imei_col_candidates if c in df.columns), df.columns[0])

    df = df.rename(columns={imei_col: "imei"})
    df["imei"] = df["imei"].astype(str).str.strip()
    df = df[df["imei"].str.len() > 0].drop_duplicates(subset=["imei"]).reset_index(drop=True)
    return df

def make_segment_list(start_ist: dt.datetime, end_ist: dt.datetime, segment_hours: int) -> List[Tuple[dt.datetime, dt.datetime]]:
    segs = []
    cur = start_ist
    delta = dt.timedelta(hours=segment_hours)
    while cur < end_ist:
        nxt = min(cur + delta, end_ist)
        segs.append((cur, nxt))
        cur = nxt
    return segs

def build_cache_key(imeis: List[str], start_str: str, end_str: str, workers: int) -> str:
    payload = {
        "imeis": sorted([str(x).strip() for x in imeis]),
        "start": start_str.strip(),
        "end": end_str.strip(),
        "buid": DEFAULT_BUID,
        "device_type": DEFAULT_DEVICE_TYPE,
        "segment_hours": DEFAULT_SEGMENT_HOURS,
        "workers": int(workers),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    avg_lat = (lat1 + lat2) / 2
    lat_dist = (lat2 - lat1) * 111
    lon_dist = (lon2 - lon1) * 111 * math.cos(avg_lat * math.pi / 180)
    return math.sqrt(lat_dist**2 + lon_dist**2)

def json_download_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

def csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def show_loader(message: str):
    st.markdown(
        f"""
        <div class="mis-loader">
            <div class="mis-loader-dot"></div>
            <div style="min-width:230px;font-weight:700;color:#1e4010;">{message}</div>
            <div class="mis-loader-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def operational_window_for_date(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    start_dt = IST.localize(dt.datetime.combine(day, dt.time(3, 0)))
    end_dt = IST.localize(dt.datetime.combine(day + dt.timedelta(days=1), dt.time(3, 0))) - dt.timedelta(minutes=1)
    return start_dt, end_dt

def build_operational_day_ranges(start_dt: dt.datetime, end_dt: dt.datetime) -> List[Tuple[str, dt.datetime, dt.datetime]]:
    dates = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="D")
    ranges = []
    for ts in dates:
        day = ts.date()
        win_start, win_end = operational_window_for_date(day)
        ranges.append((day.strftime("%d-%m-%Y"), win_start, win_end))
    return ranges

# ---------------- TOKEN CACHE ----------------
_token_lock = Lock()
_cached_token: Optional[str] = None
_cached_token_ts: float = 0.0

def extract_token_from_response(text: str) -> str:
    token = text.strip()
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1]
    token = token.replace('""', '"').strip()
    if not token:
        raise RuntimeError("Downloaded token is empty.")
    if "<html" in token.lower() or "doctype html" in token.lower():
        raise RuntimeError("Token URL returned HTML instead of token.")
    if len(token) < 20:
        raise RuntimeError(f"Downloaded token looks too short: {token!r}")
    return token

def fetch_new_token(timeout: int = 30) -> str:
    resp = requests.get(TOKEN_URL, timeout=timeout)
    resp.raise_for_status()
    return extract_token_from_response(resp.text)

def get_token(force_refresh: bool = False) -> str:
    global _cached_token, _cached_token_ts
    with _token_lock:
        now = time.time()
        needs_refresh = (
            force_refresh
            or _cached_token is None
            or (now - _cached_token_ts) >= TOKEN_REFRESH_INTERVAL_SECONDS
        )
        if needs_refresh:
            _cached_token = fetch_new_token()
            _cached_token_ts = now
        return _cached_token

# ---------------- RESULT CACHE ----------------
_result_cache_lock = Lock()
_result_cache: Dict[str, Dict[str, Any]] = {}

def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    with _result_cache_lock:
        item = _result_cache.get(cache_key)
        if not item:
            return None
        age = time.time() - item["ts"]
        if age > RESULT_CACHE_TTL_SECONDS:
            del _result_cache[cache_key]
            return None
        return item["data"]

def set_cached_result(cache_key: str, data: Dict[str, Any]) -> None:
    with _result_cache_lock:
        _result_cache[cache_key] = {"ts": time.time(), "data": data}

# ---------------- THROTTLE / BACKOFF ----------------
_rate_lock = Lock()
_last_request_ts = 0.0
RECOVERABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}

def global_cooldown(min_interval_s: float):
    global _last_request_ts
    if min_interval_s <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = (_last_request_ts + min_interval_s) - now
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.time()

def compute_backoff_s(attempt: int, base: float, cap: float, jitter: float) -> float:
    expo = min(cap, base * (2 ** max(0, attempt - 1)))
    return expo + random.uniform(0, jitter)

# ---------------- CORE FETCH ----------------
def fetch_segment(
    imei: str,
    seg_start: dt.datetime,
    seg_end: dt.datetime,
    *,
    min_interval_s: float,
    backoff_base_s: float,
    backoff_cap_s: float,
    backoff_jitter_s: float,
    max_attempts: int,
) -> Dict[str, Any]:
    seg_start_str = seg_start.strftime("%d-%m-%Y %H:%M")
    seg_end_str = seg_end.strftime("%d-%m-%Y %H:%M")

    url = (
        f"{BASE_URL}/{imei}"
        f"?buid={quote(DEFAULT_BUID)}"
        f"&startTime={quote(seg_start_str)}"
        f"&endTime={quote(seg_end_str)}"
        f"&deviceType={quote(DEFAULT_DEVICE_TYPE)}"
    )

    attempt = 0
    while True:
        attempt += 1
        if max_attempts > 0 and attempt > max_attempts:
            return {
                "imei": imei,
                "segment_start": seg_start_str,
                "segment_end": seg_end_str,
                "status": "failed",
                "error": f"exhausted_attempts={max_attempts}",
            }

        try:
            token = get_token(force_refresh=False)
        except Exception:
            sleep_s = compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s)
            time.sleep(sleep_s)
            continue

        global_cooldown(min_interval_s)

        try:
            resp = requests.get(url, headers={"x-wis-token": token}, timeout=REQUEST_TIMEOUT_S)
        except Exception:
            sleep_s = compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s)
            time.sleep(sleep_s)
            continue

        if resp.status_code in (401, 403):
            try:
                get_token(force_refresh=True)
            except Exception:
                pass
            sleep_s = max(5.0, compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s))
            time.sleep(sleep_s)
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                sleep_s = compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s)
                time.sleep(sleep_s)
                continue

            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict) and "locationCO" in data:
                entries = [data]
            else:
                entries = []

            for entry in entries:
                loc = entry.get("locationCO", {})
                loc["timestampIst"] = epoch_ms_to_ist_str(loc.get("timestamp"))
                loc["serverTimeIst"] = epoch_ms_to_ist_str(loc.get("serverTime"))
                loc["istOffsetMinutes"] = 330

            return {
                "imei": imei,
                "segment_start": seg_start_str,
                "segment_end": seg_end_str,
                "status": "saved",
                "count": len(entries),
                "data": entries,
            }

        if resp.status_code in RECOVERABLE_HTTP:
            sleep_s = compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s)
            time.sleep(sleep_s)
            continue

        return {
            "imei": imei,
            "segment_start": seg_start_str,
            "segment_end": seg_end_str,
            "status": "failed",
            "error": f"http_{resp.status_code}: {resp.text[:300]}",
        }

def fetch_run_payload(
    imeis: List[str],
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    workers: int,
    min_interval_s: float,
    max_attempts: int,
) -> Dict[str, Any]:
    segments = make_segment_list(start_dt, end_dt, DEFAULT_SEGMENT_HOURS)
    tasks = [(imei, s, e) for imei in imeis for (s, e) in segments]

    progress = st.progress(0, text="Starting location fetch...")
    status_box = st.empty()
    show_loader("Fetching raw location data")

    results: List[Dict[str, Any]] = []
    completed = 0
    saved = 0
    failed = 0
    total_rows = 0

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futures = [
            ex.submit(
                fetch_segment,
                imei,
                seg_start,
                seg_end,
                min_interval_s=float(min_interval_s),
                backoff_base_s=2.0,
                backoff_cap_s=60.0,
                backoff_jitter_s=1.5,
                max_attempts=int(max_attempts),
            )
            for imei, seg_start, seg_end in tasks
        ]

        for fut in as_completed(futures):
            item = fut.result()
            results.append(item)
            completed += 1
            if item.get("status") == "saved":
                saved += 1
                total_rows += int(item.get("count", 0))
            else:
                failed += 1

            progress.progress(
                completed / len(tasks) if tasks else 1.0,
                text=f"Completed {completed}/{len(tasks)} | Saved={saved} | Failed={failed} | Rows={total_rows}",
            )
            status_box.markdown(
                f"<div class='mis-card'><b>Live summary</b><div class='mis-footer'>Tasks: {len(tasks)} | Completed: {completed} | Saved: {saved} | Failed: {failed} | Location rows: {total_rows}</div></div>",
                unsafe_allow_html=True,
            )

    return {
        "runAtIst": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "cacheTtlSeconds": RESULT_CACHE_TTL_SECONDS,
        "buid": DEFAULT_BUID,
        "deviceType": DEFAULT_DEVICE_TYPE,
        "segmentHours": DEFAULT_SEGMENT_HOURS,
        "startInput": start_dt.strftime("%d-%m-%Y %H:%M"),
        "endInput": end_dt.strftime("%d-%m-%Y %H:%M"),
        "imeiCount": len(imeis),
        "totalTasks": len(tasks),
        "savedTasks": saved,
        "failedTasks": failed,
        "totalLocationRowsFetched": total_rows,
        "results": sorted(results, key=lambda x: (str(x.get("imei", "")), str(x.get("segment_start", "")))),
    }

def compute_distance_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    by_imei: Dict[str, List[Dict[str, Any]]] = {}

    for item in results:
        if item.get("status") != "saved":
            continue
        imei = str(item.get("imei", "")).strip()
        if not imei:
            continue
        by_imei.setdefault(imei, []).extend(item.get("data", []))

    for imei, entries in by_imei.items():
        normalized_points = []
        bad_rows = 0
        for entry in entries:
            loc = entry.get("locationCO", {})
            geo = str(loc.get("geocords", "")).strip()
            ts = loc.get("timestamp")
            if not geo or ts in (None, ""):
                bad_rows += 1
                continue
            try:
                lat_s, lon_s = geo.split(",")
                lat = float(lat_s)
                lon = float(lon_s)
                ts_ist = dt.datetime.fromtimestamp(float(ts) / 1000.0, tz=pytz.utc).astimezone(IST)
                normalized_points.append((ts_ist, lat, lon, geo))
            except Exception:
                bad_rows += 1

        if not normalized_points:
            rows.append(
                {
                    "IMEI": imei,
                    "Distance_KM": 0.0,
                    "Valid_Segments": 0,
                    "Ignored_Segments": 0,
                    "Point_Count": 0,
                    "First_Location": "",
                    "First_Location_Time": "",
                    "Last_Location": "",
                    "Last_Location_Time": "",
                    "Comment": "No usable points",
                }
            )
            continue

        normalized_points.sort(key=lambda x: x[0])
        total_dist = 0.0
        valid_segments = 0
        ignored_segments = 0
        comments = []

        for idx in range(1, len(normalized_points)):
            prev_ts, prev_lat, prev_lon, _ = normalized_points[idx - 1]
            curr_ts, curr_lat, curr_lon, _ = normalized_points[idx]
            dist = calculate_distance(prev_lat, prev_lon, curr_lat, curr_lon)
            dt_seconds = (curr_ts - prev_ts).total_seconds()
            if dt_seconds <= 0:
                ignored_segments += 1
                continue
            speed = dist / (dt_seconds / 3600.0)
            if speed > MAX_SPEED_KMPH:
                ignored_segments += 1
                comments.append(f"Ignored {dist:.2f}km at {speed:.0f}km/h")
                continue
            total_dist += dist
            valid_segments += 1

        first = normalized_points[0]
        last = normalized_points[-1]
        comment = []
        if bad_rows:
            comment.append(f"Bad rows skipped={bad_rows}")
        if comments:
            comment.append(comments[0])

        rows.append(
            {
                "IMEI": imei,
                "Distance_KM": round(total_dist, 2),
                "Valid_Segments": valid_segments,
                "Ignored_Segments": ignored_segments,
                "Point_Count": len(normalized_points),
                "First_Location": f"{first[1]:.5f},{first[2]:.5f}",
                "First_Location_Time": first[0].strftime("%d/%m/%Y %H:%M:%S"),
                "Last_Location": f"{last[1]:.5f},{last[2]:.5f}",
                "Last_Location_Time": last[0].strftime("%d/%m/%Y %H:%M:%S"),
                "Comment": "; ".join(comment),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["IMEI", "Distance_KM", "Valid_Segments", "Ignored_Segments", "Point_Count", "First_Location", "First_Location_Time", "Last_Location", "Last_Location_Time", "Comment"])

    return pd.DataFrame(rows).sort_values(["Distance_KM", "IMEI"], ascending=[False, True]).reset_index(drop=True)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="mis-hero">
        <div class="mis-badge">MoveInSync-style operational intelligence</div>
        <h1>Think Office Commute.<br>Think of Us.</h1>
        <p>Enter any time range you need. Download raw data for exactly that range, or calculate distance either for the same range or in operational 3 AM to 3 AM day buckets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### Workspace controls")
    workers = st.slider("Workers", min_value=1, max_value=24, value=DEFAULT_WORKERS)
    min_interval_s = st.slider("Min interval between requests (seconds)", min_value=0.00, max_value=2.00, value=float(DEFAULT_MIN_INTERVAL_S), step=0.05)
    max_attempts = st.number_input("Max attempts per segment", min_value=1, max_value=100, value=5)
    st.markdown("---")
    st.caption("Hardcoded internally")
    st.caption(f"BUID: {DEFAULT_BUID}")
    st.caption(f"Device Type: {DEFAULT_DEVICE_TYPE}")
    st.caption(f"Segment Hours: {DEFAULT_SEGMENT_HOURS}")
    st.caption("Token source: internal")

# ---------------- INPUTS ----------------
left, right = st.columns([1.3, 1])
with left:
    uploaded_file = st.file_uploader("Upload IMEI CSV", type=["csv"])
with right:
    c1, c2 = st.columns(2)
    with c1:
        start_str = st.text_input("Start IST (DD-MM-YYYY HH:MM)", value="01-02-2026 00:00")
    with c2:
        end_str = st.text_input("End IST (DD-MM-YYYY HH:MM)", value="01-02-2026 23:59")

run_clicked = st.button("Run location intelligence", type="primary", use_container_width=True)

if run_clicked:
    if not uploaded_file:
        st.error("Upload a CSV first.")
        st.stop()

    try:
        start_ist = parse_ist_datetime(start_str)
        end_ist = parse_ist_datetime(end_str)
        if end_ist <= start_ist:
            st.error("End must be after start.")
            st.stop()

        _ = get_token(force_refresh=True)
        df = read_imeis_from_uploaded_csv(uploaded_file)
        imeis = df["imei"].tolist()
        cache_key = build_cache_key(imeis, start_str, end_str, workers)
        cached_summary = get_cached_result(cache_key)

        if cached_summary is not None:
            raw_payload = dict(cached_summary)
            raw_payload["cacheUsed"] = True
            st.success("Using cached raw result from memory for this exact run input.")
        else:
            raw_payload = fetch_run_payload(
                imeis=imeis,
                start_dt=start_ist,
                end_dt=end_ist,
                workers=int(workers),
                min_interval_s=float(min_interval_s),
                max_attempts=int(max_attempts),
            )
            raw_payload["cacheUsed"] = False
            set_cached_result(cache_key, raw_payload)

        st.session_state["raw_payload"] = raw_payload
        st.session_state["distance_cache"] = {}
        st.success("Raw data is ready.")

    except Exception as e:
        st.error(f"Error: {e}")

raw_payload = st.session_state.get("raw_payload")

if raw_payload:
    raw_results = raw_payload.get("results", [])
    raw_preview_df = pd.DataFrame(
        [
            {
                "IMEI": x.get("imei"),
                "Segment Start": x.get("segment_start"),
                "Segment End": x.get("segment_end"),
                "Status": x.get("status"),
                "Count": x.get("count", 0),
                "Error": x.get("error", ""),
            }
            for x in raw_results
        ]
    )

    total_distance_preview = 0.0
    rows_fetched = int(raw_payload.get("totalLocationRowsFetched", 0))
    failed_tasks = int(raw_payload.get("failedTasks", 0))
    unique_imeis = int(raw_payload.get("imeiCount", 0))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='mis-kpi'><div class='label'>IMEIs</div><div class='value'>{unique_imeis}</div><div class='sub'>Unique uploaded devices</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='mis-kpi'><div class='label'>Location rows</div><div class='value'>{rows_fetched}</div><div class='sub'>Fetched into memory</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='mis-kpi'><div class='label'>Raw range</div><div class='value'>{raw_payload.get('startInput')}</div><div class='sub'>to {raw_payload.get('endInput')}</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='mis-kpi'><div class='label'>Failed tasks</div><div class='value'>{failed_tasks}</div><div class='sub'>{'Cache hit' if raw_payload.get('cacheUsed') else 'Fresh API run'}</div></div>", unsafe_allow_html=True)

    raw_tab, distance_tab, downloads_tab = st.tabs(["Raw Output", "Distance Summary", "Downloads"])

    with raw_tab:
        st.markdown("<div class='mis-tabnote'>Raw output always matches the exact start and end time entered above.</div>", unsafe_allow_html=True)
        st.dataframe(raw_preview_df, use_container_width=True, height=430)

    with distance_tab:
        st.markdown("<div class='mis-tabnote'>Choose how distance should be calculated. You can use the exact selected range or operational 3 AM to 3 AM day buckets based on the dates you entered.</div>", unsafe_allow_html=True)

        distance_mode = st.radio(
            "Distance calculation mode",
            ["Selected range", "Operational day (3 AM to 3 AM)"],
            horizontal=True,
            key="distance_mode",
        )

        if distance_mode == "Selected range":
            distance_cache_key = f"selected_range::{raw_payload.get('startInput')}::{raw_payload.get('endInput')}"
        else:
            distance_cache_key = f"operational::{raw_payload.get('startInput')}::{raw_payload.get('endInput')}"

        if st.button("Calculate distance summary", key="calc_distance", use_container_width=True):
            try:
                if distance_cache_key in st.session_state["distance_cache"]:
                    st.success("Using cached distance summary for this mode.")
                else:
                    show_loader("Calculating distance summary")
                    if distance_mode == "Selected range":
                        distance_df = compute_distance_summary(raw_payload.get("results", []))
                        st.session_state["distance_cache"][distance_cache_key] = {
                            "distance_df": distance_df.to_dict(orient="records"),
                            "source": "selected_range",
                        }
                    else:
                        start_dt = parse_ist_datetime(raw_payload.get("startInput"))
                        end_dt = parse_ist_datetime(raw_payload.get("endInput"))
                        imeis = sorted({str(x.get("imei", "")).strip() for x in raw_payload.get("results", []) if str(x.get("imei", "")).strip()})
                        op_ranges = build_operational_day_ranges(start_dt, end_dt)

                        op_results: List[Dict[str, Any]] = []
                        progress = st.progress(0, text="Starting operational day fetch...")
                        total_jobs = len(op_ranges)
                        completed_jobs = 0

                        for label, op_start, op_end in op_ranges:
                            partial_payload = fetch_run_payload(
                                imeis=imeis,
                                start_dt=op_start,
                                end_dt=op_end,
                                workers=int(workers),
                                min_interval_s=float(min_interval_s),
                                max_attempts=int(max_attempts),
                            )
                            for item in partial_payload.get("results", []):
                                item["operational_day"] = label
                            op_results.extend(partial_payload.get("results", []))
                            completed_jobs += 1
                            progress.progress(completed_jobs / total_jobs if total_jobs else 1.0, text=f"Operational day windows completed: {completed_jobs}/{total_jobs}")

                        distance_df = compute_distance_summary(op_results)
                        st.session_state["distance_cache"][distance_cache_key] = {
                            "distance_df": distance_df.to_dict(orient="records"),
                            "source": "operational_day",
                        }

            except Exception as e:
                st.error(f"Distance calculation failed: {e}")

        cached_distance = st.session_state["distance_cache"].get(distance_cache_key)
        if cached_distance:
            distance_df = pd.DataFrame(cached_distance.get("distance_df", []))
            if not distance_df.empty and "Distance_KM" in distance_df.columns:
                total_distance_preview = float(distance_df["Distance_KM"].sum())
            st.metric("Total distance (km)", f"{total_distance_preview:,.2f}")
            st.dataframe(distance_df, use_container_width=True, height=430)
        else:
            st.info("No distance summary calculated yet for the selected mode.")

    with downloads_tab:
        raw_json_bytes = json_download_bytes(raw_payload)
        raw_csv_bytes = csv_bytes_from_df(raw_preview_df if not raw_preview_df.empty else pd.DataFrame(columns=["IMEI", "Segment Start", "Segment End", "Status", "Count", "Error"]))

        active_distance_mode = st.session_state.get("distance_mode", "Selected range")
        if active_distance_mode == "Selected range":
            active_distance_key = f"selected_range::{raw_payload.get('startInput')}::{raw_payload.get('endInput')}"
        else:
            active_distance_key = f"operational::{raw_payload.get('startInput')}::{raw_payload.get('endInput')}"

        cached_distance = st.session_state["distance_cache"].get(active_distance_key)
        if cached_distance:
            distance_df = pd.DataFrame(cached_distance.get("distance_df", []))
        else:
            distance_df = pd.DataFrame(columns=["IMEI", "Distance_KM", "Valid_Segments", "Ignored_Segments", "Point_Count", "First_Location", "First_Location_Time", "Last_Location", "Last_Location_Time", "Comment"])

        distance_csv_bytes = csv_bytes_from_df(distance_df)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("raw_summary.json", raw_json_bytes)
            zf.writestr("raw_summary.csv", raw_csv_bytes)
            zf.writestr("distance_summary.csv", distance_csv_bytes)
        zip_buffer.seek(0)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download raw JSON",
                data=raw_json_bytes,
                file_name=f"raw_summary_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "Download raw CSV",
                data=raw_csv_bytes,
                file_name=f"raw_summary_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download distance summary CSV",
                data=distance_csv_bytes,
                file_name=f"distance_summary_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download everything as ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"location_intelligence_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.markdown("<div class='mis-footer'>Raw downloads always follow the exact time range entered. Distance summary can be calculated either for the entered range or operational 3 AM to 3 AM days.</div>", unsafe_allow_html=True)
