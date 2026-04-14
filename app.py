import io
import json
import time
import math
import random
import zipfile
import hashlib
import base64
import datetime as dt
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import pytz
import requests
from requests.utils import quote
import streamlit as st
import streamlit.components.v1 as components

# ---------------- HARD-CODED CONFIG ----------------
BASE_URL = "https://centraldashboard-beta.moveinsync.com/centralized-dashboard/locations/locations"
TOKEN_URL = "https://script.google.com/macros/s/AKfycbwIqPMIbFCRNkcTpN_T2iPBFCG8nXE8cvjLlVTje1LprrDC07pir54EPqPIdk4GX0yxmw/exec"
IST = pytz.timezone("Asia/Kolkata")

DEFAULT_BUID = "tepl"
DEFAULT_DEVICE_TYPE = "FIXED_DEVICE"
DEFAULT_SEGMENT_HOURS = 4
DEFAULT_WORKERS = 8
DEFAULT_MIN_INTERVAL_S = 0.20
TOKEN_REFRESH_INTERVAL_SECONDS = 600
REQUEST_TIMEOUT_S = 60
RESULT_CACHE_TTL_SECONDS = 900
MAX_SPEED_KMPH = 150.0
APP_TITLE = "MoveInSync Location Intelligence"

# ---------------- PAGE ----------------
st.set_page_config(page_title=APP_TITLE, page_icon="📍", layout="wide")


# ---------------- THEME ----------------
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #f6fbff;
            --bg-2: #eef5ff;
            --card: #ffffff;
            --text: #17324d;
            --subtle: #5c7289;
            --border: rgba(28, 73, 122, 0.10);
            --shadow: 0 12px 32px rgba(29, 60, 98, 0.08);
            --primary: #5fa8d3;
            --primary-dark: #4189b3;
            --primary-soft: #dcedf7;
            --accent: #8fd3c1;
            --accent-soft: #edf9f4;
            --lavender: #e9e4fb;
            --radius-xl: 24px;
            --radius-lg: 18px;
        }
        .stApp {
            background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            color: var(--text);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .mis-hero {
            background:
                radial-gradient(circle at top left, rgba(95,168,211,0.18), transparent 38%),
                radial-gradient(circle at top right, rgba(143,211,193,0.18), transparent 28%),
                linear-gradient(135deg, #f9fcff 0%, #f3f8ff 48%, #eef7f7 100%);
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 28px 30px;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }
        .mis-badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            background: var(--primary-soft);
            color: var(--primary-dark);
            margin-bottom: 12px;
        }
        .mis-hero h1 {
            margin: 0 0 10px 0;
            font-size: 50px;
            line-height: 1.03;
            font-weight: 800;
            letter-spacing: -1.4px;
            color: var(--text);
        }
        .mis-hero p {
            margin: 0;
            font-size: 18px;
            line-height: 1.55;
            color: var(--subtle);
            max-width: 800px;
        }
        .mis-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            padding: 20px 22px;
            box-shadow: var(--shadow);
        }
        .mis-kpi {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 18px;
            box-shadow: 0 8px 24px rgba(29, 60, 98, 0.06);
        }
        .mis-kpi .label {
            color: var(--subtle);
            font-size: 13px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .mis-kpi .value {
            color: var(--text);
            font-size: 30px;
            line-height: 1.05;
            font-weight: 800;
        }
        .mis-kpi .sub {
            margin-top: 8px;
            font-size: 12px;
            color: var(--subtle);
        }
        .mis-note {
            font-size: 14px;
            color: var(--subtle);
            margin-top: 6px;
        }
        .mis-page-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 22px;
            box-shadow: var(--shadow);
            min-height: 220px;
        }
        .mis-page-card h3 {
            margin: 0 0 8px 0;
            color: var(--text);
            font-size: 24px;
        }
        .mis-page-card p {
            margin: 0;
            color: var(--subtle);
            font-size: 15px;
            line-height: 1.55;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px !important;
            border: 1px solid rgba(65,137,179,0.10) !important;
            background: linear-gradient(180deg, #78b9de 0%, #5fa8d3 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(95,168,211,0.24);
            min-height: 46px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(180deg, #67aed7 0%, #4f99c5 100%) !important;
        }
        .stButton > button[kind="secondary"] {
            background: linear-gradient(180deg, #a8d9cd 0%, #8fd3c1 100%) !important;
            box-shadow: 0 10px 24px rgba(143,211,193,0.24);
        }
        .stTextInput input, .stDateInput input, .stTimeInput input, .stNumberInput input {
            border-radius: 12px !important;
            background: #ffffff !important;
            color: #17324d !important;
            border: 1px solid rgba(95,168,211,0.28) !important;
        }
        .stTextInput label, .stDateInput label, .stTimeInput label, .stNumberInput label,
        .stFileUploader label, .stRadio label, .stMarkdown, .stCaption {
            color: #35536f !important;
        }
        .stRadio [role="radiogroup"] label, .stRadio [role="radiogroup"] p {
            color: #17324d !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        .stRadio [data-baseweb="radio"] > div:first-child {
            background: #ffffff !important;
            border-color: rgba(95,168,211,0.55) !important;
        }
        div[data-testid="stFileUploader"] section {
            border-radius: 18px !important;
            border: 1px dashed rgba(95,168,211,0.45) !important;
            background: rgba(255,255,255,0.88) !important;
        }
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(180deg, #d8edf8 0%, #c4e3f1 100%) !important;
            color: #17324d !important;
            border: 1px solid rgba(95,168,211,0.25) !important;
            box-shadow: none !important;
        }
        .mis-login-wrap {
            max-width: 440px;
            margin: 0 auto;
        }
        .mis-login-note {
            text-align: center;
            color: var(--subtle);
            font-size: 15px;
            margin: 0 0 10px 0;
        }
        .mis-status {
            border-radius: 18px;
            border: 1px solid rgba(95,168,211,0.22);
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(236,245,255,0.95) 100%);
            padding: 18px;
            box-shadow: var(--shadow);
            margin-bottom: 12px;
        }
        .mis-status-title {
            font-size: 16px;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 4px;
        }
        .mis-status-sub {
            font-size: 13px;
            color: var(--subtle);
        }
        .mis-loader {
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 16px;
            align-items: center;
        }
        .mis-loader-visual {
            width: 94px;
            height: 94px;
            border-radius: 24px;
            position: relative;
            background: linear-gradient(180deg, #e9f5fc 0%, #f6fbff 100%);
            border: 1px solid rgba(95,168,211,0.18);
            overflow: hidden;
        }
        .mis-loader-road {
            position: absolute;
            left: 10px;
            right: 10px;
            top: 50%;
            height: 12px;
            transform: translateY(-50%);
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(95,168,211,0.15) 0%, rgba(95,168,211,0.75) 50%, rgba(95,168,211,0.15) 100%);
            background-size: 200% 100%;
            animation: mis-road 1.8s linear infinite;
        }
        .mis-loader-car {
            position: absolute;
            top: 34px;
            left: 10px;
            font-size: 26px;
            animation: mis-car 2.2s ease-in-out infinite;
        }
        .mis-step {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-right: 10px;
            margin-top: 6px;
            padding: 7px 10px;
            border-radius: 999px;
            background: #f3f8fd;
            border: 1px solid rgba(95,168,211,0.14);
            color: var(--subtle);
            font-size: 12px;
            font-weight: 700;
        }
        .mis-step.active {
            background: var(--primary-soft);
            color: var(--primary-dark);
        }
        .mis-step.done {
            background: var(--accent-soft);
            color: #3b7d6d;
        }
        @keyframes mis-road {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        @keyframes mis-car {
            0% { left: 8px; }
            50% { left: 54px; }
            100% { left: 8px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

# ---------------- SESSION ----------------
def ensure_session_defaults() -> None:
    defaults = {
        "authenticated": False,
        "nav_page": "home",
        "distance_mode": "operational",
        "distance_cache": {},
        "last_raw_zip": None,
        "last_distance_zip": None,
        "last_raw_payload": None,
        "last_distance_df": None,
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ensure_session_defaults()


# ---------------- UI HELPERS ----------------
def show_notice(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="mis-status">
            <div class="mis-status-title">{title}</div>
            <div class="mis-status-sub">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_processing_panel(title: str, body: str, active_step: int = 1, target=None) -> None:
    steps = ["Preparing input", "Calling APIs", "Building files"]
    chips = []
    for idx, label in enumerate(steps, start=1):
        state = "active" if idx == active_step else "done" if idx < active_step else ""
        chips.append(f'<span class="mis-step {state}">{idx}. {label}</span>')
    html = f"""
        <div class="mis-status">
            <div class="mis-loader">
                <div class="mis-loader-visual">
                    <div class="mis-loader-road"></div>
                    <div class="mis-loader-car">🚐</div>
                </div>
                <div>
                    <div class="mis-status-title">{title}</div>
                    <div class="mis-status-sub">{body}</div>
                    <div>{''.join(chips)}</div>
                </div>
            </div>
        </div>
        """
    if target is None:
        st.markdown(html, unsafe_allow_html=True)
    else:
        target.markdown(html, unsafe_allow_html=True)


def auto_download_bytes(data: bytes, file_name: str, mime: str, key: str) -> None:
    b64 = base64.b64encode(data).decode()
    html = f"""
    <html>
      <body>
        <a id="dl_{key}" href="data:{mime};base64,{b64}" download="{file_name}"></a>
        <script>
          const link = document.getElementById("dl_{key}");
          if (link) {{ link.click(); }}
        </script>
      </body>
    </html>
    """
    components.html(html, height=0, width=0)


def go_to(page: str) -> None:
    st.session_state["nav_page"] = page
    st.rerun()


# ---------------- AUTH ----------------
def check_password() -> bool:
    if "APP_PASSWORD" not in st.secrets:
        show_notice("Password is not configured", "Please add APP_PASSWORD in Streamlit secrets before using the app.")
        return False

    if st.session_state.get("authenticated", False):
        return True

    st.markdown(
        """
        <div class="mis-hero">
            <div class="mis-badge">Secure access</div>
            <h1>Think Office Commute.<br>Think of Us.</h1>
            <p>Choose raw location download or distance calculation after login. The app keeps processing in memory, uses multiple workers, and packages outputs as ZIP files.</p>
        </div>
        <div class="mis-login-wrap">
            <p class="mis-login-note"><b>Enter your password</b> to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Password", type="password", placeholder="Password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            show_notice("Access not granted", "The password entered did not match the configured password.")
    return False


if not check_password():
    st.stop()


# ---------------- HELPERS ----------------
def parse_ist_datetime_dash(s: str) -> dt.datetime:
    return IST.localize(dt.datetime.strptime(s.strip(), "%d-%m-%Y %H:%M"))


def parse_ist_datetime_slash(s: str, with_seconds: bool = False) -> dt.datetime:
    fmt = "%d/%m/%Y %H:%M:%S" if with_seconds else "%d/%m/%Y %H:%M"
    return IST.localize(dt.datetime.strptime(s.strip(), fmt))


def parse_ist_date_slash(s: str) -> dt.date:
    return dt.datetime.strptime(s.strip(), "%d/%m/%Y").date()


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
                candidate = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", encoding=enc)
                if candidate is not None and candidate.shape[1] >= 1:
                    df = candidate
                    break
            except Exception:
                pass
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        raise RuntimeError("Could not read the uploaded CSV.")

    df.columns = [str(c).strip() for c in df.columns]
    imei_col_candidates = ["Fixed Device IMEI", "IMEI", "imei", "FixedDeviceIMEI"]
    imei_col = next((c for c in imei_col_candidates if c in df.columns), df.columns[0])

    df = df.rename(columns={imei_col: "imei"})
    df["imei"] = df["imei"].astype(str).str.strip()
    df = df[df["imei"].str.len() > 0].drop_duplicates(subset=["imei"]).reset_index(drop=True)
    return df


def normalize_imei_for_api(imei: str) -> str:
    value = str(imei).strip()
    if value.startswith("3") and not value.startswith("0"):
        return f"0{value}"
    return value


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
    return math.sqrt(lat_dist ** 2 + lon_dist ** 2)


def json_download_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def operational_window_for_date(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    start_dt = IST.localize(dt.datetime.combine(day, dt.time(3, 0, 0)))
    end_dt = IST.localize(dt.datetime.combine(day + dt.timedelta(days=1), dt.time(3, 0, 0)))
    return start_dt, end_dt


def build_operational_day_ranges(start_day: dt.date, end_day: dt.date) -> List[Tuple[str, dt.datetime, dt.datetime]]:
    ranges = []
    cur = start_day
    while cur <= end_day:
        s, e = operational_window_for_date(cur)
        ranges.append((cur.strftime("%d/%m/%Y"), s, e))
        cur += dt.timedelta(days=1)
    return ranges


def flatten_raw_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        if item.get("status") != "saved":
            rows.append(
                {
                    "IMEI": item.get("imei", ""),
                    "API_IMEI": item.get("api_imei", normalize_imei_for_api(item.get("imei", ""))),
                    "Segment_Start": item.get("segment_start", ""),
                    "Segment_End": item.get("segment_end", ""),
                    "Status": item.get("status", ""),
                    "Latitude": "",
                    "Longitude": "",
                    "Timestamp_IST": "",
                    "ServerTime_IST": "",
                    "Operational_Day": item.get("operational_day", ""),
                    "Error": item.get("error", ""),
                }
            )
            continue

        entries = item.get("data", []) or []
        if not entries:
            rows.append(
                {
                    "IMEI": item.get("imei", ""),
                    "API_IMEI": item.get("api_imei", normalize_imei_for_api(item.get("imei", ""))),
                    "Segment_Start": item.get("segment_start", ""),
                    "Segment_End": item.get("segment_end", ""),
                    "Status": item.get("status", ""),
                    "Latitude": "",
                    "Longitude": "",
                    "Timestamp_IST": "",
                    "ServerTime_IST": "",
                    "Operational_Day": item.get("operational_day", ""),
                    "Error": "",
                }
            )
            continue

        for entry in entries:
            loc = entry.get("locationCO", {})
            geo = str(loc.get("geocords", "") or "").strip()
            lat = ""
            lon = ""
            if geo and "," in geo:
                lat, lon = [x.strip() for x in geo.split(",", 1)]
            rows.append(
                {
                    "IMEI": item.get("imei", ""),
                    "API_IMEI": item.get("api_imei", normalize_imei_for_api(item.get("imei", ""))),
                    "Segment_Start": item.get("segment_start", ""),
                    "Segment_End": item.get("segment_end", ""),
                    "Status": item.get("status", ""),
                    "Latitude": lat,
                    "Longitude": lon,
                    "Timestamp_IST": loc.get("timestampIst", ""),
                    "ServerTime_IST": loc.get("serverTimeIst", ""),
                    "Operational_Day": item.get("operational_day", ""),
                    "Error": "",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["IMEI", "API_IMEI", "Segment_Start", "Segment_End", "Status", "Latitude", "Longitude", "Timestamp_IST", "ServerTime_IST", "Operational_Day", "Error"])
    return pd.DataFrame(rows)


def preview_summary_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "IMEI": x.get("imei"),
                "API_IMEI": x.get("api_imei", normalize_imei_for_api(x.get("imei", ""))),
                "Segment Start": x.get("segment_start"),
                "Segment End": x.get("segment_end"),
                "Status": x.get("status"),
                "Count": x.get("count", 0),
                "Operational Day": x.get("operational_day", ""),
                "Error": x.get("error", ""),
            }
            for x in results
        ]
    )


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
        raise RuntimeError("Token URL returned HTML instead of a token.")
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

    api_imei = normalize_imei_for_api(imei)
    url = (
        f"{BASE_URL}/{api_imei}"
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
                "api_imei": api_imei,
                "segment_start": seg_start_str,
                "segment_end": seg_end_str,
                "status": "failed",
                "error": f"exhausted_attempts={max_attempts}",
            }

        try:
            token = get_token(force_refresh=False)
        except Exception:
            time.sleep(compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s))
            continue

        global_cooldown(min_interval_s)

        try:
            resp = requests.get(url, headers={"x-wis-token": token}, timeout=REQUEST_TIMEOUT_S)
        except Exception:
            time.sleep(compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s))
            continue

        if resp.status_code in (401, 403):
            try:
                get_token(force_refresh=True)
            except Exception:
                pass
            time.sleep(max(5.0, compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s)))
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                time.sleep(compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s))
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
                "api_imei": api_imei,
                "segment_start": seg_start_str,
                "segment_end": seg_end_str,
                "status": "saved",
                "count": len(entries),
                "data": entries,
            }

        if resp.status_code in RECOVERABLE_HTTP:
            time.sleep(compute_backoff_s(attempt, backoff_base_s, backoff_cap_s, backoff_jitter_s))
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
    panel_target=None,
) -> Dict[str, Any]:
    segments = make_segment_list(start_dt, end_dt, DEFAULT_SEGMENT_HOURS)
    tasks = [(imei, s, e) for imei in imeis for (s, e) in segments]

    progress = st.progress(0, text="Starting fetch")
    status_box = st.empty()
    show_processing_panel(
        "Processing request",
        f"Collecting location history for {len(imeis)} IMEIs from {start_dt.strftime('%d/%m/%Y %H:%M')} to {end_dt.strftime('%d/%m/%Y %H:%M')} IST.",
        active_step=2,
        target=panel_target,
    )

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

            pct = completed / len(tasks) if tasks else 1.0
            progress.progress(
                pct,
                text=f"{completed}/{len(tasks)} tasks complete | Saved {saved} | Failed {failed} | Rows {total_rows}",
            )
            status_box.markdown(
                f"""
                <div class="mis-status">
                    <div class="mis-status-title">Live processing status</div>
                    <div class="mis-status-sub">Tasks: {len(tasks)} | Completed: {completed} | Saved: {saved} | Failed: {failed} | Location rows: {total_rows}</div>
                </div>
                """,
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
                normalized_points.append((ts_ist, lat, lon))
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
        notes = []

        for idx in range(1, len(normalized_points)):
            prev_ts, prev_lat, prev_lon = normalized_points[idx - 1]
            curr_ts, curr_lat, curr_lon = normalized_points[idx]
            dist = calculate_distance(prev_lat, prev_lon, curr_lat, curr_lon)
            dt_seconds = (curr_ts - prev_ts).total_seconds()
            if dt_seconds <= 0:
                ignored_segments += 1
                continue
            speed = dist / (dt_seconds / 3600.0)
            if speed > MAX_SPEED_KMPH:
                ignored_segments += 1
                notes.append(f"Ignored {dist:.2f} km at {speed:.0f} km/h")
                continue
            total_dist += dist
            valid_segments += 1

        first = normalized_points[0]
        last = normalized_points[-1]
        comment = []
        if bad_rows:
            comment.append(f"Bad rows skipped={bad_rows}")
        if notes:
            comment.append(notes[0])

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


def build_zip(files: Dict[str, bytes]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name, payload in files.items():
            zf.writestr(file_name, payload)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def process_raw_run(uploaded_file, start_text: str, end_text: str, workers: int, min_interval_s: float, max_attempts: int) -> Tuple[Dict[str, Any], pd.DataFrame, bytes]:
    start_dt = parse_ist_datetime_slash(start_text)
    end_dt = parse_ist_datetime_slash(end_text)
    if end_dt <= start_dt:
        raise RuntimeError("End time must be after start time.")

    get_token(force_refresh=True)
    df = read_imeis_from_uploaded_csv(uploaded_file)
    imeis = df["imei"].tolist()
    cache_key = build_cache_key(imeis, start_text, end_text, workers)
    cached = get_cached_result(cache_key)

    if cached is not None:
        payload = dict(cached)
        payload["cacheUsed"] = True
        show_notice("Using in-memory raw data", "This exact raw-data request was found in the session cache, so the app skipped a fresh API run.")
    else:
        payload = fetch_run_payload(
            imeis=imeis,
            start_dt=start_dt,
            end_dt=end_dt,
            workers=int(workers),
            min_interval_s=float(min_interval_s),
            max_attempts=int(max_attempts),
            panel_target=panel_target,
        )
        payload["cacheUsed"] = False
        set_cached_result(cache_key, payload)

    preview_df = preview_summary_df(payload.get("results", []))
    flat_df = flatten_raw_results(payload.get("results", []))

    zip_bytes = build_zip(
        {
            f"raw_summary_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json": json_download_bytes(payload),
            f"raw_data_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv": csv_bytes_from_df(flat_df),
            f"raw_segments_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv": csv_bytes_from_df(preview_df),
        }
    )
    return payload, preview_df, zip_bytes


def build_operational_range_windows(start_day: dt.date, end_day: dt.date) -> List[Tuple[str, dt.datetime, dt.datetime]]:
    if end_day < start_day:
        raise RuntimeError("End date must be on or after start date.")
    windows: List[Tuple[str, dt.datetime, dt.datetime]] = []
    cur = start_day
    while cur <= end_day:
        op_start, op_end = operational_window_for_date(cur)
        windows.append((cur.strftime("%d/%m/%Y"), op_start, op_end))
        cur += dt.timedelta(days=1)
    return windows


def aggregate_operational_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["IMEI", "Distance_KM", "Valid_Segments", "Ignored_Segments", "Point_Count", "Operational_Days", "First_Location", "First_Location_Time", "Last_Location", "Last_Location_Time", "Comment"])
    grouped = detail_df.groupby("IMEI", dropna=False).agg(
        Distance_KM=("Distance_KM", "sum"),
        Valid_Segments=("Valid_Segments", "sum"),
        Ignored_Segments=("Ignored_Segments", "sum"),
        Point_Count=("Point_Count", "sum"),
        Operational_Days=("Operational_Day", "nunique"),
        First_Location_Time=("First_Location_Time", "min"),
        Last_Location_Time=("Last_Location_Time", "max"),
    ).reset_index()
    first_loc = detail_df.sort_values(["IMEI", "Operational_Day"]).groupby("IMEI", dropna=False)["First_Location"].first().reset_index(name="First_Location")
    last_loc = detail_df.sort_values(["IMEI", "Operational_Day"]).groupby("IMEI", dropna=False)["Last_Location"].last().reset_index(name="Last_Location")
    comments = detail_df.groupby("IMEI", dropna=False)["Comment"].apply(lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()}))).reset_index(name="Comment")
    grouped = grouped.merge(first_loc, on="IMEI", how="left").merge(last_loc, on="IMEI", how="left").merge(comments, on="IMEI", how="left")
    grouped["Distance_KM"] = grouped["Distance_KM"].round(2)
    grouped = grouped[["IMEI", "Distance_KM", "Valid_Segments", "Ignored_Segments", "Point_Count", "Operational_Days", "First_Location", "First_Location_Time", "Last_Location", "Last_Location_Time", "Comment"]]
    return grouped.sort_values(["Distance_KM", "IMEI"], ascending=[False, True]).reset_index(drop=True)


def fetch_operational_payload_for_window(imeis: List[str], label: str, op_start: dt.datetime, op_end: dt.datetime, workers: int, min_interval_s: float, max_attempts: int, panel_target=None) -> Dict[str, Any]:
    cache_key = build_cache_key(imeis, op_start.strftime("%d/%m/%Y %H:%M"), op_end.strftime("%d/%m/%Y %H:%M"), workers)
    cached = get_cached_result(cache_key)
    if cached is not None:
        raw_payload = dict(cached)
        raw_payload["cacheUsed"] = True
    else:
        raw_payload = fetch_run_payload(
            imeis=imeis,
            start_dt=op_start,
            end_dt=op_end,
            workers=int(workers),
            min_interval_s=float(min_interval_s),
            max_attempts=int(max_attempts),
            panel_target=panel_target,
        )
        raw_payload["cacheUsed"] = False
        set_cached_result(cache_key, raw_payload)
    for item in raw_payload.get("results", []):
        item["operational_day"] = label
    raw_payload["operationalDayLabel"] = label
    return raw_payload


def process_distance_run(uploaded_file, mode: str, workers: int, min_interval_s: float, max_attempts: int, day_input: Optional[str] = None, start_text: Optional[str] = None, end_text: Optional[str] = None, start_day_text: Optional[str] = None, end_day_text: Optional[str] = None, panel_target=None) -> Tuple[pd.DataFrame, Dict[str, Any], bytes]:
    get_token(force_refresh=True)
    imei_df = read_imeis_from_uploaded_csv(uploaded_file)
    imeis = imei_df["imei"].tolist()

    if mode == "operational":
        if not start_day_text or not end_day_text:
            raise RuntimeError("Start date and end date are required for operational-day calculation.")
        start_day = parse_ist_date_slash(start_day_text)
        end_day = parse_ist_date_slash(end_day_text)
        windows = build_operational_range_windows(start_day, end_day)
        all_results: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []
        detail_frames: List[pd.DataFrame] = []
        cache_hits = 0

        for idx, (label, op_start, op_end) in enumerate(windows, start=1):
            show_processing_panel(
                f"Calculating operational day {idx} of {len(windows)}",
                f"Fetching raw GPS data for {label} using the 03:00 to next-day 03:00 operational-day window.",
                active_step=2,
                target=panel_target,
            )
            raw_payload = fetch_operational_payload_for_window(imeis, label, op_start, op_end, workers, min_interval_s, max_attempts, panel_target=panel_target)
            if raw_payload.get("cacheUsed"):
                cache_hits += 1
            payloads.append(raw_payload)
            all_results.extend(raw_payload.get("results", []))
            detail_df = compute_distance_summary(raw_payload.get("results", []))
            detail_df.insert(0, "Operational_Day", label)
            detail_df.insert(1, "Window_Start", op_start.strftime("%d/%m/%Y %H:%M:%S"))
            detail_df.insert(2, "Window_End", op_end.strftime("%d/%m/%Y %H:%M:%S"))
            detail_frames.append(detail_df)

        detailed_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(columns=["Operational_Day", "Window_Start", "Window_End"])
        summary_df = aggregate_operational_summary(detailed_df)
        combined_payload = {
            "runAtIst": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "operational",
            "operationalStartDate": start_day.strftime("%d/%m/%Y"),
            "operationalEndDate": end_day.strftime("%d/%m/%Y"),
            "operationalDayCount": len(windows),
            "imeiCount": len(imeis),
            "cacheUsed": cache_hits > 0,
            "cacheHits": cache_hits,
            "cacheMisses": max(0, len(windows) - cache_hits),
            "totalTasks": sum(int(x.get("totalTasks", 0)) for x in payloads),
            "savedTasks": sum(int(x.get("savedTasks", 0)) for x in payloads),
            "failedTasks": sum(int(x.get("failedTasks", 0)) for x in payloads),
            "totalLocationRowsFetched": sum(int(x.get("totalLocationRowsFetched", 0)) for x in payloads),
            "results": all_results,
            "operationalPayloads": payloads,
        }
        show_processing_panel("Calculating distance", "The app is preparing both the daily operational-detail CSV and the overall summary CSV.", active_step=3, target=panel_target)
        flat_df = flatten_raw_results(all_results)
        ts = dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')
        zip_bytes = build_zip({
            f"distance_summary_{ts}.csv": csv_bytes_from_df(summary_df),
            f"distance_operational_detail_{ts}.csv": csv_bytes_from_df(detailed_df),
            f"distance_raw_data_{ts}.csv": csv_bytes_from_df(flat_df),
            f"distance_raw_payload_{ts}.json": json_download_bytes(combined_payload),
        })
        return summary_df, combined_payload, zip_bytes

    if not start_text or not end_text:
        raise RuntimeError("Selected time range is required.")
    start_dt = parse_ist_datetime_slash(start_text, with_seconds=True)
    end_dt = parse_ist_datetime_slash(end_text, with_seconds=True)
    if end_dt <= start_dt:
        raise RuntimeError("End time must be after start time.")
    cache_key = build_cache_key(imeis, start_text, end_text, workers)
    cached = get_cached_result(cache_key)
    if cached is not None:
        raw_payload = dict(cached)
        raw_payload["cacheUsed"] = True
        show_notice("Using in-memory selected-range data", "This time-range distance request was found in cache.")
    else:
        raw_payload = fetch_run_payload(
            imeis=imeis,
            start_dt=start_dt,
            end_dt=end_dt,
            workers=int(workers),
            min_interval_s=float(min_interval_s),
            max_attempts=int(max_attempts),
            panel_target=panel_target,
        )
        raw_payload["cacheUsed"] = False
        set_cached_result(cache_key, raw_payload)

    show_processing_panel("Calculating distance", "The app is now converting fetched GPS points into a distance summary with speed filtering.", active_step=3, target=panel_target)
    distance_df = compute_distance_summary(raw_payload.get("results", []))
    flat_df = flatten_raw_results(raw_payload.get("results", []))
    ts = dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')
    zip_bytes = build_zip({
        f"distance_summary_{ts}.csv": csv_bytes_from_df(distance_df),
        f"distance_raw_data_{ts}.csv": csv_bytes_from_df(flat_df),
        f"distance_raw_payload_{ts}.json": json_download_bytes(raw_payload),
    })
    return distance_df, raw_payload, zip_bytes

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="mis-hero">
        <div class="mis-badge">MoveInSync-style operational intelligence</div>
        <h1>Think Office Commute.<br>Think of Us.</h1>
        <p>After login, choose one route: download raw data for an exact date-time range, or calculate distance using either operational-day windows across a date range or a selected time range. Files are bundled as ZIP by default.</p>
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
    st.caption("Fixed internally")
    st.caption(f"BUID: {DEFAULT_BUID}")
    st.caption(f"Device Type: {DEFAULT_DEVICE_TYPE}")
    st.caption(f"Segment Hours: {DEFAULT_SEGMENT_HOURS}")
    st.caption("Token source: internal")


# ---------------- NAV ----------------
nav_page = st.session_state.get("nav_page", "home")
nav_cols = st.columns([1, 1, 1])
with nav_cols[0]:
    if st.button("Home", use_container_width=True):
        go_to("home")
with nav_cols[1]:
    if st.button("Raw data page", use_container_width=True):
        go_to("raw")
with nav_cols[2]:
    if st.button("Distance page", use_container_width=True):
        go_to("distance")

st.markdown("<div class='mis-note'>Pick a flow below. The pages are separate so the raw-data request and distance request stay independent.</div>", unsafe_allow_html=True)


# ---------------- PAGE: HOME ----------------
if nav_page == "home":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="mis-page-card">
                <h3>Download raw data</h3>
                <p>Open the raw-data page, enter a start and end IST date-time in <b>dd/mm/yyyy hh:mm</b>, fetch the raw location history, and get a ZIP download containing JSON and CSV files.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to raw data", use_container_width=True, key="goto_raw_cta"):
            go_to("raw")

    with c2:
        st.markdown(
            """
            <div class="mis-page-card">
                <h3>Calculate distance</h3>
                <p>Open the distance page and choose either <b>Operational day</b> for 3 AM to 3 AM windows across a start and end date range, or <b>Selected time range</b> for an exact hh:mm:ss window. The app fetches data, calculates distance, and packages everything as a ZIP.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to calculate distance", use_container_width=True, key="goto_distance_cta"):
            go_to("distance")


# ---------------- PAGE: RAW ----------------
elif nav_page == "raw":
    st.markdown("## Raw data download")
    show_notice("Raw-data mode", "Enter an exact IST time range in dd/mm/yyyy hh:mm. This page only fetches raw data and does not calculate distance.")

    left, right = st.columns([1.25, 1])
    with left:
        raw_file = st.file_uploader("Upload IMEI CSV", type=["csv"], key="raw_imei_file")
    with right:
        c1, c2 = st.columns(2)
        with c1:
            raw_start = st.text_input("Start IST (dd/mm/yyyy hh:mm)", value=dt.datetime.now(IST).strftime("%d/%m/%Y 00:00"), key="raw_start")
        with c2:
            raw_end = st.text_input("End IST (dd/mm/yyyy hh:mm)", value=dt.datetime.now(IST).strftime("%d/%m/%Y %H:%M"), key="raw_end")

    clicked_label = "Please wait, fetching raw data..." if st.session_state.get("processing") else "Get raw data"
    raw_clicked = st.button(clicked_label, use_container_width=True, key="run_raw")

    if raw_clicked:
        if not raw_file:
            show_notice("CSV required", "Please upload the IMEI CSV before starting the raw-data request.")
        else:
            try:
                st.session_state["processing"] = True
                show_processing_panel("Preparing raw-data fetch", "Validating the uploaded IMEIs and building the API request windows.", active_step=1)
                raw_payload, raw_preview_df, raw_zip = process_raw_run(raw_file, raw_start, raw_end, workers, min_interval_s, max_attempts)
                st.session_state["last_raw_payload"] = raw_payload
                st.session_state["last_raw_zip"] = raw_zip
                st.session_state["processing"] = False

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
                    mode_text = "Cache hit" if raw_payload.get("cacheUsed") else "Fresh API run"
                    st.markdown(f"<div class='mis-kpi'><div class='label'>Failed tasks</div><div class='value'>{failed_tasks}</div><div class='sub'>{mode_text}</div></div>", unsafe_allow_html=True)

                st.dataframe(raw_preview_df, use_container_width=True, height=420)

                zip_name = f"raw_data_bundle_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.zip"
                show_notice("ZIP bundle ready", "The app is attempting to start the ZIP download automatically. If the browser blocks it, use the backup button below.")
                auto_download_bytes(raw_zip, zip_name, "application/zip", "rawzip")
                st.download_button("Download raw-data ZIP again", data=raw_zip, file_name=zip_name, mime="application/zip", use_container_width=True)
            except Exception as exc:
                st.session_state["processing"] = False
                show_notice("Raw-data request could not complete", str(exc))


# ---------------- PAGE: DISTANCE ----------------
elif nav_page == "distance":
    st.markdown("## Distance calculation")
    show_notice("Distance mode", "Choose how the data window should be defined. Operational day uses 3 AM to 3 AM for every day between the start and end dates. Selected time range uses exact dd/mm/yyyy hh:mm:ss values.")

    distance_file = st.file_uploader("Upload IMEI CSV", type=["csv"], key="distance_imei_file")
    mode_choice = st.radio(
        "Choose distance mode",
        options=["Operational day", "Selected time range"],
        horizontal=True,
        key="distance_mode_radio",
    )

    if mode_choice == "Operational day":
        st.session_state["distance_mode"] = "operational"
        c1, c2 = st.columns(2)
        with c1:
            op_start_day = st.text_input("Operational start date (dd/mm/yyyy)", value=dt.datetime.now(IST).strftime("%d/%m/%Y"), key="op_start_day")
        with c2:
            op_end_day = st.text_input("Operational end date (dd/mm/yyyy)", value=dt.datetime.now(IST).strftime("%d/%m/%Y"), key="op_end_day")
        range_start = None
        range_end = None
    else:
        st.session_state["distance_mode"] = "selected_range"
        c1, c2 = st.columns(2)
        with c1:
            range_start = st.text_input("Start IST (dd/mm/yyyy hh:mm:ss)", value=dt.datetime.now(IST).strftime("%d/%m/%Y 00:00:00"), key="dist_start")
        with c2:
            range_end = st.text_input("End IST (dd/mm/yyyy hh:mm:ss)", value=dt.datetime.now(IST).strftime("%d/%m/%Y %H:%M:%S"), key="dist_end")
        op_start_day = None
        op_end_day = None

    calc_label = "Please wait, calculating distance..." if st.session_state.get("processing") else "Calculate distance"
    calc_clicked = st.button(calc_label, use_container_width=True, key="run_distance")

    if calc_clicked:
        if not distance_file:
            show_notice("CSV required", "Please upload the IMEI CSV before calculating distance.")
        else:
            try:
                st.session_state["processing"] = True
                show_processing_panel("Preparing distance request", "Checking inputs and getting the right raw GPS window for the selected mode.", active_step=1)
                if st.session_state["distance_mode"] == "operational":
                    distance_df, raw_payload, distance_zip = process_distance_run(
                        distance_file,
                        mode="operational",
                        workers=workers,
                        min_interval_s=min_interval_s,
                        max_attempts=max_attempts,
                        start_day_text=op_start_day,
                        end_day_text=op_end_day,
                    )
                else:
                    distance_df, raw_payload, distance_zip = process_distance_run(
                        distance_file,
                        mode="selected",
                        workers=workers,
                        min_interval_s=min_interval_s,
                        max_attempts=max_attempts,
                        start_text=range_start,
                        end_text=range_end,
                    )
                st.session_state["last_distance_df"] = distance_df.to_dict(orient="records")
                st.session_state["last_distance_zip"] = distance_zip
                st.session_state["last_raw_payload"] = raw_payload
                st.session_state["processing"] = False

                total_distance = float(distance_df["Distance_KM"].sum()) if not distance_df.empty and "Distance_KM" in distance_df.columns else 0.0
                rows_fetched = int(raw_payload.get("totalLocationRowsFetched", 0))
                failed_tasks = int(raw_payload.get("failedTasks", 0))
                imeis_count = int(raw_payload.get("imeiCount", 0))
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(f"<div class='mis-kpi'><div class='label'>IMEIs</div><div class='value'>{imeis_count}</div><div class='sub'>Devices included</div></div>", unsafe_allow_html=True)
                with k2:
                    st.markdown(f"<div class='mis-kpi'><div class='label'>Raw rows</div><div class='value'>{rows_fetched}</div><div class='sub'>Used for distance</div></div>", unsafe_allow_html=True)
                with k3:
                    st.markdown(f"<div class='mis-kpi'><div class='label'>Distance</div><div class='value'>{total_distance:,.2f}</div><div class='sub'>Total kilometers</div></div>", unsafe_allow_html=True)
                with k4:
                    mode_text = "Cache hit" if raw_payload.get("cacheUsed") else "Fresh API run"
                    st.markdown(f"<div class='mis-kpi'><div class='label'>Failed tasks</div><div class='value'>{failed_tasks}</div><div class='sub'>{mode_text}</div></div>", unsafe_allow_html=True)

                st.dataframe(distance_df, use_container_width=True, height=420)
                zip_name = f"distance_bundle_{dt.datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.zip"
                show_notice("ZIP bundle ready", "The app is attempting to start the ZIP download automatically. If the browser blocks it, use the backup button below. Operational-day mode includes both the summary CSV and a day-by-day detailed CSV for each IMEI.")
                auto_download_bytes(distance_zip, zip_name, "application/zip", "distzip")
                st.download_button("Download distance ZIP again", data=distance_zip, file_name=zip_name, mime="application/zip", use_container_width=True)
            except Exception as exc:
                st.session_state["processing"] = False
                show_notice("Distance calculation could not complete", str(exc))
