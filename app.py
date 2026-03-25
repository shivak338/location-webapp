import io
import re
import json
import time
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
DEFAULT_WORKERS = 6
DEFAULT_MIN_INTERVAL_S = 0.25
TOKEN_REFRESH_INTERVAL_SECONDS = 600
REQUEST_TIMEOUT_S = 60
RESULT_CACHE_TTL_SECONDS = 900  # 15 minutes

# ---------------- PAGE ----------------
st.set_page_config(page_title="Location JSON Fetcher", page_icon="📍", layout="wide")
st.title("📍 Location JSON Fetcher")

# ---------------- AUTH ----------------
def check_password() -> bool:
    if "APP_PASSWORD" not in st.secrets:
        st.error("Missing APP_PASSWORD in secrets.")
        return False

    if st.session_state.get("authenticated", False):
        return True

    with st.form("login_form", clear_on_submit=False):
        pwd = st.text_input("Enter access password", type="password")
        submitted = st.form_submit_button("Unlock")

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

def build_cache_key(imeis: List[str], start_str: str, end_str: str) -> str:
    payload = {
        "imeis": sorted([str(x).strip() for x in imeis]),
        "start": start_str.strip(),
        "end": end_str.strip(),
        "buid": DEFAULT_BUID,
        "device_type": DEFAULT_DEVICE_TYPE,
        "segment_hours": DEFAULT_SEGMENT_HOURS,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
        _result_cache[cache_key] = {
            "ts": time.time(),
            "data": data,
        }

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

# ---------------- UI ----------------
with st.sidebar:
    st.header("Settings")
    workers = st.number_input("Workers", min_value=1, max_value=20, value=DEFAULT_WORKERS)
    min_interval_s = st.number_input(
        "Min interval between requests (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_MIN_INTERVAL_S,
        step=0.05,
    )
    max_attempts = st.number_input("Max attempts per segment (0 = retry forever)", min_value=0, max_value=100, value=5)
    st.caption("BUID, token URL, segment hours, and device type are fixed internally.")

uploaded_file = st.file_uploader("Upload IMEI CSV", type=["csv"])

c1, c2 = st.columns(2)
with c1:
    start_str = st.text_input("Start IST (DD-MM-YYYY HH:MM)", value="01-02-2026 00:00")
with c2:
    end_str = st.text_input("End IST (DD-MM-YYYY HH:MM)", value="01-02-2026 23:59")

run_clicked = st.button("Fetch location data", type="primary", use_container_width=True)

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

        # initial token check
        token = get_token(force_refresh=True)

        df = read_imeis_from_uploaded_csv(uploaded_file)
        imeis = df["imei"].tolist()
        segments = make_segment_list(start_ist, end_ist, DEFAULT_SEGMENT_HOURS)
        tasks = [(imei, s, e) for imei in imeis for (s, e) in segments]

        cache_key = build_cache_key(imeis, start_str, end_str)
        cached_summary = get_cached_result(cache_key)

        if cached_summary is not None:
            st.success("Using cached result from last 15 minutes.")
            summary = cached_summary
            cache_used = True
        else:
            st.success(f"Initial token fetch successful. Token length={len(token)}")
            st.info(f"IMEIs: {len(imeis)} | Segments per IMEI: {len(segments)} | Total tasks: {len(tasks)}")

            progress = st.progress(0, text="Starting...")
            status_box = st.empty()

            results: List[Dict[str, Any]] = []
            completed = 0
            saved = 0
            failed = 0
            total_records = 0

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
                    r = fut.result()
                    results.append(r)
                    completed += 1
                    if r["status"] == "saved":
                        saved += 1
                        total_records += int(r.get("count", 0))
                    else:
                        failed += 1

                    progress.progress(
                        completed / len(tasks),
                        text=f"Completed {completed}/{len(tasks)} | Saved={saved} Failed={failed}"
                    )

                    status_box.markdown(
                        f"""
**Live summary**
- Total tasks: {len(tasks)}
- Completed: {completed}
- Saved: {saved}
- Failed: {failed}
- Total location rows fetched: {total_records}
"""
                    )

            summary = {
                "runAtIst": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                "cacheTtlSeconds": RESULT_CACHE_TTL_SECONDS,
                "cacheUsed": False,
                "buid": DEFAULT_BUID,
                "deviceType": DEFAULT_DEVICE_TYPE,
                "segmentHours": DEFAULT_SEGMENT_HOURS,
                "startInput": start_str,
                "endInput": end_str,
                "imeiCount": len(imeis),
                "totalTasks": len(tasks),
                "savedTasks": saved,
                "failedTasks": failed,
                "totalLocationRowsFetched": total_records,
                "results": results,
            }

            set_cached_result(cache_key, summary)
            cache_used = False

        summary = dict(summary)
        summary["cacheUsed"] = cache_used

        json_bytes = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        zip_buffer.seek(0)

        st.subheader("Final summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IMEIs", summary.get("imeiCount", 0))
        m2.metric("Tasks saved", summary.get("savedTasks", 0))
        m3.metric("Tasks failed", summary.get("failedTasks", 0))
        m4.metric("Rows fetched", summary.get("totalLocationRowsFetched", 0))

        if summary.get("cacheUsed"):
            st.info("Result source: cache (valid for 15 minutes)")
        else:
            st.info("Result source: fresh API fetch")

        preview_rows = []
        for x in summary.get("results", [])[:200]:
            preview_rows.append({
                "imei": x.get("imei"),
                "segment_start": x.get("segment_start"),
                "segment_end": x.get("segment_end"),
                "status": x.get("status"),
                "count": x.get("count", 0),
                "error": x.get("error", ""),
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

        ts = dt.datetime.now(IST).strftime("%Y%m%d_%H%M%S")

        st.download_button(
            "Download full JSON",
            data=json_bytes,
            file_name=f"location_results_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.download_button(
            "Download ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"location_results_{ts}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")