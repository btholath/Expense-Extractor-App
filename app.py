# app.py — AI Receipt Processor (Gemini) with Auth0 SSO
# Robust totals: prefer model totals when present; otherwise compute from items.
# Observability: Prometheus metrics, JSON logs, optional OpenTelemetry spans.

import io
import os
import re
import json
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import google.generativeai as genai

# Auth0 SSO
from streamlit_oauth import OAuth2Component

# Observability (idempotent via session guard; ensure observability.py is the idempotent version)
from observability import boot
if "OBS_INIT" not in st.session_state:
    st.session_state["OBS_INIT"] = boot()
LOG, TRACER, METRICS = st.session_state["OBS_INIT"]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit / API configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧾 Receipt Tracker (Gemini)",
                   page_icon="🧾", layout="wide")
st.title("🧾 AI-Powered Receipt Tracker")

# Load Google API key (secrets preferred; env fallback)
api_key = None
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("No Google API key found. Add it to `.streamlit/secrets.toml` "
             "([general]\nGOOGLE_API_KEY=\"...\") or set env var GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=api_key)

# ─────────────────────────────────────────────────────────────────────────────
# Model selection (Gemini 2.x aware) & generation config
# ─────────────────────────────────────────────────────────────────────────────
def pick_supported_model() -> str:
    """
    Return a fully-qualified model name that supports generateContent.
    Prefers 2.5-flash → 2.5-pro → 2.0-flash → latest aliases → any gemini.
    """
    try:
        ms = list(genai.list_models())
    except Exception as e:
        raise RuntimeError(f"Could not list models. Update SDK? {e}")

    supported = [m.name for m in ms
                 if "generateContent" in getattr(m, "supported_generation_methods", [])]

    prefs = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
    ]
    for p in prefs:
        for name in supported:
            if name == p or p in name:
                return name

    for name in supported:
        if "models/gemini" in name:
            return name

    raise RuntimeError("No Gemini model with generateContent support found.")

try:
    MODEL_NAME = pick_supported_model()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

GEN_CFG = {
    "response_mime_type": "application/json",
    "temperature": 0.2,
}
SAFETY = None  # receipts are low risk

# ─────────────────────────────────────────────────────────────────────────────
# Prompt & helpers
# ─────────────────────────────────────────────────────────────────────────────
RECEIPT_PROMPT = """
You are an expert in processing grocery receipts.
This image may be only part of a long receipt (top/middle/bottom).

Extract:
1) purchase_date: "YYYY-MM-DD" or null
2) total_amount: number or null (FINAL total paid)
3) line_items: array of objects: {description (short), quantity (integer, default 1), price (number)}

Important formatting rules:
- Return ONLY valid JSON in exactly this structure:
{
  "purchase_date": "YYYY-MM-DD" or null,
  "total_amount": 0.00 or null,
  "line_items": [
    {"description": "Item", "quantity": 1, "price": 0.00}
  ]
}
- If a printed number includes a currency symbol, still return just the numeric value.
"""

def image_to_part(pil_img: Image.Image, fmt: str = "PNG") -> Dict[str, Any]:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return {"mime_type": f"image/{fmt.lower()}", "data": buf.getvalue()}

_money_pattern = re.compile(r'[-+]?\d[\d,]*\.?\d*')

def coerce_money(val):
    """Parse 12.34, '12.34', '$12.34', 'USD 12,345.67' → float; None if not found."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = _money_pattern.search(str(val))
    if not m:
        return None
    return float(m.group(0).replace(',', ''))

def coerce_int(val, default=1):
    try:
        x = int(float(val))
        return x if x > 0 else default
    except Exception:
        return default

def call_gemini_on_image(img: Image.Image, prompt: str) -> Dict[str, Any]:
    """Call Gemini with an image + prompt, parse JSON, and return a dict (with metrics & tracing)."""
    model = genai.GenerativeModel(MODEL_NAME)
    parts = [image_to_part(img), {"text": prompt}]

    t0 = time.perf_counter()
    outcome = "ok"
    span_ctx = TRACER.start_as_current_span("gemini_generate_content") if TRACER else nullcontext()

    try:
        with span_ctx:
            resp = model.generate_content(parts, generation_config=GEN_CFG, safety_settings=SAFETY)
    except Exception as e:
        outcome = "error"
        LOG.error("gemini_call_failed", error=str(e), model=MODEL_NAME)
        # Helpful message if a specific 404 occurs
        if "404" in str(e) and "is not found" in str(e):
            raise RuntimeError(
                f"Model '{MODEL_NAME}' not found in your account. "
                "Use the debug expander to list supported models, "
                "or upgrade the google-generativeai SDK."
            )
        raise ValueError(f"Gemini call failed: {e}") from e
    finally:
        METRICS["hists"]["latency"].observe(time.perf_counter() - t0)
        METRICS["counters"]["calls"].labels(outcome=outcome, model=MODEL_NAME).inc()

    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            raise ValueError("Empty response from model.")

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except Exception as e:
        LOG.warning("model_invalid_json", raw=cleaned[:4000])
        raise ValueError(f"Model did not return valid JSON.\nRaw:\n{cleaned}\n\nParse error: {e}")

    if "line_items" in data and not isinstance(data["line_items"], list):
        data["line_items"] = []

    return data

def normalize_items(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """Clean + deduplicate line items across images."""
    if not items:
        return pd.DataFrame(columns=["description", "quantity", "price"])
    df = pd.DataFrame(items)
    df["description"] = df.get("description", "").astype(str).str.strip().str.lower()
    df["quantity"]   = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1).astype(int)
    df["price"]      = pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0).round(2)
    return df.drop_duplicates(subset=["description", "price"], keep="first").reset_index(drop=True)

def compute_estimated_total(df: pd.DataFrame) -> float:
    """Fallback total from items (max of sum(price) and sum(price*quantity))."""
    if df.empty:
        return 0.0
    sum_line = float(df["price"].sum())
    sum_qty  = float((df["price"] * df["quantity"]).sum())
    return round(max(sum_line, sum_qty), 2)

def choose_final_total(seen_totals: List[float], est_total: float) -> Tuple[float, str]:
    """Pick best total and label its source."""
    best_reported = max(seen_totals) if seen_totals else 0.0
    if est_total > best_reported:
        return est_total, "estimated_from_items"
    if best_reported > 0:
        return round(best_reported, 2), "reported_by_model"
    return 0.0, "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# Auth0 SSO Configuration
# ─────────────────────────────────────────────────────────────────────────────
try:
    AUTH0_CLIENT_ID = st.secrets["AUTH0_CLIENT_ID"]
    AUTH0_CLIENT_SECRET = st.secrets["AUTH0_CLIENT_SECRET"]
    AUTH0_DOMAIN = st.secrets["AUTH0_DOMAIN"]
    REDIRECT_URI = st.secrets.get("AUTH0_REDIRECT_URI", "http://localhost:8501")
    AUTHORIZE_ENDPOINT = f"https://{AUTH0_DOMAIN}/authorize"
    TOKEN_ENDPOINT = f"https://{AUTH0_DOMAIN}/oauth/token"
    REVOKE_ENDPOINT = f"https://{AUTH0_DOMAIN}/oauth/revoke"
except KeyError:
    st.error("Auth0 secrets not found! Add AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN "
             "to your .streamlit/secrets.toml (and optional AUTH0_REDIRECT_URI).")
    st.stop()

oauth2 = OAuth2Component(
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_ENDPOINT,
    token_endpoint=TOKEN_ENDPOINT,
    refresh_token_endpoint=TOKEN_ENDPOINT,
    revoke_token_endpoint=REVOKE_ENDPOINT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main UI - Gated by SSO Login
# ─────────────────────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None

if st.session_state.token is None:
    st.write("Welcome to the AI Receipt Tracker!")
    st.write("Please log in to continue.")
    result = oauth2.authorize_button(
        name="Login with Auth0",
        icon="https://auth0.com/favicon.ico",
        redirect_uri=REDIRECT_URI,  # Adjust for production
        scope="openid profile email",
    )
    if result and "token" in result:
        st.session_state.token = result.get("token")
        st.rerun()
else:
    # Sidebar user info + logout
    with st.sidebar:
        user_info = st.session_state.token.get("userinfo", {})
        display_name = user_info.get("name") or user_info.get("email") or "User"
        st.write(f"Welcome, **{display_name}**")
        if user_info.get("email"):
            st.caption(user_info["email"])
        if st.button("Logout"):
            st.session_state.token = None
            st.rerun()

    # ── MAIN APPLICATION UI ──────────────────────────────────────────────────
    with st.expander("Model / Debug"):
        st.write("Using model:", MODEL_NAME)
        if st.button("List supported models"):
            try:
                ms = [m.name for m in genai.list_models()
                      if "generateContent" in getattr(m, "supported_generation_methods", [])]
                st.write(ms)
            except Exception as e:
                st.warning(f"ListModels error: {e}")

    uploaded_files = st.file_uploader(
        "Upload receipt image(s) (JPG/PNG)…",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Preview / save uploads
    SAVE_DIR = "uploaded_receipts"
    if uploaded_files:
        os.makedirs(SAVE_DIR, exist_ok=True)
        st.subheader("Preview")
        ncols = min(3, len(uploaded_files))
        cols = st.columns(ncols)
        for i, f in enumerate(uploaded_files):
            try:
                img = Image.open(f).convert("RGB")
                cols[i % ncols].image(img, caption=f.name, width="stretch")
                img.save(os.path.join(SAVE_DIR, f.name))
                METRICS["counters"]["images"].labels(status="accepted").inc()
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")
                METRICS["counters"]["images"].labels(status="rejected").inc()
        st.success(f"Saved {len(uploaded_files)} image(s) to `{SAVE_DIR}`.")

    # Process
    if uploaded_files and st.button("Process All Receipts", type="primary"):
        all_items: List[Dict[str, Any]] = []
        final_date: str = "N/A"
        seen_totals: List[float] = []

        with st.spinner("Reading your receipt(s)…"):
            for f in uploaded_files:
                st.write(f"Processing **{f.name}** …")
                img = Image.open(f).convert("RGB")

                try:
                    data = call_gemini_on_image(img, RECEIPT_PROMPT)
                except RuntimeError as e:
                    st.error(f"Could not process {f.name}: {e}", icon="🚨")
                    st.info(
                        "Tip: Use the *Model / Debug* expander to list models your account supports. "
                        "If none of the 2.x models appear, upgrade the SDK:\n\n"
                        "```bash\npip install -U google-generativeai google-ai-generativelanguage\n```"
                    )
                    continue
                except ValueError as e:
                    st.warning(f"Skipping {f.name}: {e}")
                    continue

                # Optional: inspect raw JSON from model
                if st.checkbox(f"Show raw JSON for {f.name}", key=f"raw_{f.name}"):
                    st.code(json.dumps(data, indent=2), language="json")

                # Aggregate items (sanitize)
                items = data.get("line_items") or []
                for it in items:
                    desc = str(it.get("description", "")).strip()
                    qty = coerce_int(it.get("quantity", 1), default=1)
                    price = coerce_money(it.get("price"))
                    if desc and price is not None:
                        all_items.append(
                            {"description": desc, "quantity": qty, "price": price}
                        )

                # Date: last non-empty wins
                if data.get("purchase_date"):
                    final_date = data["purchase_date"]

                # Reported total: collect numeric candidates
                t = coerce_money(data.get("total_amount"))
                if t is not None:
                    seen_totals.append(t)

        # Deduplicate items BEFORE estimating total
        df = normalize_items(all_items)
        est_total = compute_estimated_total(df)
        final_total, total_source = choose_final_total(seen_totals, est_total)

        # Observability
        METRICS["gauges"]["total"].set(final_total)
        LOG.info(
            "receipt_aggregate_done",
            images=len(uploaded_files),
            purchase_date=final_date,
            total_final=final_total,
            total_source=total_source,
            totals_seen=seen_totals,
        )

        st.success("Done.")

        st.subheader("Consolidated Information")
        c1, c2 = st.columns(2)
        c1.metric("Purchase Date", final_date)
        c2.metric("Total Amount", f"${final_total:.2f}")
        st.caption(
            f"Total source: **{total_source}** "
            f"{'(model reported)' if total_source=='reported_by_model' else '(computed from items)'}"
        )

        st.write("All Purchased Items (duplicates removed):")
        st.dataframe(df, width="stretch")

        # Save / download CSV
        if not df.empty or final_total > 0:
            out = df.copy() if not df.empty else pd.DataFrame(columns=["description", "quantity", "price"])
            out["purchase_date"] = final_date
            out["receipt_total_reported"] = max(seen_totals) if seen_totals else 0.0
            out["receipt_total_estimated"] = est_total
            out["receipt_total_final"] = final_total
            out["total_source"] = total_source

            csv_path = "receipts_data.csv"
            if os.path.exists(csv_path):
                try:
                    prev = pd.read_csv(csv_path)
                    out_all = pd.concat([prev, out], ignore_index=True)
                except Exception:
                    out_all = out
            else:
                out_all = out

            out_all.to_csv(csv_path, index=False)
            st.info(f"Saved consolidated data to `{csv_path}`.")

            st.download_button(
                "⬇️ Download CSV",
                out_all.to_csv(index=False).encode("utf-8"),
                file_name="all_receipts.csv",
                mime="text/csv",
            )
