#!/usr/bin/env python3
"""
Unified pipeline using Ksu246/nolbert-classifier with safe chunking
- Prompts for: ticker, company name, start, end
- FINRA short interest via API (prompts for creds if missing)
- WRDS transcripts filtered by ticker OR company name in text
- Sentiment via Hugging Face model Ksu246/nolbert-classifier (3-class)
- Derives features and saves {TICKER}_features.csv in CWD
"""

import io
import os
import sys
from typing import Optional, List

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# ----------------------
# Lazy imports for heavy libs
# ----------------------
def _lazy_import_wrds():
    import importlib
    return importlib.import_module("wrds")

def _lazy_import_torch():
    import importlib
    return importlib.import_module("torch")


# ----------------------
# Inputs
# ----------------------
load_dotenv()

def ask_inputs():
    ticker = input("Enter ticker (e.g., AAPL): ").strip().upper()
    company = input("Enter company name (e.g., APPLE) to search transcripts: ").strip().upper()
    start  = input("Enter start date (YYYY-MM-DD): ").strip()
    end    = input("Enter end date (YYYY-MM-DD): ").strip()
    if not ticker or not start or not end:
        print("Ticker, start, and end are required.")
        sys.exit(2)
    return ticker, company, start, end


# ----------------------
# FINRA (Short Interest)
# ----------------------
def _get_finra_creds():
    cid = os.getenv("FINRA_CLIENT_ID") or ""
    cpass = os.getenv("FINRA_CLIENT_PASS") or ""
    if cid and cpass:
        return cid, cpass
    try:
        import getpass
        if not cid:
            cid = input("FINRA_CLIENT_ID not set. Enter FINRA client id: ").strip()
        if not cpass:
            cpass = getpass.getpass("FINRA_CLIENT_PASS not set. Enter FINRA client secret: ")
    except Exception:
        if not cid:
            cid = input("FINRA_CLIENT_ID: ").strip()
        if not cpass:
            cpass = input("FINRA_CLIENT_PASS: ").strip()
    return cid, cpass

def _finra_token(cid: str, cpass: str) -> str:
    auth = HTTPBasicAuth(cid, cpass)
    r = requests.post(
        "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token",
        data={"grant_type":"client_credentials"},
        auth=auth, timeout=30
    )
    r.raise_for_status()
    return r.json()["access_token"]

def fetch_finra_short_interest(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    cid, cpass = _get_finra_creds()
    if not cid or not cpass:
        raise RuntimeError("FINRA credentials are required.")
    token = _finra_token(cid, cpass)
    payload = {
        "limit": 1000,
        "compareFilters": [{
            "compareType": "EQUAL",
            "fieldName": "symbolCode",
            "fieldValue": symbol.upper()
        }]
    }
    if start or end:
        dr = {"fieldName": "settlementDate"}
        if start: dr["startDate"] = start
        if end:   dr["endDate"] = end
        payload["dateRangeFilters"] = [dr]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/plain"
    }
    r = requests.post(
        "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest",
        headers=headers, json=payload, timeout=60
    )
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise ValueError(f"No FINRA short interest for {symbol}")
    df = df[["settlementDate","currentShortPositionQuantity"]].copy()
    df.columns = ["date","short_interest"]
    df["date"] = pd.to_datetime(df["date"])
    df["short_interest"] = pd.to_numeric(df["short_interest"], errors="coerce")
    df = df.dropna(subset=["short_interest"]).sort_values("date").set_index("date")
    return df


# ----------------------
# WRDS transcripts (ticker OR company name in transcript text)
# ----------------------
WRDS_USERNAME = os.getenv("WRDS_USERNAME")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD")
WRDS_SCHEMA   = os.getenv("WRDS_TRANSCRIPTS_SCHEMA")
WRDS_TABLE    = os.getenv("WRDS_TRANSCRIPTS_TABLE")
WRDS_SQL      = os.getenv("WRDS_TRANSCRIPTS_SQL")  # optional override

def fetch_wrds_transcripts(symbol: str, company: str, start: str, end: str) -> pd.DataFrame:
    if not WRDS_USERNAME or not WRDS_PASSWORD:
        raise RuntimeError("WRDS credentials missing. Set WRDS_USERNAME and WRDS_PASSWORD")
    wrds = _lazy_import_wrds()
    db = wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)

    company_pattern = f"%{company.upper()}%" if company else None

    if WRDS_SQL:
        params = {"ticker": symbol, "start": start, "end": end}
        if "%(company_pattern)s" in WRDS_SQL and company_pattern:
            params["company_pattern"] = company_pattern
        df = db.raw_sql(WRDS_SQL, params=params)
    else:
        if not (WRDS_SCHEMA and WRDS_TABLE):
            raise RuntimeError("Set WRDS_TRANSCRIPTS_SCHEMA and WRDS_TRANSCRIPTS_TABLE or provide WRDS_TRANSCRIPTS_SQL")
        sql = f"""
            SELECT
                event_dt::date AS date,
                ticker,
                transcript AS text
            FROM {WRDS_SCHEMA}.{WRDS_TABLE}
            WHERE event_dt::date >= %(start)s::date
              AND event_dt::date <= %(end)s::date
              AND (
                    UPPER(ticker) = %(ticker)s
                    {" OR UPPER(transcript) LIKE %(company_pattern)s" if company_pattern else ""}
                  )
        """
        params = {"ticker": symbol.upper(), "start": start, "end": end}
        if company_pattern:
            params["company_pattern"] = company_pattern
        df = db.raw_sql(sql, params=params)

    if df.empty:
        raise ValueError(f"No WRDS transcripts for {symbol} / {company} in range")
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower() or c.lower().endswith("_dt"):
                df["date"] = pd.to_datetime(df[c])
                break
    else:
        df["date"] = pd.to_datetime(df["date"])
    if "text" not in df.columns:
        for c in df.columns:
            if "text" in c.lower() or "transcript" in c.lower():
                df["text"] = df[c].astype(str)
                break
    if "ticker" not in df.columns:
        df["ticker"] = symbol.upper()

    df = df[["date","ticker","text"]].sort_values("date").reset_index(drop=True)
    return df


# ----------------------
# NoLBERT HF model (Ksu246/nolbert-classifier) with safe chunking
# ----------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Load model + tokenizer
MODEL_ID = "Ksu246/nolbert-classifier"   # 3-class
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.sep_token or tok.cls_token

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
torch = _lazy_import_torch()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# dynamic label map (robust if model label order differs)
id2label = model.config.id2label
label_names = [id2label[i] for i in range(len(id2label))]
# normalize to NEUTRAL/POSITIVE/NEGATIVE column names
def to_std_cols(cols):
    mapped = {}
    for i, c in enumerate(cols):
        up = c.upper()
        if "NEU" in up: mapped[c] = "NEUTRAL"
        elif "POS" in up: mapped[c] = "POSITIVE"
        elif "NEG" in up: mapped[c] = "NEGATIVE"
        else: mapped[c] = c
    return mapped

# 2) Hard-token chunker (IDs never exceed 512 after adding specials)
def token_windows(input_ids, max_len=512, stride=64):
    """Slide over token IDs with overlap, add special tokens per chunk, ensure len <= max_len."""
    core_max = max_len - tok.num_special_tokens_to_add(pair=False)
    start = 0
    n = len(input_ids)
    chunks = []
    while start < n:
        end = min(start + core_max, n)
        core = input_ids[start:end]
        with_special = tok.build_inputs_with_special_tokens(core)
        attn = [1]*len(with_special)
        chunks.append((with_special, attn))
        if end == n:
            break
        start = max(end - stride, 0)
    return chunks

@torch.no_grad()
def score_text_safe(text: str, max_len=512, stride=64):
    if not isinstance(text, str) or not text.strip():
        return {"NEUTRAL": np.nan, "POSITIVE": np.nan, "NEGATIVE": np.nan}
    enc = tok(text, add_special_tokens=False)
    ids = enc.get("input_ids", [])
    if not ids:
        return {"NEUTRAL": np.nan, "POSITIVE": np.nan, "NEGATIVE": np.nan}

    chunks = token_windows(ids, max_len=max_len, stride=stride)

    # batch the chunks
    input_ids_list = [torch.tensor(c[0], dtype=torch.long) for c in chunks]
    attn_list      = [torch.tensor(c[1], dtype=torch.long) for c in chunks]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tok.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list,      batch_first=True, padding_value=0)

    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [num_chunks, num_labels]

    weights = attention_mask.sum(dim=1).cpu().numpy()
    w = np.maximum(weights, 1.0).astype(float)
    wp = (probs * w[:, None]).sum(axis=0) / w.sum()

    # map to dynamic labels -> standardized columns
    prob_df = pd.DataFrame([wp], columns=label_names)
    prob_df = prob_df.rename(columns=to_std_cols(prob_df.columns))
    # ensure all three exist
    for col in ["NEUTRAL", "POSITIVE", "NEGATIVE"]:
        if col not in prob_df.columns:
            prob_df[col] = np.nan
    return {k: float(prob_df[k].iloc[0]) for k in ["NEUTRAL","POSITIVE","NEGATIVE"]}

def score_texts_to_df(texts, max_len=512, stride=64, batch=64):
    out_rows = []
    for i in range(0, len(texts), batch):
        for s in texts[i:i+batch]:
            out_rows.append(score_text_safe(s, max_len=max_len, stride=stride))
    return pd.DataFrame(out_rows)


# ----------------------
# Combine + Features
# ----------------------
def combine_and_features(short_df: pd.DataFrame,
                         transcripts_df: pd.DataFrame) -> pd.DataFrame:
    # Daily transcript sentiment via HF model
    if transcripts_df.empty:
        daily_sent = pd.DataFrame(columns=["date","sent_pos","sent_neu","sent_neg"])
    else:
        probs = score_texts_to_df(transcripts_df["text"].astype(str).tolist())
        tmp = pd.concat([transcripts_df[["date"]].reset_index(drop=True), probs], axis=1)
        # rename standardized NEUTRAL/POSITIVE/NEGATIVE -> sent_* for compatibility
        tmp = tmp.rename(columns={"POSITIVE":"sent_pos","NEUTRAL":"sent_neu","NEGATIVE":"sent_neg"})
        agg = tmp.groupby(tmp["date"].dt.date).mean(numeric_only=True).reset_index()
        agg["date"] = pd.to_datetime(agg["date"])
        daily_sent = agg

    # Build daily index from all sources (here: just short interest + sentiment)
    dates = set(short_df.index.normalize().tolist())
    if not daily_sent.empty:
        dates |= set(daily_sent["date"].dt.normalize().tolist())
    if not dates:
        raise ValueError("No overlapping dates to combine.")
    base = pd.DataFrame({"date": sorted(pd.to_datetime(list(dates)))}).set_index("date")

    # Short interest (biweekly -> ffill)
    si = short_df.copy()
    si = si[~si.index.duplicated()].sort_index()
    base = base.join(si.rename(columns={"short_interest":"short_interest"}), how="left")
    base["short_interest"] = base["short_interest"].ffill()

    # Sentiment
    if not daily_sent.empty:
        base = base.join(daily_sent.set_index("date"), how="left")

    # Extra features (compatible naming)
    if "short_interest" in base.columns:
        base["si_pct_change"] = base["short_interest"].pct_change()
    if "sent_pos" in base.columns:
        base["sent_pos_7d"] = base["sent_pos"].rolling(7, min_periods=1).mean()
    if "sent_neg" in base.columns:
        base["sent_neg_7d"] = base["sent_neg"].rolling(7, min_periods=1).mean()

    return base.reset_index().sort_values("date")


# ----------------------
# Main
# ----------------------
def main():
    ticker, company, start, end = ask_inputs()
    finra = fetch_finra_short_interest(ticker, start, end)
    transcripts = fetch_wrds_transcripts(ticker, company, start, end)
    features = combine_and_features(finra, transcripts)
    out = f"{ticker}_features.csv"
    features.to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
