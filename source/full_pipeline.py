"""
Stock Short Interest Prediction Pipeline - Refactored

Clean implementation with clear separation of concerns:
1. Data Acquisition
2. Feature Engineering
3. Model Training & Evaluation
4. Visualization

Key improvements:
- Removed _BoosterWrapper complexity
- Unified lagging strategy
- Simplified feature validation
- Clear train/test splits
- Better error handling
"""

import argparse
import base64
import importlib.util
import io
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================
load_dotenv()

# API Configuration
ALPHA_API = os.getenv('ALPHA_API')
CLAUDE_API = os.getenv('CLAUDE_API')
FINRA_CLIENT_ID = os.getenv('FINRA_CLIENT_ID')
FINRA_CLIENT_PASS = os.getenv('FINRA_CLIENT_PASS')
CLAUDE_MODEL = 'claude-sonnet-4-20250514'
CLAUDE_MAX_TOKENS = 52000

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH_BBG = os.path.join(SCRIPT_DIR, '..', 'data', 'BloombergJPM_noformula.xlsx')
IN_IMAGES_DIR = "in_images"
IN_TABULAR_DIR = "in_tabular"
os.makedirs(IN_IMAGES_DIR, exist_ok=True)
os.makedirs(IN_TABULAR_DIR, exist_ok=True)

# Model Configuration
DEFAULT_LAG_DAYS = 14  # Look-ahead bias prevention
TEST_SIZE = 0.2
N_CV_SPLITS = 5
n_passes = 2


# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================
def get_finra_auth() -> str:
    """Authenticate with FINRA API."""
    auth = HTTPBasicAuth(FINRA_CLIENT_ID, FINRA_CLIENT_PASS)
    response = requests.post(
        url="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token",
        data={'grant_type': 'client_credentials'},
        auth=auth
    )
    response.raise_for_status()
    return response.json()['access_token']


def get_finra_short_data(symbol: str, date_range: List[Optional[str]] = [None, None]) -> pd.DataFrame:
    """Fetch and process short interest data from FINRA."""
    token = get_finra_auth()
    payload = {
        'limit': 1000,
        'compareFilters': [{
            'compareType': 'EQUAL',
            'fieldName': 'symbolCode',
            'fieldValue': symbol.upper()
        }]
    }

    if date_range[0] or date_range[1]:
        date_filter = {'fieldName': 'settlementDate'}
        if date_range[0]:
            date_filter['startDate'] = date_range[0]
        if date_range[1]:
            date_filter['endDate'] = date_range[1]
        payload['dateRangeFilters'] = [date_filter]

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Accept': 'text/plain'
    }

    response = requests.post(
        'https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest',
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), sep=",", engine="python", keep_default_na=False)

    if df.empty:
        raise ValueError(f"No short interest data found for {symbol}")

    df = df[['settlementDate', 'currentShortPositionQuantity']].copy()
    df.columns = ['date', 'short_interest']
    df['date'] = pd.to_datetime(df['date'])
    df['short_interest'] = df['short_interest'].astype(float)
    df = df.sort_values('date').set_index('date')

    # Save visualization
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['short_interest'], marker='o')
    plt.title(f"{symbol} - Short Interest (FINRA)")
    plt.xlabel("Date")
    plt.ylabel("Short Interest")
    plt.grid(alpha=0.3)
    plt.savefig(f"outputs/{symbol}_short_interest.png", dpi=150, bbox_inches='tight')
    plt.close()

    return df


def parse_bloomberg_excel(file_path: str, start_row: int = 1, start_col: int = 2) -> Dict[str, pd.DataFrame]:
    """Parse Bloomberg Excel with multiple tickers - flexible column handling."""
    try:
        df = pd.read_excel(file_path, header=None)
    except:
        df = pd.read_csv(file_path, header=None)

    data_section = df.iloc[start_row:, start_col:]
    expected_columns = [
        'Date', 'PX_LAST', 'PX_VOLUME', 'DVD_SH_LAST',
        'PX_BID', 'PX_ASK', 'OPEN_INT_TOTAL_PUT', 'OPEN_INT_TOTAL_CALL'
    ]

    # Find ticker positions
    first_row = data_section.iloc[0].fillna('').astype(str)
    ticker_positions = [(cell.strip(), idx) for idx, cell in enumerate(first_row)
                        if 'Equity' in cell or 'Index' in cell]

    if not ticker_positions:
        raise ValueError("No tickers found in Excel file")

    result_dict = {}
    for i, (ticker_name, col_start) in enumerate(ticker_positions):
        # Determine actual number of columns for this ticker
        if i + 1 < len(ticker_positions):
            col_end = ticker_positions[i + 1][1]
        else:
            col_end = len(data_section.columns)

        n_cols = col_end - col_start
        ticker_data = data_section.iloc[:, col_start:col_end]
        ticker_values = ticker_data.iloc[2:].copy()

        # Use actual column names or generate generic ones
        if n_cols == len(expected_columns):
            ticker_values.columns = expected_columns
        else:
            # Use first n columns or pad with generic names
            actual_cols = expected_columns[:n_cols] if n_cols <= len(expected_columns) else \
                expected_columns + [f'Col_{j}' for j in range(len(expected_columns), n_cols)]
            ticker_values.columns = actual_cols

        # Clean data
        mask = ticker_values.isin(['#N/A N/A', '#N/A', 'N/A', '', '#N/A Field Not Applicable'])
        ticker_values = ticker_values.mask(mask, np.nan)

        if 'Date' not in ticker_values.columns:
            print(f" ⚠ Skipping {ticker_name}: No Date column found")
            continue

        ticker_values = ticker_values[ticker_values['Date'].notna()]
        if ticker_values.empty:
            continue

        # Convert dates
        date_col = pd.to_numeric(ticker_values['Date'], errors='coerce')
        ticker_values['Date'] = pd.to_datetime(date_col, origin='1899-12-30', unit='D', errors='coerce')
        ticker_values = ticker_values[ticker_values['Date'].notna()]

        # Convert numeric columns
        numeric_columns = [col for col in ticker_values.columns if col != 'Date']
        for col in numeric_columns:
            ticker_values[col] = pd.to_numeric(ticker_values[col], errors='coerce')

        ticker_values = ticker_values.set_index('Date').sort_index()
        result_dict[ticker_name] = ticker_values

    return result_dict


def get_stock_data(ticker: str, file_path: str = FILE_PATH_BBG) -> pd.DataFrame:
    """Extract specific ticker from Bloomberg file."""
    all_data = parse_bloomberg_excel(file_path)

    # Exact match
    if ticker in all_data:
        return all_data[ticker].dropna(how='all')

    # Partial match
    matching = [t for t in all_data.keys() if ticker in t]
    if len(matching) == 1:
        return all_data[matching[0]].dropna(how='all')
    elif len(matching) > 1:
        raise ValueError(f"Multiple tickers match '{ticker}': {matching}")
    else:
        raise ValueError(f"Ticker '{ticker}' not found. Available: {list(all_data.keys())}")


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
def create_daily_master_df(stock_dict: Dict[str, pd.DataFrame],
                           spx_df: Optional[pd.DataFrame] = None,
                           vix_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Combine all daily data sources into single DataFrame."""
    all_dates = pd.DatetimeIndex([])
    for df in stock_dict.values():
        all_dates = all_dates.union(df.index)
    if spx_df is not None:
        all_dates = all_dates.union(spx_df.index)
    if vix_df is not None:
        all_dates = all_dates.union(vix_df.index)

    all_dates = all_dates.sort_values()
    master_df = pd.DataFrame(index=all_dates)

    # Add stock data with prefixes
    for stock, df in stock_dict.items():
        aligned = df.reindex(master_df.index).ffill(limit=5)  # 5-day stale limit
        aligned = aligned.add_prefix(f"{stock}_")
        master_df = master_df.join(aligned, how='left')

    # Add indices
    if spx_df is not None:
        aligned = spx_df.reindex(master_df.index).ffill(limit=5).add_prefix("SPX_")
        master_df = master_df.join(aligned, how='left')

    if vix_df is not None:
        aligned = vix_df.reindex(master_df.index).ffill(limit=5).add_prefix("VIX_")
        master_df = master_df.join(aligned, how='left')

    master_df = master_df.dropna(how='all', axis=1)
    return master_df


def align_to_targets_with_lag(daily_df: pd.DataFrame,
                               target_dates: pd.DatetimeIndex,
                               lag_days: int = DEFAULT_LAG_DAYS) -> pd.DataFrame:
    """
    Align daily features to target dates with lag to prevent lookahead bias.
    For each target date T, use data from T - lag_days.
    """
    aligned_rows = []
    for target_date in target_dates:
        cutoff_date = target_date - pd.Timedelta(days=lag_days)
        historical_data = daily_df[daily_df.index <= cutoff_date]

        if len(historical_data) > 0:
            last_row = historical_data.iloc[-1:].copy()
            last_row.index = [target_date]
            aligned_rows.append(last_row)
        else:
            # No data available before this date
            empty_row = pd.DataFrame(index=[target_date], columns=daily_df.columns)
            aligned_rows.append(empty_row)

    return pd.concat(aligned_rows, axis=0)


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

# ---- Compact dataset summary (text, not images) ----------------------------
def _summarize_dataset(daily_df: pd.DataFrame, max_cols: int = 200) -> str:
    """
    Build a compact JSON string with per-column metadata that the LLM can use
    to design robust, leakage-safe daily features without relying on images.
    """
    cols = list(daily_df.columns[:max_cols])
    summary = {
        "index_type": str(type(daily_df.index).__name__),
        "n_rows": int(len(daily_df)),
        "n_cols": int(daily_df.shape[1]),
        "columns": [],
        "notes": [
            "Columns are prefixed with their source ticker/index (e.g., AAPL_PX_LAST).",
            "Daily frequency with missing days possible; prefer robust rolling/EW ops with min_periods.",
            "Pipeline aligns features to biweekly targets with a 14-day lag to prevent lookahead."
        ],
    }

    for c in cols:
        s = daily_df[c]
        n = int(s.notna().sum())
        nan_ratio = float(s.isna().mean())
        dtype = str(s.dtype)

        if n >= 30:
            p01 = float(s.quantile(0.01)) if n > 0 else None
            p50 = float(s.median()) if n > 0 else None
            p99 = float(s.quantile(0.99)) if n > 0 else None
            stdev_63 = float(s.rolling(63, min_periods=20).std().median()) if n > 0 else None
            summary["columns"].append({
                "name": c, "dtype": dtype, "n_non_nan": n, "nan_ratio": round(nan_ratio, 4),
                "p01": p01, "p50": p50, "p99": p99, "stdev_63": stdev_63
            })
        else:
            summary["columns"].append({
                "name": c, "dtype": dtype, "n_non_nan": n, "nan_ratio": round(nan_ratio, 4)
            })

    return json.dumps(summary, ensure_ascii=False, indent=2)


# ---- Anthropic payload (system + user text only) ---------------------------
def _build_claude_payload(
    daily_df: pd.DataFrame,
    short_stock: str,
    max_number_of_features: int = 15,
    last_error: str = "",
    feedback_md: str = "",
    must_keep: Optional[List[str]] = None,
    avoid: Optional[List[str]] = None,
    orth_corr_target: float = 0.6,
) -> dict:
    """
    Build a strict system+user payload with a plan-first requirement and a
    hardened code-only contract for feature_* functions.
    """
    schema_text = _summarize_dataset(daily_df, max_cols=200)

    system_constraints = f"""
You are generating leakage-safe DAILY features to help predict the biweekly short interest for {short_stock}.
Output one Python module ONLY with:
- An initial single-line JSON comment manifest beginning with: # PLAN: {{"features":[...]}}
- A set of functions named feature_* that each accept df: pandas.DataFrame (daily master; DateTimeIndex; prefixed columns) and return a Series or DataFrame indexed daily.

STRICT REQUIREMENTS:
- Use only pandas and numpy; no other imports; no file or network I/O.
- Daily-only construction with rolling/EW transforms; set min_periods to avoid NaN cascades.
- No target-aware transforms; no references to short interest labels or their lags/derivatives.
- Keep total NEW columns ≤ {max_number_of_features} with simple ASCII names.
- Do not redefine anything listed in MUST_KEEP; do not recreate items listed in AVOID.
- Prefer low rolling Spearman correlation to MUST_KEEP by design (aim |rho| ≤ {orth_corr_target} over 63-day windows).
- Favor robust families relevant to short interest dynamics: price/volume momentum, EW volatility, spread/liquidity proxies, options OI ratios/asymmetry, SPX/VIX-relative terms.
- Do not use identifiers containing: short_interest, si_lag, si_logdiff, target, label; use neutral names like x_, feat_, f_ instead

MODULE STRUCTURE:
- Top line: '# PLAN: {{...}}' describing feature names, family, windows, inputs, expected_sign, and rationale.
- Below: only feature_* function definitions implementing exactly the PLAN.
"""

    # User content with summary + critique
    user_parts = []
    user_parts.append("MASTER_SUMMARY_JSON:\n" + schema_text)

    if feedback_md:
        user_parts.append("FEEDBACK_MD (top/weak/redundancy insights from last pass):\n" + feedback_md)

    if must_keep:
        user_parts.append(
            "MUST_KEEP (do not redefine):\n" + "\n".join(map(str, must_keep[:50]))
        )
    if avoid:
        user_parts.append(
            "AVOID (do not recreate):\n" + "\n".join(map(str, avoid[:50]))
    )

    if last_error:
        user_parts.append("LAST_ERROR (import/validation/runtime issues last pass):\n" + last_error)

    # Final instruction line to ensure code-only output inside a single fenced block
    user_parts.append(
        "Emit code ONLY inside one fenced Python block. No prose. No tests. No prints. "
        "No top-level execution. Define helpers inline if needed."
    )

    content_text = "\n\n".join(user_parts)

    return {
        "model": CLAUDE_MODEL,
        "system": system_constraints.strip(),
        "max_tokens": min(CLAUDE_MAX_TOKENS, 4000),
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": content_text}]}
        ],
    }


# ---- Response parsing and static leakage guard -----------------------------
FORBIDDEN_TOKENS = [
    "short_interest", "si_lag", "si_logdiff", "target", "label"#, "y_", "y[", "y."
]


def static_leakage_guard(code: str) -> Tuple[bool, str]:
    """Reject modules that reference target-like tokens before import."""
    lower = code.lower()
    for tok in FORBIDDEN_TOKENS:
        if tok in lower:
            return False, f"Static leakage guard: forbidden token '{tok}' found in generated code."
    return True, ""


def parse_response(response: str, out_file: str) -> Optional[str]:
    """
    Extract the single fenced Python code block and write it to out_file.
    Enforces presence of the PLAN manifest header.
    """
    # Prefer ```python
    start = response.find("```python")
    if start != -1:
        end = response.find("```", start + 9)
        code_block = response[start + 9:end].strip() if end != -1 else ""
    else:
        # Fallback to first generic code fence
        start = response.find("```")
        end = response.find("```", start + 3) if start != -1 else -1
        code_block = response[start + 3:end].strip() if end != -1 else ""

    if not code_block:
        print("No code block found in Claude response.")
        return None, "no_code_block"

    # Require a PLAN manifest line
    if not re.search(r"^#\s*PLAN:\s*\{.*\}\s*$", code_block.splitlines()[0].strip()):
        print("PLAN manifest missing or malformed in first line.")
        return None, "no_plan"

    ok, msg = static_leakage_guard(code_block)
    if not ok:
        print(msg)
        return None, f"leak_guard:{msg}"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(code_block)
    print(f"Saved {out_file}")
    return code_block, ""


def _call_claude_and_write_generated_features(payload: dict, out_file: str) -> Optional[str]:
    client = anthropic.Anthropic(api_key=CLAUDE_API)
    full = ""
    try:
        with client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full += text
    except Exception as e:
        print(f"Claude API error: {e}")
        return None, f"claude_error:{e}"

    return parse_response(full, out_file)


# ---- Generation entrypoint (signature preserved) ---------------------------
def generate_llm_features(
    daily_df: pd.DataFrame,
    short_df: pd.DataFrame,
    stock_list: List[str],
    short_stock: str,
    max_features: int = 8,
    last_error: str = "",
    feedback_md: str = "",
    must_keep: Optional[List[str]] = None,
    avoid: Optional[List[str]] = None,
    out_file: str = "generated_features.py",
) -> bool:
    """
    Generate feature module with strict system contract and dataset summary
    (no images), writing deterministically to out_file for the current pass.
    """
    payload = _build_claude_payload(
        daily_df=daily_df,
        short_stock=short_stock,
        max_number_of_features=max_features,
        last_error=last_error or "",
        feedback_md=feedback_md or "",
        must_keep=must_keep or [],
        avoid=avoid or [],
        orth_corr_target=0.6,
    )

    code, err = _call_claude_and_write_generated_features(payload, out_file)
    return (code is not None), (err or "")


# ---- Hardened validation and safe execution --------------------------------
def _max_consecutive_unchanged(series: pd.Series) -> int:
    """Compute the longest run of unchanged consecutive values (ignoring NaN)."""
    s = series.copy()
    s = s.where(~s.isna(), np.nan)

    # Identify changed points
    changed = s != s.shift(1)

    # Run-length encode
    run_ids = changed.cumsum().fillna(0)

    # Count lengths per run id among non-nan
    lengths = run_ids.groupby(run_ids).transform("size") if len(run_ids) else pd.Series([], dtype=int)

    # But only where s is not NaN
    lengths = lengths.where(~s.isna(), 0)

    return int(lengths.max() if len(lengths) else 0)


def validate_feature(
    feature_df: pd.DataFrame,
    feature_name: str,
    min_valid_ratio: float = 0.6,
    max_constant_ratio: float = 0.90,
    max_unchanged_ratio: float = 0.80,
) -> Tuple[bool, str]:
    """
    Stricter feature validation:
    - Rejects excessive NaNs
    - Rejects near-constants
    - Rejects long stretches of unchanged values (indicative of over-ffill/static artifacts)
    """
    if feature_df is None or feature_df.empty:
        return False, f"{feature_name}: returned empty DataFrame"

    if not isinstance(feature_df.index, pd.DatetimeIndex):
        try:
            feature_df.index = pd.to_datetime(feature_df.index)
        except Exception as e:
            return False, f"{feature_name}: invalid date index - {e}"

    for col in feature_df.columns:
        series = feature_df[col].copy()

        # Replace infinites
        if np.isinf(series).any():
            series = series.replace([np.inf, -np.inf], np.nan)
            feature_df[col] = series

        # NaN check
        nan_ratio = float(series.isna().mean())
        if nan_ratio > (1 - min_valid_ratio):
            return False, f"{feature_name}[{col}]: {nan_ratio:.1%} NaN (threshold: {(1-min_valid_ratio):.1%})"

        non_nan = series.dropna()
        if len(non_nan) == 0:
            return False, f"{feature_name}[{col}]: all NaN values"

        # Near-constant check
        if non_nan.nunique() == 1:
            return False, f"{feature_name}[{col}]: constant value"

        most_common_ratio = float(non_nan.value_counts().iloc[0] / len(non_nan))
        if most_common_ratio > max_constant_ratio:
            return False, f"{feature_name}[{col}]: {most_common_ratio:.1%} identical values"

        # Long unchanged runs check
        mcur = _max_consecutive_unchanged(non_nan)
        if mcur >= 5:
            unchanged_ratio = float(mcur / max(1, len(non_nan)))
            if unchanged_ratio > max_unchanged_ratio:
                return False, f"{feature_name}[{col}]: {unchanged_ratio:.1%} longest unchanged run"

    return True, ""


def safe_execute_features(
    daily_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    feature_files: List[str],
    lag_days: int = DEFAULT_LAG_DAYS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Import each generated module, execute feature_* functions, validate outputs,
    and align the union to target dates with the specified lag.
    """
    feature_dfs, error_log = [], []

    for feature_file in feature_files:
        if not os.path.exists(feature_file):
            error_log.append(f"{feature_file} not found")
            continue

        mod_name = os.path.splitext(os.path.basename(feature_file))[0]
        spec = importlib.util.spec_from_file_location(mod_name, feature_file)
        module = importlib.util.module_from_spec(spec)

        # Ensure clean import on repeated passes
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            error_log.append(f"Failed to import {feature_file}: {e}")
            continue

        funcs = [
            getattr(module, name)
            for name in dir(module)
            if callable(getattr(module, name)) and name.startswith("feature_")
        ]

        print(f"\n{feature_file}: found {len(funcs)} feature functions")

        for func in funcs:
            try:
                res = func(daily_df)
                if isinstance(res, pd.Series):
                    res = res.to_frame()

                ok, msg = validate_feature(res, func.__name__)
                if not ok:
                    print(f" ✗ {func.__name__}: {msg}")
                    error_log.append(f"{feature_file}:{func.__name__}: {msg}")
                    continue

                print(f" ✓ {func.__name__}: {len(res.columns)} column(s)")
                feature_dfs.append(res)
            except Exception as e:
                error_log.append(f"{feature_file}:{func.__name__}: {e}")

    if not feature_dfs:
        return pd.DataFrame(), error_log

    all_features_daily = pd.concat(feature_dfs, axis=1)
    all_features_daily = all_features_daily.loc[:, ~all_features_daily.columns.duplicated()]

    features_aligned = align_to_targets_with_lag(all_features_daily, target_dates, lag_days)
    print(f"\nFeatures aligned to targets (union): {features_aligned.shape}")

    return features_aligned, error_log


def create_full_feature_set(
    daily_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    lag_days: int = DEFAULT_LAG_DAYS,
    feature_files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Combine baseline raw features (lagged alignment of master) with the union of
    validated LLM features, de-duplicated by name.
    """
    raw_features = align_to_targets_with_lag(daily_df, target_dates, lag_days)
    engineered_features = pd.DataFrame()

    if feature_files:
        eng, errors = safe_execute_features(
            daily_df, target_dates, feature_files=feature_files, lag_days=lag_days
        )

        if errors:
            print(f"\n⚠ {len(errors)} generated features rejected")
            for err in errors:
                print(f" - {err}")

        engineered_features = eng

    combined = pd.concat([raw_features, engineered_features], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


# ---- Feedback summarization (interface preserved) --------------------------
def summarize_feedback_for_llm(
    gain_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    X_train: pd.DataFrame,
    top_k: int = 15,
    corr_thresh: float = 0.9,
) -> Tuple[str, List[str], List[str]]:
    """
    Summarize top/weak features and redundancy clusters to guide the next pass.
    This keeps the previous interface but the narrative strongly encourages
    orthogonality and family diversity in the next prompt.
    """
    top_by_gain = gain_df.head(top_k)["feature"].tolist()
    top_by_perm = perm_df.head(top_k)["feature"].tolist()
    top_union = list(dict.fromkeys(top_by_gain + top_by_perm))
    weak = perm_df.tail(max(5, min(15, len(perm_df)//4)))["feature"].tolist()

    Xf = X_train[[c for c in top_union if c in X_train.columns]].ffill().bfill().fillna(0)
    groups = []

    if Xf.shape[1] > 1:
        cm = Xf.corr(method="spearman").abs()
        used = set()
        for c in cm.columns:
            if c in used:
                continue
            grp = [c]
            used.add(c)
            for c2 in cm.columns:
                if c2 in used:
                    continue
                if cm.loc[c, c2] >= corr_thresh:
                    grp.append(c2)
                    used.add(c2)
            if len(grp) > 1:
                groups.append(grp)

    md = []
    md.append("Top features by gain/perm (union):")
    for f in top_union[:top_k]:
        g = float(gain_df[gain_df.feature == f]["gain_norm"].values[0]) if f in set(gain_df.feature) else 0.0
        p = float(perm_df[perm_df.feature == f]["perm_r2_drop"].values[0]) if f in set(perm_df.feature) else 0.0
        md.append(f"- {f} | gain_norm={g:.4f} | perm_r2_drop={p:.4f}")

    if weak:
        md.append("\nWeak/low-utility features candidates for removal/replacement:")
        for f in weak[:15]:
            md.append(f"- {f}")

    if groups:
        md.append("\nRedundancy groups (high Spearman |rho|):")
        for grp in groups[:8]:
            md.append("- " + ", ".join(grp))

    # MUST_KEEP biases next pass toward strong features; AVOID prunes weak ones
    feedback_md = "\n".join(md)
    must_keep_raw = top_union[: min(12, len(top_union))]
    avoid_raw = weak[: min(10, len(weak))]
    must_keep = [str(x) for x in must_keep_raw]
    avoid = [str(x) for x in avoid_raw]
    return feedback_md, must_keep, avoid



# ============================================================================
# 4. FEATURE PRUNING
# ============================================================================
def prune_features_smart(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         nan_thresh: float = 0.60,
                         min_unique: int = 4,
                         corr_thresh: float = 0.92,
                         max_features: int = 120) -> List[str]:
    """
    Intelligent feature selection based on:
    1. Missing value ratio
    2. Uniqueness
    3. Correlation with target
    4. Multicollinearity
    """
    # Step 1: Drop high-NaN columns
    valid_cols = X_train.columns[X_train.isna().mean() < nan_thresh]
    X = X_train[valid_cols]

    # Step 2: Drop near-constant columns
    valid_cols = [c for c in X.columns if X[c].nunique(dropna=True) >= min_unique]
    X = X[valid_cols]

    if len(X.columns) == 0:
        return []

    # Step 3: Rank by Spearman correlation with target
    y_log = np.log1p(y_train)
    scores = []
    for col in X.columns:
        try:
            mask = ~(X[col].isna() | y_log.isna())
            if mask.sum() < 8:
                score = 0.0
            else:
                rho, _ = spearmanr(X.loc[mask, col], y_log[mask])
                score = abs(rho) if not np.isnan(rho) else 0.0
        except:
            score = 0.0
        scores.append((col, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_cols = [col for col, _ in scores[:max_features]]
    X_ranked = X[top_cols]

    # Step 4: Remove highly correlated features
    X_filled = X_ranked.ffill().bfill().fillna(0)
    corr_matrix = X_filled.corr().abs()
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper_tri)
    drop_cols = [col for col in upper_corr.columns if any(upper_corr[col] > corr_thresh)]
    final_cols = [c for c in X_ranked.columns if c not in drop_cols]

    return final_cols[:max_features]


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
def _sealed_cut_index(target_idx: pd.DatetimeIndex, test_size: float = TEST_SIZE) -> int:
    n_total = len(target_idx)
    n_test = max(int(n_total * test_size), 5)
    return max(n_total - n_test, 1)


def _prep_block(X: pd.DataFrame) -> pd.DataFrame:
    Xp = X.ffill().bfill()
    return Xp.fillna(Xp.median())


def _split_by_cut(features_df: pd.DataFrame,
                  short_df: pd.DataFrame,
                  cut_idx: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Timestamp]:
    data = features_df.join(short_df['short_interest'], how='inner').sort_index()

    # Past-only target features (safe)
    data['si_lag1'] = data['short_interest'].shift(1)
    data['si_lag2'] = data['short_interest'].shift(2)

    # Replace leakage feature with a lagged delta that depends only on the past
    data['si_logdiff_lag1'] = (
        np.log1p(data['short_interest'].shift(1)) - np.log1p(data['short_interest'].shift(2))
    )

    train_data = data.iloc[:cut_idx]
    test_data = data.iloc[cut_idx:]

    X_train = train_data.drop(columns=['short_interest'])
    y_train = train_data['short_interest']
    X_test = test_data.drop(columns=['short_interest'])
    y_test = test_data['short_interest']

    cutoff_date = train_data.index[-1]

    return X_train, y_train, X_test, y_test, cutoff_date


def _build_features(
    daily_df: pd.DataFrame,
    target_idx: pd.DatetimeIndex,
    files: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    # Baseline raw block (lag-aligned)
    raw = align_to_targets_with_lag(daily_df, target_idx, DEFAULT_LAG_DAYS)

    if files:
        eng, errors = safe_execute_features(
            daily_df, target_idx, feature_files=files, lag_days=DEFAULT_LAG_DAYS
        )
        combined = pd.concat([raw, eng], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined, errors

    return raw, []


def _train_on_block(X: pd.DataFrame, y: pd.Series, n_trials: int):
    model, best_params, kept_cols, medians = train_xgboost_with_optuna(X, y, n_trials=n_trials)

    val_size = max(3, int(0.2 * len(X)))
    X_tr, X_dev = X.iloc[:-val_size], X.iloc[-val_size:]
    y_tr, y_dev = y.iloc[:-val_size], y.iloc[-val_size:]

    m = evaluate_model(model, X_tr, y_tr, X_dev, y_dev, kept_cols, medians)
    return model, best_params, kept_cols, medians, float(m["test_r2"]), m


def compute_medians(X: pd.DataFrame) -> pd.Series:
    Xf = X.ffill().bfill()
    return Xf.median()


def prep_with_medians(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    Xf = X.ffill().bfill()

    # Ensure the exact training feature order and columns
    missing_cols = [c for c in med.index if c not in Xf.columns]
    if missing_cols:
        for c in missing_cols:
            Xf[c] = np.nan

    # Drop any columns not seen at training
    Xf = Xf.reindex(columns=med.index)
    return Xf.fillna(med)


def train_xgboost_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_splits: int = N_CV_SPLITS
) -> Tuple[xgb.Booster, Dict, List[str], Optional[pd.Series]]:
    y_train_scaled = y_train

    # Expanding-window CV
    min_train_size = int(len(X_train) * 0.6)
    step = max(1, (len(X_train) - min_train_size) // n_splits)
    cv_splits = []

    for i in range(n_splits):
        end = min(min_train_size + i * step, len(X_train) - 1)
        train_idx = np.arange(0, end)
        val_idx = np.arange(end, min(end + step, len(X_train)))
        if len(val_idx) > 0:
            cv_splits.append((train_idx, val_idx))

    print(f"\nCreated {len(cv_splits)} expanding window CV splits")

    # Prune BEFORE Optuna
    kept_cols = prune_features_smart(
        X_train, y_train,
        nan_thresh=0.40,
        max_features=40
    )

    # Keep safe target lags if present
    must_keep = [c for c in X_train.columns if c in ['si_lag1', 'si_lag2', 'si_logdiff_lag1']]
    kept_cols = list(dict.fromkeys(must_keep + kept_cols))

    print(f"\nSelected {len(kept_cols)} features before optimization")
    X_train_pruned = X_train[kept_cols]

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "seed": 42,
            "eta": trial.suggest_float("eta", 0.03, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0.1, 1.0),
            "alpha": trial.suggest_float("alpha", 0.0, 3.0),
            "lambda": trial.suggest_float("lambda", 0.5, 10.0),
        }

        n_estimators = trial.suggest_int("n_estimators", 300, 1200)
        rmses = []

        for train_idx, val_idx in cv_splits:
            X_tr = X_train_pruned.iloc[train_idx]
            y_tr = y_train_scaled.iloc[train_idx]
            X_val = X_train_pruned.iloc[val_idx]
            y_val = y_train_scaled.iloc[val_idx]

            med = compute_medians(X_tr)
            dtrain = xgb.DMatrix(prep_with_medians(X_tr, med), label=y_tr)
            dval = xgb.DMatrix(prep_with_medians(X_val, med), label=y_val)

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "validation")],
                early_stopping_rounds=100,
                verbose_eval=False
            )

            y_pred = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
            rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
            rmses.append(rmse)

        return float(np.mean(rmses))

    print("\nRunning Optuna hyperparameter optimization...")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # Final fit with early stop on last 20% of train
    val_size = max(3, int(0.2 * len(X_train_pruned)))
    X_tr_final = X_train_pruned.iloc[:-val_size]
    y_tr_final = y_train_scaled.iloc[:-val_size]
    X_val_final = X_train_pruned.iloc[-val_size:]
    y_val_final = y_train_scaled.iloc[-val_size:]

    med_final = compute_medians(X_tr_final)
    dtrain = xgb.DMatrix(prep_with_medians(X_tr_final, med_final), label=y_tr_final)
    dval = xgb.DMatrix(prep_with_medians(X_val_final, med_final), label=y_val_final)

    final_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": 42,
        "eta": best_params["eta"],
        "max_depth": best_params["max_depth"],
        "min_child_weight": best_params["min_child_weight"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "gamma": best_params["gamma"],
        "alpha": best_params["alpha"],
        "lambda": best_params["lambda"],
    }

    final_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=best_params["n_estimators"],
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=150,
        verbose_eval=50
    )

    # Return medians so downstream eval uses the exact same preprocessing
    return final_model, best_params, kept_cols, med_final


def evaluate_model(model: xgb.Booster,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   kept_cols: List[str],
                   medians: Optional[pd.Series]) -> Dict:
    X_train_pruned = X_train[kept_cols]
    X_test_pruned = X_test.reindex(columns=kept_cols)

    med = medians if medians is not None else compute_medians(X_train_pruned)

    dtrain = xgb.DMatrix(prep_with_medians(X_train_pruned, med))
    dtest = xgb.DMatrix(prep_with_medians(X_test_pruned, med))

    y_train_pred = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
    y_test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"IN-SAMPLE (Train): RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
    print(f"OUT-OF-SAMPLE (Test): RMSE={test_rmse:.4f}, R²={test_r2:.4f}")
    print("="*60 + "\n")

    return {
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "train_pred": y_train_pred,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_pred": y_test_pred
    }


def compute_feature_importances(
    model: xgb.Booster,
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    kept_cols: List[str],
    n_repeats: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Normalize gain importance
    gain_raw = model.get_score(importance_type="gain")

    # Map booster feature names (f0, f1, ...) to actual names if needed
    # XGBoost keeps DataFrame column names in DMatrix; if absent, fallback to kept_cols order.
    col_map = {f: f for f in X_ref.columns}
    gain = []
    for k, v in gain_raw.items():
        name = col_map.get(k, k)
        gain.append((name, float(v)))

    gain_df = pd.DataFrame(gain, columns=["feature", "gain"]).groupby("feature", as_index=False)["gain"].sum()
    if gain_df["gain"].sum() > 0:
        gain_df["gain_norm"] = gain_df["gain"] / gain_df["gain"].sum()
    else:
        gain_df["gain_norm"] = 0.0

    # Permutation importance on a small holdout (last 20% of ref)
    m = max(3, int(0.2 * len(X_ref)))
    Xh, yh = X_ref.iloc[-m:].copy(), y_ref.iloc[-m:].copy()
    base = r2_score(yh, model.predict(xgb.DMatrix(_prep_block(Xh)), iteration_range=(0, model.best_iteration + 1)))

    drops = []
    rng = np.random.default_rng(42)
    for feat in kept_cols:
        if feat not in Xh.columns:
            continue
        drop_vals = []
        for _ in range(n_repeats):
            Xp = Xh.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            r2p = r2_score(yh, model.predict(xgb.DMatrix(_prep_block(Xp)), iteration_range=(0, model.best_iteration + 1)))
            drop_vals.append(base - r2p)
        drops.append((feat, float(np.nanmean(drop_vals))))

    perm_df = pd.DataFrame(drops, columns=["feature", "perm_r2_drop"]).sort_values("perm_r2_drop", ascending=False)

    return gain_df.sort_values("gain_norm", ascending=False), perm_df


# ============================================================================
# 6. VISUALIZATION
# ============================================================================
def plot_predictions(y_train: pd.Series,
                     y_train_pred: np.ndarray,
                     y_test: pd.Series,
                     y_test_pred: np.ndarray,
                     cutoff_date: pd.Timestamp,
                     symbol: str,
                     metrics: Dict,
                     save_path: str = "outputs/predictions.png"):
    """Create comprehensive prediction visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Time series plot
    ax1 = axes[0, 0]
    ax1.plot(y_train.index, y_train.values, 'o-', color='blue', alpha=0.6,
             label='Actual (In-Sample)', markersize=6, linewidth=2)
    ax1.plot(y_train.index, y_train_pred, 's--', color='lightblue', alpha=0.8,
             label='Predicted (In-Sample)', markersize=5, linewidth=1.5)
    ax1.plot(y_test.index, y_test.values, 'o-', color='red', alpha=0.8,
             label='Actual (Out-of-Sample)', markersize=8, linewidth=2)
    ax1.plot(y_test.index, y_test_pred, 's--', color='orange', alpha=0.9,
             label='Predicted (Out-of-Sample)', markersize=7, linewidth=2)
    ax1.axvline(cutoff_date, color='green', linestyle='--', linewidth=2.5,
                alpha=0.7, label=f'Train/Test Split')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Short Interest', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - Predictions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. In-sample scatter
    ax2 = axes[0, 1]
    ax2.scatter(y_train.values, y_train_pred, alpha=0.6, s=80,
                edgecolors='darkblue', linewidth=1.5, color='lightblue')
    min_val = min(y_train.values.min(), y_train_pred.min())
    max_val = max(y_train.values.max(), y_train_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Short Interest', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Short Interest', fontsize=12, fontweight='bold')
    ax2.set_title(f'In-Sample\nRMSE: {metrics["train_rmse"]:.4f}, R²: {metrics["train_r2"]:.4f}',
                  fontsize=13, fontweight='bold', color='blue')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Out-of-sample scatter
    ax3 = axes[1, 0]
    ax3.scatter(y_test.values, y_test_pred, alpha=0.8, s=120,
                edgecolors='darkred', linewidth=2, color='orange')
    min_val = min(y_test.values.min(), y_test_pred.min())
    max_val = max(y_test.values.max(), y_test_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
    ax3.set_xlabel('Actual Short Interest', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Short Interest', fontsize=12, fontweight='bold')
    ax3.set_title(f'Out-of-Sample (HELD-OUT TEST)\nRMSE: {metrics["test_rmse"]:.4f}, R²: {metrics["test_r2"]:.4f}',
                  fontsize=13, fontweight='bold', color='red')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Residuals
    ax4 = axes[1, 1]
    train_residuals = y_train.values - y_train_pred
    test_residuals = y_test.values - y_test_pred
    bp = ax4.boxplot([train_residuals, test_residuals],
                      positions=[1, 2],
                      widths=0.6,
                      patch_artist=True,
                      tick_labels=['In-Sample', 'Out-of-Sample'])
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('orange')
    ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax4.set_title('Prediction Residuals', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================
def run_pipeline(
    stock_list: List[str],
    short_stock: str,
    use_llm: bool = True,                 # LLM-first by default
    n_trials: int = 15,
    skip_initial_baseline: bool = True,   # keep baseline disabled
    n_passes: int = 3,                     
    retry_on_leakage: bool = True,       
    max_leakage_retries: int = 3        
):
    """
    Execute the short interest prediction pipeline in an LLM-first manner.

    This version supports an arbitrary number of LLM feature-generation passes.
    Each pass may add features based on feedback from the previous pass.
    """

    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")

    print("=" * 60)
    print(f"SHORT INTEREST PREDICTION PIPELINE (LLM {n_passes} pass{'es' if n_passes > 1 else ''})")
    print("=" * 60)

    # 1) Target data
    print(f"\n1. Loading short interest data for {short_stock}...")
    short_df = get_finra_short_data(short_stock)
    target_idx = short_df.index
    print(f" Found {len(short_df)} biweekly measurements")

    # 2) Daily sources
    print(f"\n2. Loading daily stock data...")
    stock_dict = {}
    for stock in tqdm(stock_list, desc="Tickers"):
        try:
            stock_dict[stock] = get_stock_data(stock)
        except Exception as e:
            print(f" ⚠ Skipping {stock}: {e}")
    if not stock_dict:
        raise RuntimeError("No daily stock data loaded — cannot proceed.")

    # Build unified daily master
    daily_master = create_daily_master_df(stock_dict=stock_dict)

    # 3) LLM passes
    if not use_llm or not skip_initial_baseline:
        # Strongly recommend LLM-first without baseline; keep guard to avoid accidental baseline runs
        raise RuntimeError("This configuration expects LLM-first. Set use_llm=True and skip_initial_baseline=True.")

    os.makedirs("generated", exist_ok=True)
    feature_files: List[str] = []

    # Feedback state carried between passes
    last_error = ""
    feedback_md = ""
    must_keep = None
    avoid = None

    model = None
    kept_cols = []
    medians = None
    metrics = None
    cutoff_date = None

    for p in range(1, n_passes + 1):
        out_file = os.path.join("generated", f"{short_stock}_features_pass{p}.py")

        attempts = max_leakage_retries if retry_on_leakage else 1
        att = 0
        ok = False
        err = ""

        while att < attempts:
            att += 1
            print(f"\n3.{p}.{att} Generating LLM features (attempt {att}/{attempts})...")
            ok, err = generate_llm_features(
                daily_df=daily_master,
                short_df=short_df,
                stock_list=stock_list,
                short_stock=short_stock,
                max_features=8,
                last_error=last_error,
                feedback_md=feedback_md,
                must_keep=must_keep,
                avoid=avoid,
                out_file=out_file,
            )

            if ok and os.path.exists(out_file):
                break

            # If the guard tripped, warn and tighten guidance, then retry
            if isinstance(err, str) and err.startswith("leak_guard:"):
                print(f"Warning: Static leakage guard triggered on pass {p}, attempt {att}: {err}")
                last_error = (
                    "Previous attempt tripped the leakage guard; "
                    "do not reference or name anything with short_interest, si_lag, si_logdiff, target, label; "
                    "use neutral identifiers like feat_*, f_*, x_*; daily-only transforms with min_periods; "
                    "no access to labels or their lags."
                )
                continue

            # Non-guard failure: break early to surface the error
            break

        if not (ok and os.path.exists(out_file)):
            raise RuntimeError(
                f"LLM feature generation failed at pass {p} after {att} attempt(s) — aborting."
            )

        feature_files.append(out_file)

        print(f"\n4.{p}. Building aligned features (raw + LLM union) for pass {p}...")
        features_df = create_full_feature_set(
            daily_df=daily_master,
            target_dates=target_idx,
            lag_days=DEFAULT_LAG_DAYS,
            feature_files=feature_files,
        )
        if features_df.empty or features_df.isna().all(axis=None):
            raise RuntimeError(f"Aligned feature matrix is empty/NaN after pass {p}.")

        cut_idx = _sealed_cut_index(target_idx, test_size=TEST_SIZE)
        X_train, y_train, X_test, y_test, cutoff_date = _split_by_cut(features_df, short_df, cut_idx)

        print(f"\n5.{p}. Training model on pass {p} feature set...")
        model, best_params, kept_cols, medians, dev_r2, dev_metrics = _train_on_block(X_train, y_train, n_trials=n_trials)

        print(f"\n6.{p}. Held-out evaluation for pass {p}...")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test, kept_cols, medians)

        # Prepare feedback for next pass unless this is the last pass
        if p < n_passes:
            try:
                print(f"\n7.{p}. Computing importances and summarizing feedback for next pass...")
                gain_df, perm_df = compute_feature_importances(model, X_train[kept_cols], y_train, kept_cols)
                feedback_md, must_keep, avoid = summarize_feedback_for_llm(gain_df, perm_df, X_train)
                last_error = ""
            except Exception as e:
                last_error = str(e)
                # Fallback: proceed to next pass without feedback
                feedback_md, must_keep, avoid = "", None, None

        # Optional: per-pass plots
        val_size = max(3, int(0.2 * len(X_train)))
        y_train_for_plot = y_train.iloc[:-val_size]

        # Use the aligned targets for the in-sample plot
        plot_predictions(
            y_train_for_plot,                # aligned with dev_metrics["train_pred"]
            dev_metrics["train_pred"],       # predictions for X_tr only
            y_test,
            metrics["test_pred"],            # predictions align with full X_test
            cutoff_date,
            short_stock,
            metrics,
            save_path=f"outputs/{short_stock}_predictions_pass{p}.png"
        )

    print("\nDone.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Short Interest Prediction Pipeline')
    parser.add_argument('--use-llm', action='store_true', help='Enable LLM feature generation')
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials (default: 100)')
    args = parser.parse_args()

    # User inputs
    stock_list = input("Enter comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL): ").split(',')
    stock_list = [s.strip().upper() for s in stock_list]
    short_stock = input("Enter stock to predict short interest for (e.g., TSLA): ").strip().upper()

    # Run pipeline
    run_pipeline(stock_list, short_stock, use_llm=args.use_llm, n_trials=args.n_trials)
