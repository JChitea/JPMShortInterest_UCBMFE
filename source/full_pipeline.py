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

import os
import json
import base64
import io
from typing import Dict, List, Optional, Tuple
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anthropic
import requests
from requests.auth import HTTPBasicAuth
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm
from dotenv import load_dotenv
import optuna
import argparse
import base64
import json
import os



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
            print(f"  ⚠ Skipping {ticker_name}: No Date column found")
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


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _save_short_interest_plot(short_df: pd.DataFrame, short_stock: str) -> str:
    path = os.path.join(IN_IMAGES_DIR, f"{short_stock}_short_interest.png")
    plt.figure(figsize=(10, 6))
    plt.plot(short_df.index, short_df["short_interest"], marker="o", alpha=0.8)
    plt.title(f"{short_stock} - Short Interest")
    plt.xlabel("Date")
    plt.ylabel("Short Interest")
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def _save_master_schema_json(daily_df: pd.DataFrame, stock_list: List[str], short_stock: str) -> str:
    # Keep the prompt payload compact but informative
    schema = {
        "index_type": "DatetimeIndex",
        "shape": [int(daily_df.shape[0]), int(daily_df.shape[1])],
        "columns": list(daily_df.columns[:250]),  # cap to keep payload small
        "dtypes_sample": {c: str(daily_df[c].dtype) for c in daily_df.columns[:150]},
        "stocks_provided": stock_list,
        "target_symbol": short_stock,
        "notes": [
            "Columns are prefixed by the originating ticker/index (e.g., AAPL_PX_LAST).",
            "Daily frequency with possible missing days; use robust rolling ops.",
            "Temporal leakage must be avoided; model pipeline applies a 14-day lag alignment."
        ],
    }
    path = os.path.join(IN_TABULAR_DIR, f"{short_stock}_master_schema.json")
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    return path

def _save_sample_column_plots(daily_df: pd.DataFrame, max_plots: int = 8) -> List[str]:
    # Plot a representative subset of columns to keep the payload reasonable
    cols = []
    # Prioritize common, broad signals if present
    for key in ["PX_LAST", "PX_VOLUME", "DVD_SH_LAST", "PX_BID", "PX_ASK", "OPEN_INT_TOTAL_PUT", "OPEN_INT_TOTAL_CALL"]:
        candidates = [c for c in daily_df.columns if c.endswith(key)]
        cols.extend(candidates[:2])
    # Fill remaining slots with a few arbitrary columns
    remaining = [c for c in daily_df.columns if c not in cols]
    cols = (cols + remaining)[:max_plots]

    paths = []
    for c in cols:
        path = os.path.join(IN_IMAGES_DIR, f"{c.replace('/', '_')}.png")
        plt.figure(figsize=(10, 4))
        plt.plot(daily_df.index, daily_df[c], alpha=0.8)
        plt.title(c)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths

def _build_claude_payload_from_files(
    master_schema_json: str,
    short_image_path: str,
    feature_image_paths: List[str],
    short_stock: str,
    max_number_of_features: int = 8,
    last_error: str = ""
) -> dict:
    with open(master_schema_json, "r") as f:
        schema_text = f.read()

    prompt = f"""
Generate leakage-safe daily features for predicting the biweekly short interest of {short_stock}. 
Constraints:
- Output Python code ONLY, no explanation, inside a single module containing functions named feature_*.
- Each function must accept a pandas.DataFrame df (daily master; DateTimeIndex; prefixed columns) and return a Series or DataFrame indexed daily.
- Do NOT aggregate to the biweekly target; the training pipeline will align and lag features by 14 days to prevent lookahead.
- Use only information available up to each day; avoid any target-aware transforms or label leakage.
- Prefer robust transformations: rolling means/volatility, exponentially-weighted stats, cross-asset spreads, normalized flows, volatility regimes, etc.
- Cap feature count to at most {max_number_of_features}. Keep columns well-named.

Example function signature:
def feature_rolling_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    # return a daily DataFrame with stable, non-leaky rolling stats

Important:
- Handle missing data gracefully with ffill/bfill caps where reasonable, then compute.
- Keep computation efficient; avoid O(n^2) operations or giant joins.
- Columns should be simple ASCII names; avoid special characters.
""".strip()

    content = [{"type": "text", "text": prompt}]
    # schema as text
    content.append({"type": "text", "text": f"MASTER_SCHEMA_JSON:\n{schema_text}"})
    # short interest plot
    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _encode_image_to_base64(short_image_path),
        },
    })
    # representative feature images
    for p in feature_image_paths:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_image_to_base64(p),
            },
        })
    if last_error:
        content.append({"type": "text", "text": f"PREVIOUS_ERROR_CONTEXT:\n{last_error}"})

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": min(CLAUDE_MAX_TOKENS, 4000),  # keep the reply focused on code
        "temperature": 0.2,
        "messages": [{"role": "user", "content": content}],
    }
    return payload

def parse_response(response):

    start = response.find("```python")
    if start == -1:
        start = response.find("```")
        if start == -1:
            print("No code block found in the response.")
            return None
        else:
            end = response.find("```", start + 3)
            code_block = response[start + 10:end].strip()
    else:
        end = response.find("```", start + 9)
        code_block = response[start + 9:end].strip()
    
    # Now we need to save this code block as a python script
    with open("generated_features.py", "w") as f:
        f.write(code_block)
        print('Saved generated_features.py')
    
    return code_block


def _call_claude_and_write_generated_features(payload: dict) -> Optional[str]:
    client = anthropic.Anthropic(api_key=CLAUDE_API)
    full = ""
    try:
        with client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full += text
    except Exception as e:
        print(f"Claude API error: {e}")
        return None
    
    code_block = parse_response(full)

    return code_block


def generate_llm_features(daily_df: pd.DataFrame, short_df: pd.DataFrame, stock_list: List[str], short_stock: str, max_features: int = 8, last_error: str = "") -> bool:
    """
    Returns True if generated_features.py is successfully written.
    """
    # Persist compact context
    master_schema_json = _save_master_schema_json(daily_df, stock_list, short_stock)
    short_plot_path = _save_short_interest_plot(short_df, short_stock)
    feature_image_paths = _save_sample_column_plots(daily_df, max_plots=8)

    # Build payload and call Claude
    payload = _build_claude_payload_from_files(
        master_schema_json,
        short_plot_path,
        feature_image_paths,
        short_stock,
        max_number_of_features=max_features,
        last_error=last_error
    )
    code = _call_claude_and_write_generated_features(payload)
    return code is not None


def validate_feature(feature_df: pd.DataFrame, 
                     feature_name: str,
                     min_valid_ratio: float = 0.5,
                     max_constant_ratio: float = 0.95) -> Tuple[bool, str]:
    """Validate generated feature for common issues."""
    if feature_df is None or feature_df.empty:
        return False, f"{feature_name}: returned empty DataFrame"

    if not isinstance(feature_df.index, pd.DatetimeIndex):
        try:
            feature_df.index = pd.to_datetime(feature_df.index)
        except Exception as e:
            return False, f"{feature_name}: invalid date index - {e}"

    for col in feature_df.columns:
        series = feature_df[col]

        # Replace infinites
        if np.isinf(series).any():
            series = series.replace([np.inf, -np.inf], np.nan)
            feature_df[col] = series

        # Check NaN ratio
        nan_ratio = series.isna().sum() / len(series)
        if nan_ratio > (1 - min_valid_ratio):
            return False, f"{feature_name}[{col}]: {nan_ratio:.1%} NaN (threshold: {1-min_valid_ratio:.1%})"

        # Check for constants
        non_nan = series.dropna()
        if len(non_nan) > 0:
            if non_nan.nunique() == 1:
                return False, f"{feature_name}[{col}]: constant value"

            most_common_ratio = non_nan.value_counts().iloc[0] / len(non_nan)
            if most_common_ratio > max_constant_ratio:
                return False, f"{feature_name}[{col}]: {most_common_ratio:.1%} identical values"

    return True, ""


def safe_execute_features(daily_df: pd.DataFrame,
                          target_dates: pd.DatetimeIndex,
                          feature_file: str = "generated_features.py",
                          lag_days: int = DEFAULT_LAG_DAYS) -> Tuple[pd.DataFrame, List[str]]:
    """Execute LLM-generated feature functions with validation."""
    import importlib.util
    import sys

    if not os.path.exists(feature_file):
        return pd.DataFrame(), [f"{feature_file} not found"]

    # Import generated features
    spec = importlib.util.spec_from_file_location("generated_features", feature_file)
    gen_features = importlib.util.module_from_spec(spec)

    if "generated_features" in sys.modules:
        del sys.modules["generated_features"]

    try:
        spec.loader.exec_module(gen_features)
    except Exception as e:
        return pd.DataFrame(), [f"Failed to import {feature_file}: {e}"]

    # Find feature functions
    feature_funcs = [
        getattr(gen_features, name)
        for name in dir(gen_features)
        if callable(getattr(gen_features, name)) and name.startswith('feature_')
    ]

    print(f"\nFound {len(feature_funcs)} feature functions")

    feature_dfs = []
    error_log = []

    for func in feature_funcs:
        func_name = func.__name__

        try:
            feature_result = func(daily_df)

            if isinstance(feature_result, pd.Series):
                feature_result = feature_result.to_frame()

            is_valid, error_msg = validate_feature(feature_result, func_name)

            if not is_valid:
                print(f"  ✗ {func_name}: {error_msg}")
                error_log.append(error_msg)
                continue

            print(f"  ✓ {func_name}: {len(feature_result.columns)} column(s)")
            feature_dfs.append(feature_result)

        except Exception as e:
            error_msg = f"{func_name}: Execution error - {e}"
            print(f"  ✗ {error_msg}")
            error_log.append(error_msg)

    if not feature_dfs:
        return pd.DataFrame(), error_log

    # Concatenate and align to target dates
    all_features_daily = pd.concat(feature_dfs, axis=1)
    features_aligned = align_to_targets_with_lag(all_features_daily, target_dates, lag_days)

    print(f"\nFeatures aligned to targets: {features_aligned.shape}")

    return features_aligned, error_log


def create_full_feature_set(daily_df: pd.DataFrame,
                            target_dates: pd.DatetimeIndex,
                            lag_days: int = DEFAULT_LAG_DAYS) -> pd.DataFrame:
    """Combine raw features and engineered features."""
    # Raw features (aligned with lag)
    raw_features = align_to_targets_with_lag(daily_df, target_dates, lag_days)

    # Engineered features (if available)
    if os.path.exists("generated_features.py"):
        engineered_features, errors = safe_execute_features(daily_df, target_dates, lag_days=lag_days)

        if errors:
            print(f"\n⚠ {len(errors)} features rejected")
            for err in errors:
                print(f"  - {err}")

        combined = pd.concat([raw_features, engineered_features], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined
    else:
        print("\n⚠ No generated_features.py found - using raw features only")
        return raw_features


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

def _prep_block(X):
    Xp = X.ffill().bfill()
    return Xp.fillna(Xp.median())
    
def train_xgboost_with_optuna(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               n_trials: int = 100,
                               n_splits: int = N_CV_SPLITS) -> Tuple[xgb.Booster, Dict, List[str], StandardScaler]:
    """
    Train XGBoost with Optuna hyperparameter optimization.
    Returns trained booster, best params, kept column names, and target scaler.
    """
    # Use StandardScaler for target (better for values in millions)
    y_scaler = y_train.copy() #RobustScaler()
    # y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    # y_train_scaled = pd.Series(y_train_scaled, index=y_train.index)
    y_train_scaled = y_train

    # Expanding window CV splits
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

    # Prune features BEFORE Optuna optimization (critical fix)
    kept_cols = prune_features_smart(X_train, y_train, 
                                    nan_thresh=0.40,
                                    max_features=40)
    # Force to keep the lagged short interest
    must_keep = [c for c in X_train.columns if c in ['si_lag1', 'si_lag2', 'si_logdiff1']]
    kept_cols = list(dict.fromkeys(must_keep + kept_cols)) 
    print(f"\nSelected {len(kept_cols)} features before optimization")
    X_train_pruned = X_train[kept_cols]


    
    # Optuna objective
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "seed": 42,
            "eta": trial.suggest_float("eta", 0.03, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            "lambda": trial.suggest_float("lambda", 0.1, 3.0),
        }
        n_estimators = trial.suggest_int("n_estimators", 400, 1600)

        rmses = []
        for train_idx, val_idx in cv_splits:
            X_tr = _prep_block(X_train_pruned.iloc[train_idx])
            y_tr = y_train_scaled.iloc[train_idx]
            X_val = _prep_block(X_train_pruned.iloc[val_idx])
            y_val = y_train_scaled.iloc[val_idx]

            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "validation")],
                early_stopping_rounds=100,
                verbose_eval=False
            )

            y_pred_scaled = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
            # Inverse transform to original scale
            #y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_pred = y_pred_scaled
            y_true = y_train.iloc[val_idx].values

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rmses.append(rmse)

        return float(np.mean(rmses))

    # Run optimization
    print("\nRunning Optuna hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # Final training with best params
    val_size = max(3, int(0.2 * len(X_train)))
    X_tr_final = X_train_pruned.iloc[:-val_size]
    y_tr_final = y_train_scaled.iloc[:-val_size]
    X_val_final = X_train_pruned.iloc[-val_size:]
    y_val_final = y_train_scaled.iloc[-val_size:]

    X_tr_pruned_final = _prep_block(X_tr_final)
    X_val_pruned_final = _prep_block(X_val_final)

    dtrain = xgb.DMatrix(X_tr_pruned_final, label=y_tr_final)
    dval = xgb.DMatrix(X_val_pruned_final, label=y_val_final)

    # Final params
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

    return final_model, final_params, kept_cols, y_scaler



def evaluate_model(model: xgb.Booster,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  kept_cols: List[str],
                  y_scaler: StandardScaler) -> Dict:
    """Evaluate model with scaler."""
    X_train_pruned = _prep_block(X_train[kept_cols])
    X_test_pruned  = _prep_block(X_test[kept_cols])
    
    dtrain = xgb.DMatrix(X_train_pruned)
    dtest = xgb.DMatrix(X_test_pruned)
    
    # Predict in scaled space, then inverse transform
    y_train_pred_scaled = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))
    y_test_pred_scaled = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    
    # y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    # y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    y_train_pred = y_train_pred_scaled
    y_test_pred = y_test_pred_scaled
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"IN-SAMPLE (Train):  RMSE={train_rmse:.4f}, R²={train_r2:.4f}")
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
                      labels=['In-Sample', 'Out-of-Sample'])
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

def run_pipeline(stock_list: List[str],
                 short_stock: str,
                 use_llm: bool = False,
                 n_trials: int = 100):
    """Execute full prediction pipeline."""
    print("="*60)
    print("SHORT INTEREST PREDICTION PIPELINE")
    print("="*60)

    # 1. Load short interest data (target)
    print(f"\n1. Loading short interest data for {short_stock}...")
    short_df = get_finra_short_data(short_stock)
    target_dates = short_df.index
    print(f"   Found {len(short_df)} biweekly measurements")

    # 2. Load stock data
    print(f"\n2. Loading daily stock data...")
    stock_dict = {}
    for stock in tqdm(stock_list, desc="Loading stocks"):
        try:
            stock_dict[stock] = get_stock_data(stock)
        except Exception as e:
            print(f"   ✗ Failed to load {stock}: {e}")

    if not stock_dict:
        raise ValueError("No valid stock data loaded")

    # 3. Load indices
    print("\n3. Loading market indices...")
    try:
        spx_df = get_stock_data('SPX Index')
        vix_df = get_stock_data('VIX Index')
        print("   ✓ Loaded SPX and VIX")
    except:
        spx_df = None
        vix_df = None
        print("   ⚠ Could not load indices")

    # 4. Create master daily dataframe
    print("\n4. Creating daily master dataframe...")
    daily_df = create_daily_master_df(stock_dict, spx_df, vix_df)
    print(f"   Shape: {daily_df.shape}")
    print(f"   Date range: {daily_df.index.min().date()} to {daily_df.index.max().date()}")

    # 5. Generate features (with LLM if enabled)
    if use_llm and CLAUDE_API:
        print("\n5. Generating features with Claude API...")
        try:
            ok = generate_llm_features(
                daily_df=daily_df,
                short_df=short_df,
                stock_list=stock_list,
                short_stock=short_stock,
                max_features=8,
                last_error=""
            )
            if ok:
                print(" ✓ LLM-generated features module written (generated_features.py)")
            else:
                print(" ✗ LLM generation failed; proceeding with raw features only")
        except Exception as e:
            print(f" ✗ LLM pipeline exception: {e}; proceeding with raw features only")
    else:
        print("\n5. Skipping LLM feature generation.")

    # 6. Create feature set
    print("\n6. Creating feature set...")
    features_df = create_full_feature_set(daily_df, target_dates, lag_days=DEFAULT_LAG_DAYS)
    print(f"   Total features: {features_df.shape[1]}")

    # 7. Train/test split
    print("\n7. Creating train/test split...")
    n_total = len(features_df)
    n_test = max(int(n_total * TEST_SIZE), 5)
    n_train = n_total - n_test

    # Align with target
    data = features_df.join(short_df['short_interest'], how='inner').sort_index()
    
    # Add leakage-safe biweekly lags
    data['si_lag1'] = data['short_interest'].shift(1)
    data['si_lag2'] = data['short_interest'].shift(2)

    # Include a stationary signal the trees can exploit
    data['si_logdiff1'] = np.log1p(data['short_interest']) - np.log1p(data['short_interest'].shift(1))

    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]

    X_train = train_data.drop(columns=['short_interest'])
    y_train = train_data['short_interest']
    X_test = test_data.drop(columns=['short_interest'])
    y_test = test_data['short_interest']

    cutoff_date = train_data.index[-1]

    print(f"   Cutoff date: {cutoff_date.date()}")
    print(f"   Train: {len(X_train)} samples ({train_data.index[0].date()} to {cutoff_date.date()})")
    print(f"   Test:  {len(X_test)} samples ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    # 8. Train model
    print("\n8. Training XGBoost model with Optuna...")
    model, best_params, kept_cols, y_scaler = train_xgboost_with_optuna(
        X_train, y_train, n_trials=n_trials
    )

    # 9. Evaluate
    print("\n9. Evaluating model...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, kept_cols, y_scaler)

    # 10. Visualize
    print("\n10. Creating visualizations...")
    plot_predictions(
        y_train, metrics['train_pred'],
        y_test, metrics['test_pred'],
        cutoff_date, short_stock, metrics
    )

    # 11. Save model
    print("\n11. Saving model...")
    os.makedirs("outputs", exist_ok=True)
    model.save_model("outputs/model.json")

    with open("outputs/model_meta.json", "w") as f:
        json.dump({
            "kept_cols": kept_cols,
            "best_params": best_params,
            "metrics": {
                "train_rmse": float(metrics['train_rmse']),
                "train_r2": float(metrics['train_r2']),
                "test_rmse": float(metrics['test_rmse']),
                "test_r2": float(metrics['test_r2'])
            },
            "cutoff_date": cutoff_date.isoformat()
        }, f, indent=2)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"✓ Model saved to: outputs/model.json")
    print(f"✓ Predictions saved to: outputs/predictions.png")
    print(f"✓ OOS RMSE: {metrics['test_rmse']:.4f}")
    print(f"✓ OOS R²: {metrics['test_r2']:.4f}")
    print("="*60)

    return model, metrics, kept_cols


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
