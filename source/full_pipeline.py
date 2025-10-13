"""

# Available Market Indices:
# - SPX: S&P 500 Index
# - RUI: Russell 1000 Index
# - RUA: Russell 3000 Index


Stock Short Interest Prediction Pipeline - Dual Model (XGBoost + Ridge)

This pipeline predicts biweekly short interest using:
1. Daily market data (Bloomberg or Alpha Vantage)
2. Optional LLM-generated features via Claude API
3. TWO machine learning models with Optuna optimization:
   - XGBoost (gradient boosting)
   - Ridge Regression (linear model with L2 regularization)

Usage:
    # Text-based mode (default, efficient)
    python full_pipeline.py --use-llm
    
    # Image-based mode (sends visualizations to Claude)
    python full_pipeline.py --use-llm --use-images
    
    # Quick test with fewer trials
    python full_pipeline.py --use-llm --n-trials 20

Key Features:
- 14-day lag alignment to prevent lookahead bias
- Multi-pass LLM feature refinement with feedback
- Expanding window cross-validation
- Hyperparameter optimization via Optuna for BOTH models
- Separate predictions and visualizations for XGBoost and Ridge
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# API Keys
ALPHA_API = os.getenv('ALPHA_API')
CLAUDE_API = os.getenv('CLAUDE_API')
FINRA_CLIENT_ID = os.getenv('FINRA_CLIENT_ID')
FINRA_CLIENT_PASS = os.getenv('FINRA_CLIENT_PASS')

# Claude Configuration
CLAUDE_MODEL = 'claude-sonnet-4-20250514'
CLAUDE_MAX_TOKENS = 52000

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH_BBG = os.path.join(SCRIPT_DIR, '..', 'data', 'BloombergJPM_noformula.xlsx')
IN_IMAGES_DIR = "in_images"
IN_TABULAR_DIR = "in_tabular"
OUTPUT_DIR = "outputs"

# Create directories
os.makedirs(IN_IMAGES_DIR, exist_ok=True)
os.makedirs(IN_TABULAR_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Configuration
DEFAULT_LAG_DAYS = 14  # Look-ahead bias prevention
TEST_SIZE = 0.2
N_CV_SPLITS = 5

# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================

def get_finra_auth() -> str:
    """
    Authenticate with FINRA API using OAuth2 client credentials.
    
    Returns:
        str: Access token for FINRA API
    """
    auth = HTTPBasicAuth(FINRA_CLIENT_ID, FINRA_CLIENT_PASS)
    response = requests.post(
        url="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token",
        data={'grant_type': 'client_credentials'},
        auth=auth
    )
    response.raise_for_status()
    return response.json()['access_token']

def get_finra_short_data(symbol: str, date_range: List[Optional[str]] = [None, None]) -> pd.DataFrame:
    """
    Fetch biweekly short interest data from FINRA API.
    
    Args:
        symbol: Stock ticker symbol
        date_range: [start_date, end_date] in 'YYYY-MM-DD' format, or [None, None]
    
    Returns:
        DataFrame with DateTimeIndex and 'short_interest' column
    
    Note:
        - Saves visualization to outputs/{symbol}_short_interest.png
        - Saves metadata to in_tabular/{symbol}_description.json
    """
    token = get_finra_auth()
    
    # Build API request
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
    
    # Parse CSV response
    df = pd.read_csv(io.StringIO(response.text), sep=",", engine="python", keep_default_na=False)
    
    if df.empty:
        raise ValueError(f"No short interest data found for {symbol}")
    
    # Process data
    df = df[['settlementDate', 'currentShortPositionQuantity']].copy()
    df.columns = ['date', 'short_interest']
    df['date'] = pd.to_datetime(df['date'])
    df['short_interest'] = df['short_interest'].astype(float)
    df = df.sort_values('date').set_index('date')
    
    # Save metadata
    description = {
        "symbol": symbol,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).tolist(),
        "shape": df.shape
    }
    with open(f"{IN_TABULAR_DIR}/{symbol}_description.json", "w") as f:
        json.dump(description, f, indent=2)
    
    # Save visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['short_interest'], marker='o')
    plt.title(f"{symbol} - Short Interest (FINRA)")
    plt.xlabel("Date")
    plt.ylabel("Short Interest")
    plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/{symbol}_short_interest.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return df

def get_alpha_vantage_data(symbol: str, date_range: List[Optional[str]] = [None, None]) -> pd.DataFrame:
    """
    Fetch daily stock data from Alpha Vantage API.
    
    Args:
        symbol: Stock ticker symbol
        date_range: [start_date, end_date] in 'YYYY-MM-DD' format, or [None, None]
    
    Returns:
        DataFrame with DateTimeIndex and columns: open, high, low, close, adjusted_close, volume, etc.
    
    Note:
        Requires ALPHA_API key in .env file
    """
    if not ALPHA_API:
        raise ValueError("ALPHA_API key not found in .env file")
    
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'apikey': ALPHA_API,
        'outputsize': 'full'
    }
    
    response = requests.get('https://www.alphavantage.co/query', params=params)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Error fetching data from Alpha Vantage for {symbol}: {data}")
    
    # Parse time series
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    
    # Apply date range filter
    if date_range[0]:
        df = df[df.index >= pd.to_datetime(date_range[0])]
    if date_range[1]:
        df = df[df.index <= pd.to_datetime(date_range[1])]
    
    # Clean column names
    df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df = df.astype(float).sort_index()
    
    return df

def parse_bloomberg_excel(file_path: str, start_row: int = 1, start_col: int = 2) -> Dict[str, pd.DataFrame]:
    """
    Parse Bloomberg Excel file containing multiple tickers.
    
    Args:
        file_path: Path to Bloomberg Excel file
        start_row: Row index where data starts
        start_col: Column index where data starts
    
    Returns:
        Dict mapping ticker names to DataFrames with columns:
        Date, PX_LAST, PX_VOLUME, DVD_SH_LAST, PX_BID, PX_ASK, OPEN_INT_TOTAL_PUT, OPEN_INT_TOTAL_CALL
    """
    try:
        df = pd.read_excel(file_path, header=None)
    except:
        df = pd.read_csv(file_path, header=None)
    
    data_section = df.iloc[start_row:, start_col:]
    
    expected_columns = [
        'Date', 'PX_LAST', 'PX_VOLUME', 'DVD_SH_LAST',
        'PX_BID', 'PX_ASK', 'OPEN_INT_TOTAL_PUT', 'OPEN_INT_TOTAL_CALL'
    ]
    
    # Find ticker positions in first row
    first_row = data_section.iloc[0].fillna('').astype(str)
    ticker_positions = [(cell.strip(), idx) for idx, cell in enumerate(first_row)
                        if 'Equity' in cell or 'Index' in cell]
    
    if not ticker_positions:
        raise ValueError("No tickers found in Excel file")
    
    result_dict = {}
    
    for i, (ticker_name, col_start) in enumerate(ticker_positions):
        # Determine column range for this ticker
        if i + 1 < len(ticker_positions):
            col_end = ticker_positions[i + 1][1]
        else:
            col_end = len(data_section.columns)
        
        n_cols = col_end - col_start
        ticker_data = data_section.iloc[:, col_start:col_end]
        ticker_values = ticker_data.iloc[2:].copy()
        
        # Assign column names
        if n_cols == len(expected_columns):
            ticker_values.columns = expected_columns
        else:
            actual_cols = expected_columns[:n_cols] if n_cols <= len(expected_columns) else \
                         expected_columns + [f'Col_{j}' for j in range(len(expected_columns), n_cols)]
            ticker_values.columns = actual_cols
        
        # Clean data
        mask = ticker_values.isin(['#N/A N/A', '#N/A', 'N/A', '', '#N/A Field Not Applicable'])
        ticker_values = ticker_values.mask(mask, np.nan)
        
        if 'Date' not in ticker_values.columns:
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

def get_stock_data(ticker: str, file_path: str = FILE_PATH_BBG,
                   use_alpha_vantage: bool = False) -> pd.DataFrame:
    """
    Get stock data from Bloomberg file or Alpha Vantage.
    
    Args:
        ticker: Stock symbol
        file_path: Path to Bloomberg Excel file
        use_alpha_vantage: If True, use Alpha Vantage instead of Bloomberg
    
    Returns:
        DataFrame with daily stock data
    """
    if use_alpha_vantage:
        return get_alpha_vantage_data(ticker)
    
    # Use Bloomberg file
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

def create_daily_master_df(stock_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all daily stock data into single master DataFrame.
    
    Args:
        stock_dict: Dict mapping stock symbols to their DataFrames
    
    Returns:
        Master DataFrame with columns prefixed by stock symbol (e.g., AAPL_PX_LAST)
    
    Note:
        - Forward-fills missing values with 5-day limit to avoid stale data
        - All stocks aligned to union of dates
    """
    all_dates = pd.DatetimeIndex([])
    for df in stock_dict.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    
    master_df = pd.DataFrame(index=all_dates)
    
    # Add stock data with prefixes
    for stock, df in stock_dict.items():
        aligned = df.reindex(master_df.index).ffill(limit=5)  # 5-day stale limit
        aligned = aligned.add_prefix(f"{stock}_")
        master_df = master_df.join(aligned, how='left')
    
    master_df = master_df.dropna(how='all', axis=1)
    return master_df

def align_to_targets_with_lag(daily_df: pd.DataFrame,
                               target_dates: pd.DatetimeIndex,
                               lag_days: int = DEFAULT_LAG_DAYS) -> pd.DataFrame:
    """
    Align daily features to target dates with lag to prevent lookahead bias.
    
    Args:
        daily_df: Daily feature DataFrame
        target_dates: Biweekly target dates (short interest dates)
        lag_days: Number of days to lag (default: 14)
    
    Returns:
        DataFrame aligned to target dates, using data from lag_days before each target
    
    Example:
        For target date 2024-03-15 with lag_days=14:
        - Cutoff date: 2024-03-01
        - Uses last available data on or before 2024-03-01
        - Ensures model cannot see future data
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
# 3. IMAGE GENERATION (OPTIONAL)
# ============================================================================

def visualize_data_for_llm(daily_df: pd.DataFrame,
                           stock_list: List[str],
                           short_stock: str,
                           max_cols_per_stock: int = 6) -> None:
    """
    Generate time series visualizations for LLM analysis.
    
    Args:
        daily_df: Master daily DataFrame with prefixed columns
        stock_list: List of stock symbols to visualize
        short_stock: Target stock symbol
        max_cols_per_stock: Maximum columns to plot per stock
    
    Note:
        - Saves plots to in_images/ directory
        - Each column gets its own plot
        - Short interest plot copied from outputs/ to in_images/
    """
    for stock in stock_list:
        stock_cols = [c for c in daily_df.columns if c.startswith(f"{stock}_")][:max_cols_per_stock]
        
        for col in tqdm(stock_cols, desc=f"  Plotting {stock}"):
            plt.figure(figsize=(10, 6))
            daily_df[col].plot()
            plt.title(f"{stock} - {col.replace(f'{stock}_', '')}")
            plt.xlabel("Date")
            plt.ylabel(col.replace(f'{stock}_', ''))
            plt.grid(alpha=0.3)
            safe_name = col.replace('/', '_').replace(' ', '_')
            plt.savefig(f"{IN_IMAGES_DIR}/{safe_name}.png", dpi=100, bbox_inches='tight')
            plt.close()
    
    # Copy short interest plot to in_images if it exists
    short_plot_src = f"{OUTPUT_DIR}/{short_stock}_short_interest.png"
    short_plot_dst = f"{IN_IMAGES_DIR}/{short_stock}_short_interest.png"
    if os.path.exists(short_plot_src):
        import shutil
        shutil.copy(short_plot_src, short_plot_dst)

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string for Claude API.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base64-encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_image_dict(stock_list: List[str], short_stock: str) -> Dict[str, Dict]:
    """
    Build dictionary of encoded images for Claude API.
    
    Args:
        stock_list: List of stock symbols
        short_stock: Target stock symbol
    
    Returns:
        Dict mapping filenames to {'type': 'image', 'data': base64_string}
    """
    image_dict = {}
    all_stocks = stock_list + [short_stock]
    
    for file in os.listdir(IN_IMAGES_DIR):
        if file.endswith('.png'):
            # Check if relevant to our stocks
            relevant = any(file.startswith(stock) for stock in all_stocks)
            if relevant:
                encoded = encode_image_to_base64(os.path.join(IN_IMAGES_DIR, file))
                image_dict[file] = {
                    'type': 'image',
                    'data': encoded
                }
    
    return image_dict

def clear_generated_artifacts(stock_list: List[str], short_stock: str) -> None:
    """
    Clean up generated images and JSON files.
    
    Args:
        stock_list: List of stock symbols
        short_stock: Target stock symbol
    """
    all_stocks = stock_list + [short_stock]
    
    # Clear images
    if os.path.exists(IN_IMAGES_DIR):
        for file in os.listdir(IN_IMAGES_DIR):
            if any(file.startswith(stock) for stock in all_stocks) and file.endswith('.png'):
                try:
                    os.remove(os.path.join(IN_IMAGES_DIR, file))
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")
    
    # Clear tabular metadata
    if os.path.exists(IN_TABULAR_DIR):
        for file in os.listdir(IN_TABULAR_DIR):
            if any(stock in file for stock in all_stocks) and file.endswith('.json'):
                try:
                    os.remove(os.path.join(IN_TABULAR_DIR, file))
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")

# ============================================================================
# 4. LLM FEATURE GENERATION
# ============================================================================

def _summarize_dataset(daily_df: pd.DataFrame, max_cols: int = 200) -> str:
    """
    Build compact JSON string with per-column metadata for LLM.
    
    Args:
        daily_df: Master daily DataFrame
        max_cols: Maximum columns to include
    
    Returns:
        JSON string with column statistics
    """
    cols = list(daily_df.columns[:max_cols])
    summary = {
        "index_type": str(type(daily_df.index).__name__),
        "n_rows": int(len(daily_df)),
        "n_cols": int(daily_df.shape[1]),
        "columns": [],
        "notes": [
            "Columns prefixed by ticker (e.g., AAPL_PX_LAST)",
            "Daily frequency with possible gaps",
            "Pipeline aligns features to biweekly targets with 14-day lag"
        ]
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

def _build_claude_payload(
    daily_df: pd.DataFrame,
    short_stock: str,
    max_number_of_features: int = 15,
    last_error: str = "",
    feedback_md: str = "",
    must_keep: Optional[List[str]] = None,
    avoid: Optional[List[str]] = None,
    use_images: bool = False,
    image_dict: Optional[Dict] = None,
) -> dict:
    """
    Build Claude API payload for feature generation.
    
    Args:
        daily_df: Master daily DataFrame
        short_stock: Target stock symbol
        max_number_of_features: Maximum features to generate
        last_error: Error from previous attempt
        feedback_md: Feedback from previous pass
        must_keep: Features to preserve
        avoid: Features to avoid recreating
        use_images: Whether to include images
        image_dict: Dictionary of encoded images
    
    Returns:
        Dict ready for Claude API
    """
    # Optimized system prompt - CRITICAL: enforce "feature_" prefix
    system_constraints = f"""Generate {max_number_of_features} leakage-safe DAILY features for {short_stock} short interest prediction.

{{
  "output_format": {{
    "line_1": "# PLAN: {{\\"features\\":[...]}}",
    "remaining": "Python functions named feature_* taking df→Series/DataFrame"
  }},
  "constraints": {{
    "imports": ["pandas as pd", "numpy as np ONLY"],
    "transforms": ["rolling/ewm windows", "set min_periods≥1"],
    "forbidden": ["short_interest", "si_lag", "target", "label", "bfill()", "backfill", "fillna(method="],
    "naming": ["MUST start with feature_", "lowercase_with_underscores", "ASCII", "max_{max_number_of_features} features"]
  }},
  "rules": [
    "CRITICAL: All function names MUST start with 'feature_' (e.g., feature_price_momentum_5d)",
    "Use df.ffill() NOT fillna(method='ffill') - deprecated syntax causes errors",
    "Use df column names from schema",
    "No MUST_KEEP redefinitions",
    "No AVOID recreations",
    "Low correlation with MUST_KEEP"
  ]
}}

EXAMPLE:
def feature_price_momentum_5d(df):
    close = df['TICKER_close'].ffill()
    return (close / close.rolling(5, min_periods=1).mean() - 1).fillna(0)

Emit ONLY fenced Python code. No explanations."""
    
    # Build user content with condensed structure
    parts = []
    
    # Data context
    if use_images and image_dict:
        parts.append(f"CONTEXT: {len(image_dict)} time series visualizations showing {short_stock} metrics.")
    
    schema = _summarize_dataset(daily_df, max_cols=200)
    parts.append(f"SCHEMA:\n{schema}")
    
    # Feedback/constraints (compact format)
    if feedback_md:
        parts.append(f"\nFEEDBACK:\n{feedback_md}")
    
    if must_keep:
        parts.append(f"\nMUST_KEEP: {', '.join(map(str, must_keep[:50]))}")
    
    if avoid:
        parts.append(f"\nAVOID: {', '.join(map(str, avoid[:50]))}")
    
    if last_error:
        parts.append(f"\nFIX_ERROR: {last_error}")
    
    # Build content array
    content = []
    
    # Images first if using
    if use_images and image_dict:
        for filename, img_data in image_dict.items():
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': img_data['data']
                }
            })
    
    # Text content
    content.append({
        'type': 'text',
        'text': "\n\n".join(parts)
    })
    
    return {
        "model": CLAUDE_MODEL,
        "system": system_constraints.strip(),
        "max_tokens": min(CLAUDE_MAX_TOKENS, 4000),
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": content}
        ]
    }

# Forbidden tokens for leakage detection
FORBIDDEN_TOKENS = ["short_interest", "si_lag", "si_logdiff", "target", "label"]

def static_leakage_guard(code: str) -> Tuple[bool, str]:
    """
    Check generated code for target leakage and syntax issues before execution.
    
    Args:
        code: Generated Python code
    
    Returns:
        (is_valid, error_message)
    """
    lower = code.lower()
    
    # Check for target leakage
    for tok in FORBIDDEN_TOKENS:
        if tok in lower:
            return False, f"Target leakage: forbidden token '{tok}' found"
    
    # Check for deprecated fillna syntax
    if "fillna(method=" in code:
        return False, "Deprecated syntax: use .ffill() or .fillna(value) instead of fillna(method=)"
    
    # Check for bfill/backfill
    if "bfill()" in lower or "backfill" in lower:
        return False, "Lookahead bias: bfill() or backfill detected"
    
    # Check for proper function naming
    func_names = re.findall(r'def\s+(\w+)\s*\(', code)
    bad_names = [n for n in func_names if not n.startswith('feature_')]
    if bad_names:
        return False, f"Naming violation: functions {bad_names} must start with 'feature_'"
    
    return True, ""

def parse_response(response: str, out_file: str) -> Tuple[Optional[str], str]:
    """
    Extract Python code block from Claude response and save to file.
    
    Args:
        response: Claude API response text
        out_file: Path to save generated code
    
    Returns:
        (code_block, error_message)
    
    Note:
        Handles PLAN manifest either inside or before the code fence
    """
    # Find code block
    start = response.find("```python")
    if start != -1:
        end = response.find("```", start + 9)
        code_block = response[start + 9:end].strip() if end != -1 else ""
        
        # Check if there's a PLAN before the fence
        pre_fence = response[:start].strip()
        plan_match = re.search(r"#\s*PLAN:\s*\{.*\}", pre_fence, re.DOTALL)
        if plan_match:
            # Prepend the PLAN to the code block
            code_block = plan_match.group(0) + "\n\n" + code_block
    else:
        # Fallback to generic fence
        start = response.find("```python")
        end = response.find("```", start + 3) if start != -1 else -1
        code_block = response[start + 3:end].strip() if end != -1 else ""
        
        # Check for PLAN before fence
        if start != -1:
            pre_fence = response[:start].strip()
            plan_match = re.search(r"#\s*PLAN:\s*\{.*\}", pre_fence, re.DOTALL)
            if plan_match:
                code_block = plan_match.group(0) + "\n\n" + code_block
    
    if not code_block:
        return None, "no_code_block"
    
    # Require PLAN manifest (check first 3 lines to be flexible)
    has_plan = False
    for line in code_block.splitlines()[:3]:
        if re.search(r"#\s*PLAN:\s*\{.*\}", line.strip()):
            has_plan = True
            break
    
    if not has_plan:
        return None, "no_plan"
    
    # Check for leakage and syntax issues
    ok, msg = static_leakage_guard(code_block)
    if not ok:
        return None, f"validation_error:{msg}"
    
    # Save to file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(code_block)
    
    return code_block, ""

def call_claude_and_write_features(payload: dict, out_file: str) -> Tuple[Optional[str], str]:
    """
    Call Claude API and save generated code.
    
    Args:
        payload: Claude API payload
        out_file: Path to save generated code
    
    Returns:
        (code_block, error_message)
    """
    client = anthropic.Anthropic(api_key=CLAUDE_API)
    full = ""
    
    try:
        with client.messages.stream(**payload) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full += text
    except Exception as e:
        print(f"\nError calling Claude API: {e}")
        return None, f"claude_error:{e}"
    
    return parse_response(full, out_file)

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
    use_images: bool = False,
) -> Tuple[bool, str]:
    """
    Generate feature module using Claude API.
    
    Args:
        daily_df: Master daily DataFrame
        short_df: Short interest DataFrame
        stock_list: List of stock symbols
        short_stock: Target stock symbol
        max_features: Maximum features to generate
        last_error: Error from previous attempt
        feedback_md: Feedback from previous pass
        must_keep: Features to preserve
        avoid: Features to avoid
        out_file: Path to save generated code
        use_images: Whether to use image mode
    
    Returns:
        (success, error_message)
    """
    # Build image dict if needed
    image_dict = None
    if use_images:
        image_dict = build_image_dict(stock_list, short_stock)
    
    payload = _build_claude_payload(
        daily_df=daily_df,
        short_stock=short_stock,
        max_number_of_features=max_features,
        last_error=last_error or "",
        feedback_md=feedback_md or "",
        must_keep=must_keep or [],
        avoid=avoid or [],
        use_images=use_images,
        image_dict=image_dict,
    )
    
    code, err = call_claude_and_write_features(payload, out_file)
    
    return (code is not None), (err or "")

# ============================================================================
# 5. FEATURE VALIDATION & EXECUTION
# ============================================================================

def _max_consecutive_unchanged(series: pd.Series) -> int:
    """
    Compute longest run of unchanged consecutive values.
    
    Args:
        series: Pandas Series
    
    Returns:
        Maximum consecutive unchanged count
    """
    s = series.copy()
    changed = s != s.shift(1)
    run_ids = changed.cumsum().fillna(0)
    lengths = run_ids.groupby(run_ids).transform("size") if len(run_ids) else pd.Series([], dtype=int)
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
    Validate generated feature for quality.
    
    Args:
        feature_df: Feature DataFrame
        feature_name: Name of feature
        min_valid_ratio: Minimum non-NaN ratio
        max_constant_ratio: Maximum ratio of most common value
        max_unchanged_ratio: Maximum ratio of unchanged consecutive values
    
    Returns:
        (is_valid, error_message)
    """
    if feature_df is None or feature_df.empty:
        return False, f"{feature_name}: empty DataFrame"
    
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
            return False, f"{feature_name}[{col}]: {nan_ratio:.1%} NaN"
        
        non_nan = series.dropna()
        if len(non_nan) == 0:
            return False, f"{feature_name}[{col}]: all NaN"
        
        # Constant check
        if non_nan.nunique() == 1:
            return False, f"{feature_name}[{col}]: constant value"
        
        # Near-constant check
        most_common_ratio = float(non_nan.value_counts().iloc[0] / len(non_nan))
        if most_common_ratio > max_constant_ratio:
            return False, f"{feature_name}[{col}]: {most_common_ratio:.1%} identical"
        
        # Unchanged run check
        mcur = _max_consecutive_unchanged(non_nan)
        if mcur >= 5:
            unchanged_ratio = float(mcur / max(1, len(non_nan)))
            if unchanged_ratio > max_unchanged_ratio:
                return False, f"{feature_name}[{col}]: {unchanged_ratio:.1%} unchanged run"
    
    return True, ""

def safe_execute_features(
    daily_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    feature_files: List[str],
    lag_days: int = DEFAULT_LAG_DAYS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Import and execute generated feature modules with validation.
    
    Args:
        daily_df: Master daily DataFrame
        target_dates: Target dates to align features to
        feature_files: List of generated feature file paths
        lag_days: Lag days for alignment
    
    Returns:
        (aligned_features_df, error_log)
    """
    feature_dfs, error_log = [], []
    
    for feature_file in feature_files:
        if not os.path.exists(feature_file):
            error_log.append(f"{feature_file} not found")
            continue
        
        mod_name = os.path.splitext(os.path.basename(feature_file))[0]
        spec = importlib.util.spec_from_file_location(mod_name, feature_file)
        module = importlib.util.module_from_spec(spec)
        
        # Clean import
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            error_log.append(f"Failed to import {feature_file}: {e}")
            continue
        
        # Find feature functions
        funcs = [
            getattr(module, name)
            for name in dir(module)
            if callable(getattr(module, name)) and name.startswith("feature_")
        ]
        
        print(f"\n  {feature_file}: found {len(funcs)} feature functions")
        
        if len(funcs) == 0:
            error_log.append(f"{feature_file}: No functions starting with 'feature_' found")
            print(f"  ✗ No valid feature functions (must start with 'feature_')")
            continue
        
        for func in funcs:
            try:
                res = func(daily_df)
                if isinstance(res, pd.Series):
                    # Name the Series after the function to avoid "0" column names
                    res.name = func.__name__
                    res = res.to_frame()
                
                ok, msg = validate_feature(res, func.__name__)
                if not ok:
                    error_log.append(f"{feature_file}:{func.__name__}: {msg}")
                    continue
                
                print(f"  ✓ {func.__name__}: {len(res.columns)} column(s)")
                feature_dfs.append(res)
            except Exception as e:
                error_log.append(f"{feature_file}:{func.__name__}: {e}")
                print(f"  ✗ {func.__name__}: {e}")
    
    if not feature_dfs:
        return pd.DataFrame(), error_log
    
    # Combine and align
    all_features_daily = pd.concat(feature_dfs, axis=1)
    all_features_daily = all_features_daily.loc[:, ~all_features_daily.columns.duplicated()]
    features_aligned = align_to_targets_with_lag(all_features_daily, target_dates, lag_days)
    
    return features_aligned, error_log

def create_full_feature_set(
    daily_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    lag_days: int = DEFAULT_LAG_DAYS,
    feature_files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Use ONLY LLM-generated features (no raw base features).
    Lagged short interest features will be added separately in the pipeline.
    
    Args:
        daily_df: Master daily DataFrame
        target_dates: Target dates to align to
        lag_days: Lag days for alignment
        feature_files: List of generated feature files
    
    Returns:
        LLM-generated features DataFrame aligned to target dates
    """
    # REMOVED: Raw features are no longer included
    # Only use LLM-generated engineered features
    engineered_features = pd.DataFrame()
    
    if feature_files:
        eng, errors = safe_execute_features(
            daily_df, target_dates, feature_files=feature_files, lag_days=lag_days
        )
        
        if errors:
            print(f"\n  ⚠ {len(errors)} features rejected:")
            for err in errors[:10]:  # Show first 10
                print(f"    - {err}")
        
        engineered_features = eng
    
    # Return only LLM features (deduplicate just in case)
    engineered_features = engineered_features.loc[:, ~engineered_features.columns.duplicated()]
    return engineered_features

def summarize_feedback_for_llm(
    gain_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    X_train: pd.DataFrame,
    top_k: int = 15,
    corr_thresh: float = 0.9,
) -> Tuple[str, List[str], List[str]]:
    """
    Summarize feature importance and redundancy for next LLM pass.
    
    Args:
        gain_df: Feature importance by gain
        perm_df: Feature importance by permutation
        X_train: Training features
        top_k: Number of top features to report
        corr_thresh: Correlation threshold for redundancy
    
    Returns:
        (feedback_markdown, must_keep_list, avoid_list)
    """
    top_by_gain = gain_df.head(top_k)["feature"].tolist()
    top_by_perm = perm_df.head(top_k)["feature"].tolist()
    top_union = list(dict.fromkeys(top_by_gain + top_by_perm))
    
    weak = perm_df.tail(max(5, min(15, len(perm_df)//4)))["feature"].tolist()
    
    # Find redundancy groups
    Xf = X_train[[c for c in top_union if c in X_train.columns]].ffill().fillna(0)
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
    
    # Build markdown
    md = []
    md.append("Top features by gain/perm (union):")
    for f in top_union[:top_k]:
        g = float(gain_df[gain_df.feature == f]["gain_norm"].values[0]) if f in set(gain_df.feature) else 0.0
        p = float(perm_df[perm_df.feature == f]["perm_r2_drop"].values[0]) if f in set(perm_df.feature) else 0.0
        md.append(f"- {f} | gain={g:.4f} | perm={p:.4f}")
    
    if weak:
        md.append("\nWeak features (candidates for removal):")
        for f in weak[:15]:
            md.append(f"- {f}")
    
    if groups:
        md.append("\nRedundancy groups (high correlation):")
        for grp in groups[:8]:
            md.append("- " + ", ".join(grp))
    
    feedback_md = "\n".join(md)
    must_keep = [str(x) for x in top_union[:min(12, len(top_union))]]
    avoid = [str(x) for x in weak[:min(10, len(weak))]]
    
    return feedback_md, must_keep, avoid

# ============================================================================
# 6. MODEL TRAINING & EVALUATION (DUAL MODEL: XGBoost + Ridge)
# ============================================================================

def compute_medians(X: pd.DataFrame) -> pd.Series:
    """Compute median values for imputation (no lookahead bias)."""
    Xf = X.ffill()  # Only forward fill
    return Xf.median()

def prep_with_medians(X: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    """Prepare data using training medians (no lookahead bias)."""
    Xf = X.ffill()  # Only forward fill, never backfill
    
    # Add missing columns
    missing_cols = [c for c in med.index if c not in Xf.columns]
    for c in missing_cols:
        Xf[c] = np.nan
    
    # Reorder to match training
    Xf = Xf.reindex(columns=med.index)
    return Xf.fillna(med)

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

def train_xgboost_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_splits: int = N_CV_SPLITS,
    use_pruning: bool = False
) -> Tuple[xgb.Booster, Dict, List[str], Optional[pd.Series]]:
    """
    Train XGBoost model with Optuna hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_splits: Number of CV splits
        use_pruning: If True, apply feature pruning before training (default: False)
    
    Returns:
        (trained_model, best_params, kept_feature_columns, training_medians)
    """
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
    
    print(f"\n  Created {len(cv_splits)} expanding window CV splits")
    
    if use_pruning:
        print(f"\n  🔧 Feature pruning ENABLED")
        kept_cols = prune_features_smart(
            X_train, y_train,
            nan_thresh=0.40,
            max_features=25
        )
    else:
        print(f"\n  ⚙ Feature pruning DISABLED - using all features")
        kept_cols = X_train.columns.tolist()
    
    # Keep safe target lags if present
    must_keep = [c for c in X_train.columns if c in ['si_lag1', 'si_lag2', 'si_logdiff_lag1']]
    kept_cols = list(dict.fromkeys(must_keep + kept_cols))
    
    # Display feature observation counts
    print("\n" + "="*90)
    print("XGBOOST - FEATURE OBSERVATION COUNTS")
    print("="*90)
    print(f"{'Feature':<50} {'Total':>10} {'Non-Null':>10} {'Null':>10} {'Valid %':>10}")
    print("-"*90)
    
    for col in kept_cols:
        total = len(X_train[col])
        non_null = int(X_train[col].notna().sum())
        null = int(X_train[col].isna().sum())
        valid_pct = (non_null / total * 100) if total > 0 else 0
        print(f"{col:<50} {total:>10} {non_null:>10} {null:>10} {valid_pct:>9.1f}%")
    
    print("="*90)
    print(f"Total features to be used in XGBoost training: {len(kept_cols)}")
    print("="*90 + "\n")
    
    X_train_pruned = X_train[kept_cols]
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'max_depth': 2,  # Maximum depth 2-3
            'min_child_weight': 10,  # Prevent overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eta': 0.1,
            'gamma': 1.0,  # Conservative splits
            'alpha': 5.0,  # Strong L1
            'lambda': 10.0,  # Strong L2
        }
        
        n_estimators = trial.suggest_int("n_estimators", 300, 1200)
        rmses = []
        
        for train_idx, val_idx in cv_splits:
            X_tr = X_train_pruned.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train_pruned.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            
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
    
    print("\n  Running XGBoost Optuna optimization...")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    
    print(f"\n  ✓ Best CV RMSE: {study.best_value:.4f}")
    print(f"    Best params: {best_params}")
    
    # Final fit with early stopping
    val_size = max(3, int(0.2 * len(X_train_pruned)))
    X_tr_final = X_train_pruned.iloc[:-val_size]
    y_tr_final = y_train.iloc[:-val_size]
    X_val_final = X_train_pruned.iloc[-val_size:]
    y_val_final = y_train.iloc[-val_size:]
    
    med_final = compute_medians(X_tr_final)
    dtrain = xgb.DMatrix(prep_with_medians(X_tr_final, med_final), label=y_tr_final)
    dval = xgb.DMatrix(prep_with_medians(X_val_final, med_final), label=y_val_final)
    
    final_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": 42,
        **{k: v for k, v in best_params.items() if k != "n_estimators"}
    }
    
    final_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=best_params["n_estimators"],
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=150,
        verbose_eval=50
    )
    
    return final_model, best_params, kept_cols, med_final

def train_ridge_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_splits: int = N_CV_SPLITS,
    use_pruning: bool = False
) -> Tuple[Ridge, Dict, List[str], Optional[pd.Series]]:
    """
    Train Ridge Regression model with Optuna hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_splits: Number of CV splits
        use_pruning: If True, apply feature pruning before training (default: False)
    
    Returns:
        (trained_model, best_params, kept_feature_columns, training_medians)
    """
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
    
    print(f"\n  Created {len(cv_splits)} expanding window CV splits")
    
    if use_pruning:
        print(f"\n  🔧 Feature pruning ENABLED")
        kept_cols = prune_features_smart(
            X_train, y_train,
            nan_thresh=0.40,
            max_features=25
        )
    else:
        print(f"\n  ⚙ Feature pruning DISABLED - using all features")
        kept_cols = X_train.columns.tolist()
    
    # Keep safe target lags if present
    must_keep = [c for c in X_train.columns if c in ['si_lag1', 'si_lag2', 'si_logdiff_lag1']]
    kept_cols = list(dict.fromkeys(must_keep + kept_cols))
    
    # Display feature observation counts
    print("\n" + "="*90)
    print("RIDGE - FEATURE OBSERVATION COUNTS")
    print("="*90)
    print(f"{'Feature':<50} {'Total':>10} {'Non-Null':>10} {'Null':>10} {'Valid %':>10}")
    print("-"*90)
    
    for col in kept_cols:
        total = len(X_train[col])
        non_null = int(X_train[col].notna().sum())
        null = int(X_train[col].isna().sum())
        valid_pct = (non_null / total * 100) if total > 0 else 0
        print(f"{col:<50} {total:>10} {non_null:>10} {null:>10} {valid_pct:>9.1f}%")
    
    print("="*90)
    print(f"Total features to be used in Ridge training: {len(kept_cols)}")
    print("="*90 + "\n")
    
    X_train_pruned = X_train[kept_cols]
    
    def objective(trial):
        # Ridge hyperparameters
        alpha = trial.suggest_float("alpha", 0.001, 1000.0, log=True)
        
        rmses = []
        
        for train_idx, val_idx in cv_splits:
            X_tr = X_train_pruned.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train_pruned.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            
            med = compute_medians(X_tr)
            X_tr_prep = prep_with_medians(X_tr, med)
            X_val_prep = prep_with_medians(X_val, med)
            
            # Train Ridge
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_tr_prep, y_tr)
            
            y_pred = model.predict(X_val_prep)
            rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
            rmses.append(rmse)
        
        return float(np.mean(rmses))
    
    print("\n  Running Ridge Optuna optimization...")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    
    print(f"\n  ✓ Best CV RMSE: {study.best_value:.4f}")
    print(f"    Best params: {best_params}")
    
    # Final fit on full training set
    med_final = compute_medians(X_train_pruned)
    X_train_prep = prep_with_medians(X_train_pruned, med_final)
    
    final_model = Ridge(alpha=best_params["alpha"], random_state=42)
    final_model.fit(X_train_prep, y_train)
    
    return final_model, best_params, kept_cols, med_final

def evaluate_xgboost(model: xgb.Booster,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_test: pd.DataFrame,
                     y_test: pd.Series,
                     kept_cols: List[str],
                     medians: Optional[pd.Series]) -> Dict:
    """
    Evaluate trained XGBoost model on train and test sets.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        kept_cols: Selected feature columns
        medians: Training medians for imputation
    
    Returns:
        Dict with metrics and predictions
    """
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
    print("XGBOOST - MODEL EVALUATION")
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

def evaluate_ridge(model: Ridge,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   kept_cols: List[str],
                   medians: Optional[pd.Series]) -> Dict:
    """
    Evaluate trained Ridge model on train and test sets.
    
    Args:
        model: Trained Ridge model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        kept_cols: Selected feature columns
        medians: Training medians for imputation
    
    Returns:
        Dict with metrics and predictions
    """
    X_train_pruned = X_train[kept_cols]
    X_test_pruned = X_test.reindex(columns=kept_cols)
    
    med = medians if medians is not None else compute_medians(X_train_pruned)
    
    X_train_prep = prep_with_medians(X_train_pruned, med)
    X_test_prep = prep_with_medians(X_test_pruned, med)
    
    y_train_pred = model.predict(X_train_prep)
    y_test_pred = model.predict(X_test_prep)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*60)
    print("RIDGE - MODEL EVALUATION")
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
    """
    Compute feature importance using gain and permutation methods (no lookahead bias).
    
    Args:
        model: Trained model
        X_ref: Reference features
        y_ref: Reference target
        kept_cols: Selected columns
        n_repeats: Permutation repetitions
    
    Returns:
        (gain_importance_df, permutation_importance_df)
    """
    # Gain importance
    gain_raw = model.get_score(importance_type="gain")
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
    
    # Permutation importance (no lookahead bias in prep)
    m = max(3, int(0.2 * len(X_ref)))
    Xh, yh = X_ref.iloc[-m:].copy(), y_ref.iloc[-m:].copy()
    
    def prep(df):
        """Forward fill only, then fill remaining with median"""
        return df.ffill().fillna(df.median())
    
    base = r2_score(yh, model.predict(xgb.DMatrix(prep(Xh)), iteration_range=(0, model.best_iteration + 1)))
    
    drops = []
    rng = np.random.default_rng(42)
    
    for feat in kept_cols:
        if feat not in Xh.columns:
            continue
        
        drop_vals = []
        for _ in range(n_repeats):
            Xp = Xh.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            r2p = r2_score(yh, model.predict(xgb.DMatrix(prep(Xp)), iteration_range=(0, model.best_iteration + 1)))
            drop_vals.append(base - r2p)
        
        drops.append((feat, float(np.nanmean(drop_vals))))
    
    perm_df = pd.DataFrame(drops, columns=["feature", "perm_r2_drop"]).sort_values("perm_r2_drop", ascending=False)
    
    return gain_df.sort_values("gain_norm", ascending=False), perm_df

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def plot_predictions(y_train: pd.Series,
                     y_train_pred: np.ndarray,
                     y_test: pd.Series,
                     y_test_pred: np.ndarray,
                     cutoff_date: pd.Timestamp,
                     symbol: str,
                     metrics: Dict,
                     model_name: str,
                     save_path: str = "outputs/predictions.png"):
    """
    Create comprehensive prediction visualization.
    
    Args:
        y_train: Training actuals
        y_train_pred: Training predictions
        y_test: Test actuals
        y_test_pred: Test predictions
        cutoff_date: Train/test split date
        symbol: Stock symbol
        metrics: Evaluation metrics
        model_name: Name of the model (e.g., "XGBoost", "Ridge")
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Time series
    ax1 = axes[0, 0]
    ax1.plot(y_train.index, y_train.values, 'o-', color='blue', alpha=0.6,
             label='Actual (Train)', markersize=6, linewidth=2)
    ax1.plot(y_train.index, y_train_pred, 's--', color='lightblue', alpha=0.8,
             label='Predicted (Train)', markersize=5, linewidth=1.5)
    ax1.plot(y_test.index, y_test.values, 'o-', color='red', alpha=0.8,
             label='Actual (Test)', markersize=8, linewidth=2)
    ax1.plot(y_test.index, y_test_pred, 's--', color='orange', alpha=0.9,
             label='Predicted (Test)', markersize=7, linewidth=2)
    ax1.axvline(cutoff_date, color='green', linestyle='--', linewidth=2.5,
                alpha=0.7, label='Train/Test Split')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Short Interest', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} - {model_name} Predictions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # In-sample scatter
    ax2 = axes[0, 1]
    ax2.scatter(y_train.values, y_train_pred, alpha=0.6, s=80,
                edgecolors='darkblue', linewidth=1.5, color='lightblue')
    min_val = min(y_train.values.min(), y_train_pred.min())
    max_val = max(y_train.values.max(), y_train_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, label='Perfect')
    ax2.set_xlabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_title(f'In-Sample ({model_name})\nRMSE: {metrics["train_rmse"]:.4f}, R²: {metrics["train_r2"]:.4f}',
                  fontsize=13, fontweight='bold', color='blue')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Out-of-sample scatter
    ax3 = axes[1, 0]
    ax3.scatter(y_test.values, y_test_pred, alpha=0.8, s=120,
                edgecolors='darkred', linewidth=2, color='orange')
    min_val = min(y_test.values.min(), y_test_pred.min())
    max_val = max(y_test.values.max(), y_test_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect')
    ax3.set_xlabel('Actual', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted', fontsize=12, fontweight='bold')
    ax3.set_title(f'Out-of-Sample ({model_name})\nRMSE: {metrics["test_rmse"]:.4f}, R²: {metrics["test_r2"]:.4f}',
                  fontsize=13, fontweight='bold', color='red')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Residuals
    ax4 = axes[1, 1]
    train_residuals = y_train.values - y_train_pred
    test_residuals = y_test.values - y_test_pred
    
    bp = ax4.boxplot([train_residuals, test_residuals],
                     positions=[1, 2],
                     widths=0.6,
                     patch_artist=True,
                     tick_labels=['Train', 'Test'])
    
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('orange')
    ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax4.set_title(f'Prediction Residuals ({model_name})', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def run_pipeline(
    stock_list: List[str],
    short_stock: str,
    use_llm: bool = True,
    use_images: bool = False,
    use_alpha_vantage: bool = False,
    n_trials: int = 100,
    n_passes: int = 4,
    max_leakage_retries: int = 3,
    use_pruning: bool = False
):
    """
    Execute short interest prediction pipeline with DUAL MODELS (XGBoost + Ridge).
    
    Args:
        stock_list: List of stock symbols for features
        short_stock: Target stock to predict short interest
        use_llm: Enable LLM feature generation
        use_images: Send visualizations to LLM
        use_alpha_vantage: Use Alpha Vantage instead of Bloomberg
        n_trials: Optuna optimization trials
        n_passes: Number of LLM feature generation passes
        max_leakage_retries: Retry attempts if leakage detected
        use_pruning: Enable feature pruning
    """
    print("=" * 60)
    print(f"SHORT INTEREST PREDICTION PIPELINE - DUAL MODEL")
    print(f"Models: XGBoost + Ridge Regression")
    print(f"LLM: {'Enabled' if use_llm else 'Disabled'}")
    print(f"Mode: {'Image' if use_images else 'Text'}")
    print(f"Passes: {n_passes}")
    print("=" * 60)
    
    # 1. Load target data
    print(f"\n1. Loading short interest data for {short_stock}...")
    short_df = get_finra_short_data(short_stock)
    target_idx = short_df.index
    print(f"  ✓ Found {len(short_df)} biweekly measurements")
    
    # 2. Load daily data
    print(f"\n2. Loading daily stock data...")
    stock_dict = {}
    
    for stock in tqdm(stock_list, desc="  Processing"):
        try:
            stock_dict[stock] = get_stock_data(stock, use_alpha_vantage=use_alpha_vantage)
        except Exception as e:
            print(f"  ⚠ Skipping {stock}: {e}")
    
    if not stock_dict:
        raise RuntimeError("No stock data loaded")
    
    # Build master DataFrame
    daily_master = create_daily_master_df(stock_dict=stock_dict)
    print(f"  ✓ Master DataFrame: {daily_master.shape}")
    
    # 3. Generate visualizations if image mode
    if use_llm and use_images:
        visualize_data_for_llm(daily_master, stock_list, short_stock)
    
    # 4. LLM feature generation (multi-pass)
    if not use_llm:
        raise RuntimeError("This pipeline requires LLM mode (--use-llm)")
    
    os.makedirs("generated", exist_ok=True)
    feature_files = []
    last_error = ""
    feedback_md = ""
    must_keep = None
    avoid = None
    
    for p in range(1, n_passes + 1):
        out_file = os.path.join("generated", f"{short_stock}_features_pass{p}.py")
        
        # Retry logic for leakage detection
        for attempt in range(1, max_leakage_retries + 1):
            print(f"\n3.{p}.{attempt} Generating LLM features (pass {p}, attempt {attempt})...")
            
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
                use_images=use_images,
            )
            
            if ok and os.path.exists(out_file):
                break
            
            if err.startswith("validation_error:"):
                print(f"  ⚠ Validation failed: {err}")
                last_error = f"PREVIOUS ERROR: {err.replace('validation_error:', '')}. Fix this issue."
                continue
            else:
                break
        
        if not (ok and os.path.exists(out_file)):
            raise RuntimeError(f"Feature generation failed at pass {p} after {max_leakage_retries} attempts")
        
        feature_files.append(out_file)
        
        # 5. Build features
        print(f"\n4.{p} Building feature set for pass {p}...")
        features_df = create_full_feature_set(
            daily_df=daily_master,
            target_dates=target_idx,
            lag_days=DEFAULT_LAG_DAYS,
            feature_files=feature_files,
        )
        
        if features_df.empty:
            raise RuntimeError(f"Empty feature matrix at pass {p}. Check that LLM generated valid 'feature_*' functions.")
        
        # 6. Train/test split
        cut_idx = max(int(len(target_idx) * (1 - TEST_SIZE)), 1)
        data = features_df.join(short_df['short_interest'], how='inner').sort_index()
        
        # Add safe lag features
        data['si_lag1'] = data['short_interest'].shift(1)
        data['si_lag2'] = data['short_interest'].shift(2)
        data['si_logdiff_lag1'] = (
            np.log1p(data['short_interest'].shift(1)) -
            np.log1p(data['short_interest'].shift(2))
        )
        
        train_data = data.iloc[:cut_idx]
        test_data = data.iloc[cut_idx:]
        
        X_train = train_data.drop(columns=['short_interest'])
        y_train = train_data['short_interest']
        X_test = test_data.drop(columns=['short_interest'])
        y_test = test_data['short_interest']
        
        cutoff_date = train_data.index[-1]
        
        # ========== TRAIN BOTH MODELS ==========
        
        # 7a. Train XGBoost
        print(f"\n5.{p}a Training XGBoost model for pass {p}...")
        xgb_model, xgb_params, xgb_cols, xgb_medians = train_xgboost_with_optuna(
            X_train, y_train, n_trials=n_trials, use_pruning=use_pruning
        )
        
        # 7b. Train Ridge
        print(f"\n5.{p}b Training Ridge model for pass {p}...")
        ridge_model, ridge_params, ridge_cols, ridge_medians = train_ridge_with_optuna(
            X_train, y_train, n_trials=n_trials, use_pruning=use_pruning
        )
        
        # 8a. Evaluate XGBoost
        print(f"\n6.{p}a Evaluating XGBoost model for pass {p}...")
        xgb_metrics = evaluate_xgboost(xgb_model, X_train, y_train, X_test, y_test, xgb_cols, xgb_medians)
        
        # 8b. Evaluate Ridge
        print(f"\n6.{p}b Evaluating Ridge model for pass {p}...")
        ridge_metrics = evaluate_ridge(ridge_model, X_train, y_train, X_test, y_test, ridge_cols, ridge_medians)
        
        # 9. Feedback for next pass (use XGBoost for feature importance)
        if p < n_passes:
            print(f"\n7.{p} Computing feedback for pass {p+1}...")
            try:
                gain_df, perm_df = compute_feature_importances(
                    xgb_model, X_train[xgb_cols], y_train, xgb_cols
                )
                
                feedback_md, must_keep, avoid = summarize_feedback_for_llm(
                    gain_df, perm_df, X_train
                )
                
                last_error = ""
            except Exception as e:
                last_error = str(e)
                feedback_md, must_keep, avoid = "", None, None
        
        # 10. Plot both models
        print(f"\n8.{p} Saving prediction visualizations for pass {p}...")
        
        # Plot XGBoost
        plot_predictions(
            y_train, xgb_metrics['train_pred'],
            y_test, xgb_metrics['test_pred'],
            cutoff_date, short_stock, xgb_metrics,
            model_name="XGBoost",
            save_path=f"{OUTPUT_DIR}/{short_stock}_xgboost_predictions_pass{p}.png"
        )
        
        # Plot Ridge
        plot_predictions(
            y_train, ridge_metrics['train_pred'],
            y_test, ridge_metrics['test_pred'],
            cutoff_date, short_stock, ridge_metrics,
            model_name="Ridge",
            save_path=f"{OUTPUT_DIR}/{short_stock}_ridge_predictions_pass{p}.png"
        )
    
    # Cleanup
    if use_images:
        clear_generated_artifacts(stock_list, short_stock)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE - BOTH MODELS TRAINED")
    print("="*60)
    print(f"XGBoost predictions saved to: {OUTPUT_DIR}/{short_stock}_xgboost_predictions_pass*.png")
    print(f"Ridge predictions saved to: {OUTPUT_DIR}/{short_stock}_ridge_predictions_pass*.png")

# ============================================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Stock Short Interest Prediction Pipeline - Dual Model (XGBoost + Ridge)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_pipeline.py --use-llm
  python full_pipeline.py --use-llm --use-images
  python full_pipeline.py --use-llm --use-alpha-vantage --n-trials 20
"""
    )
    
    parser.add_argument('--use-llm', action='store_true',
                        help='Enable LLM feature generation (required)')
    parser.add_argument('--use-images', action='store_true',
                        help='Send visualizations to LLM (uses more tokens)')
    parser.add_argument('--use-alpha-vantage', action='store_true',
                        help='Use Alpha Vantage API instead of Bloomberg file')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optuna optimization trials (default: 100)')
    parser.add_argument('--n-passes', type=int, default=3,
                        help='Number of LLM feature generation passes (default: 3)')
    parser.add_argument('--use-pruning', action='store_true',
                        help='Enable feature pruning before training (default: disabled)')
    
    args = parser.parse_args()
    
    # User input for market index
    print("Available market indices: SPX (S&P 500), RUI (Russell 1000), RUA (Russell 3000)")
    market_index = input("Enter market index to use (default: SPX): ").strip().upper()
    if not market_index:
        market_index = "SPX"
    
    if market_index not in ["SPX", "RUI", "RUA"]:
        print(f"Warning: {market_index} not recognized. Using SPX as default.")
        market_index = "SPX"
    print(f"Using market index: {market_index}\n")

    stock_list = input("Enter stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").split(',')
    stock_list = [s.strip().upper() for s in stock_list]

# Add selected market index to stock list
    if market_index not in stock_list:
        stock_list.append(market_index)
        print(f"Added {market_index} to feature stock list")
        
        short_stock = input("Enter stock to predict short interest for (e.g., TSLA): ").strip().upper()
        
        # Run pipeline
        run_pipeline(
            stock_list=stock_list,
            short_stock=short_stock,
            use_llm=args.use_llm,
            use_images=args.use_images,
            use_alpha_vantage=args.use_alpha_vantage,
            n_trials=args.n_trials,
            n_passes=args.n_passes,
            use_pruning=args.use_pruning
        )