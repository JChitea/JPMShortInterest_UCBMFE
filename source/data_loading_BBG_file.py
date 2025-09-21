import pandas as pd
import numpy as np


def parse_bloomberg_with_ticker_dates(file_path, sheet_name=0):
    """
    Parse Bloomberg data where each ticker has its own Date column
    Structure: Ticker1_Date | Ticker1_Fields | Ticker2_Date | Ticker2_Fields | etc.
    """
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    # Find the row with multiple "Date" columns (field header row)
    date_header_row = None
    for i, row in df_raw.iterrows():
        date_columns = [j for j, val in enumerate(row) if pd.notna(val) and str(val).strip() == 'Date']
        if len(date_columns) >= 2:
            date_header_row = i
            print(f"Found {len(date_columns)} Date columns at row {i}")
            break
    
    # Find ticker row (usually 1-2 rows before date headers)
    ticker_row = None
    if date_header_row is not None:
        for i in range(max(0, date_header_row - 3), date_header_row):
            row = df_raw.iloc[i]
            ticker_count = sum(1 for val in row if pd.notna(val) and 
                             ('US Equity' in str(val) or 'Index' in str(val)))
            if ticker_count >= 2:
                ticker_row = i
                break
    
    # Get date column positions
    date_columns = [j for j, val in enumerate(df_raw.iloc[date_header_row]) 
                   if pd.notna(val) and str(val).strip() == 'Date']
    
    # Match tickers to date columns
    ticker_names = []
    for date_col in date_columns:
        # Find ticker name near this date column
        ticker_found = False
        for offset in range(-2, 3):
            check_pos = date_col + offset
            if 0 <= check_pos < len(df_raw.columns):
                ticker_val = df_raw.iloc[ticker_row, check_pos]
                if pd.notna(ticker_val) and ('US Equity' in str(ticker_val) or 'Index' in str(ticker_val)):
                    ticker_names.append(str(ticker_val))
                    ticker_found = True
                    break
        if not ticker_found:
            ticker_names.append(f"Unknown_Ticker_{len(ticker_names)}")
    
    print(f"Matched date columns with tickers:")
    for date_col, ticker in zip(date_columns, ticker_names):
        print(f"  Column {date_col}: {ticker}")
    
    # Extract data rows
    data_rows = []
    data_start_row = date_header_row + 1
    
    for row_idx in range(data_start_row, len(df_raw)):
        row_data = df_raw.iloc[row_idx]
        
        # Process each ticker's section
        for date_col_idx, (date_col, ticker) in enumerate(zip(date_columns, ticker_names)):
            # Get date from this ticker's Date column
            ticker_date = None
            if date_col < len(row_data):
                date_val = row_data.iloc[date_col]
                if pd.notna(date_val):
                    try:
                        ticker_date = pd.to_datetime(date_val)
                    except:
                        pass
            
            # Only process if we have a valid date
            if ticker_date is None:
                continue
            
            # Determine column range for this ticker
            next_date_col = (date_columns[date_col_idx + 1] 
                           if date_col_idx + 1 < len(date_columns) 
                           else len(row_data))
            
            # Extract field data
            ticker_data = {
                'Date': ticker_date,
                'Ticker': ticker
            }
            
            # Bloomberg fields in typical order after Date column
            field_names = ['PX_LAST', 'PX_VOLUME', 'DVD_SH_LAST', 'DVD_EX_DT', 
                          'PX_BID', 'PX_ASK', 'OPEN_INT_TOTAL_PUT', 'OPEN_INT_TOTAL_CALL']
            
            has_data = False
            for field_idx, field_name in enumerate(field_names):
                col_pos = date_col + 1 + field_idx  # +1 to skip Date column
                if col_pos < next_date_col and col_pos < len(row_data):
                    value = row_data.iloc[col_pos]
                    if pd.notna(value) and str(value) not in ['#N/A N/A', '#NAME?']:
                        ticker_data[field_name] = value
                        has_data = True
            
            # Add row if it has meaningful data
            if has_data:
                data_rows.append(ticker_data)
    
    # Create and clean DataFrame
    if data_rows:
        df = pd.DataFrame(data_rows)
        
        # Convert numeric columns
        numeric_cols = ['PX_LAST', 'PX_VOLUME', 'PX_BID', 'PX_ASK', 
                       'OPEN_INT_TOTAL_PUT', 'OPEN_INT_TOTAL_CALL', 'DVD_SH_LAST']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'DVD_EX_DT' in df.columns:
            df['DVD_EX_DT'] = pd.to_datetime(df['DVD_EX_DT'], errors='coerce')
        
        print(f"\nCreated DataFrame with {len(df)} rows")
        print(f"Unique tickers: {df['Ticker'].nunique()}")
        print(f"Overall date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Show date range by ticker (since they start at different times)
        date_summary = (df.groupby('Ticker')['Date']
                       .agg(['min', 'max', 'count'])
                       .reset_index())
        print("\nDate ranges by ticker:")
        print(date_summary)
        
        return df
    
    return pd.DataFrame()
