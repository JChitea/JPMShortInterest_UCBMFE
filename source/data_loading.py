import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tolerance settings
RECOMMENDATION_TOLERANCE = pd.Timedelta(days=90)
FUNDAMENTALS_TOLERANCE = pd.Timedelta(days=180)

# Column mappings
SHORT_INTEREST_COLUMNS = {
    'settlementDate': 'date',
    'symbolCode': 'TSYMBOL'
}

IBES_COLUMNS = {
    'ANNDATS': 'date'
}

COMPUSTAT_COLUMNS = {
    'datadate': 'date',
    'cusip': 'CUSIP'
}


def safe_load_csv(filepath: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Safely load CSV with comprehensive error handling."""
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully loaded {filepath.name} with {len(df):,} rows")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def merge_csvs_safely(folder_path: Path) -> Optional[pd.DataFrame]:
    """
    Merge CSV files from a folder with error handling.
    
    Args:
        folder_path: Path to folder containing CSV files
        
    Returns:
        Merged DataFrame or None if no files found/loaded
    """
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return None
        
    csv_files = list(folder_path.glob('*.csv'))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {folder_path}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files to merge")
    
    dataframes: List[pd.DataFrame] = []
    
    for file_path in csv_files:
        df = safe_load_csv(
            file_path, 
            sep='|', 
            parse_dates=['settlementDate']
        )
        if df is not None:
            dataframes.append(df)
    
    if not dataframes:
        logger.error("No CSV files could be loaded successfully")
        return None
    
    merged_df = (pd.concat(dataframes, ignore_index=True)
                   .sort_values(['settlementDate', 'symbolCode'])
                   .reset_index(drop=True))
    
    logger.info(f"Successfully merged {len(dataframes)} files into {len(merged_df):,} rows")
    return merged_df


def load_and_prepare_crsp(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load and clean CRSP market data.
    
    Args:
        filepath: Path to CRSP CSV file
        
    Returns:
        Cleaned and sorted DataFrame or None
    """
    df = safe_load_csv(filepath, parse_dates=['date'])
    
    if df is None:
        return None
    
    # Clean and prepare data
    cleaned_df = (df.drop(columns=['DIVAMT'], errors='ignore')  # Ignore if column doesn't exist
                   .dropna(subset=['PRC'])
                   .drop_duplicates()
                   .sort_values('date')
                   .reset_index(drop=True))
    
    logger.info(f"CRSP data cleaned: {len(cleaned_df):,} rows after filtering")
    return cleaned_df


def load_and_aggregate_recommendations(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load and aggregate IBES recommendations data.
    
    Args:
        filepath: Path to IBES CSV file
        
    Returns:
        Aggregated recommendations DataFrame or None
    """
    df = safe_load_csv(filepath, parse_dates=['ANNDATS'])
    
    if df is None:
        return None
    
    # Rename columns and aggregate
    df_renamed = df.rename(columns=IBES_COLUMNS)
    
    consensus_recs = (df_renamed.groupby(['date', 'CUSIP'])
                               .agg({
                                   'IRECCD': 'mean',
                                   'ANALYST': 'count'
                               })
                               .rename(columns={
                                   'IRECCD': 'consensus_rec',
                                   'ANALYST': 'analyst_count'
                               })
                               .reset_index()
                               .sort_values('date')
                               .reset_index(drop=True))
    
    logger.info(f"Recommendations aggregated: {len(consensus_recs):,} consensus records")
    return consensus_recs


def load_and_prepare_fundamentals(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load and prepare Compustat fundamentals data.
    
    Args:
        filepath: Path to Compustat CSV file
        
    Returns:
        Prepared fundamentals DataFrame or None
    """
    df = safe_load_csv(filepath, parse_dates=['datadate'])
    
    if df is None:
        return None
    
    prepared_df = (df.rename(columns=COMPUSTAT_COLUMNS)
                    .sort_values('date')
                    .reset_index(drop=True))
    
    logger.info(f"Fundamentals data prepared: {len(prepared_df):,} rows")
    return prepared_df


def merge_with_tolerance(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on_column: str,
    by_column: str,
    tolerance: Optional[pd.Timedelta] = None,
    data_type: str = "data"
) -> pd.DataFrame:
    """
    Generic function for time-series merging with tolerance.
    
    Args:
        left_df: Left DataFrame (already sorted)
        right_df: Right DataFrame (already sorted)
        on_column: Column to merge on (typically 'date')
        by_column: Column to match by (e.g., 'TSYMBOL', 'CUSIP')
        tolerance: Maximum time difference allowed
        data_type: Description for logging
        
    Returns:
        Merged DataFrame
    """
    initial_rows = len(left_df)
    
    merged_df = pd.merge_asof(
        left_df,
        right_df,
        on=on_column,
        by=by_column,
        direction='backward',
        tolerance=tolerance
    )
    
    if tolerance:
        matched_rows = merged_df[right_df.columns.difference([on_column, by_column])].notna().any(axis=1).sum()
        logger.info(f"Merged {data_type}: {matched_rows:,} of {initial_rows:,} rows matched within {tolerance}")
    else:
        logger.info(f"Merged {data_type}: {len(merged_df):,} rows")
    
    return merged_df


def build_panel_dataset(
    data_folder: Path = Path('Short Interest Data'),
    crsp_file: Path = Path('CRSP Market Data 2.csv'),
    ibes_file: Path = Path('IBES Recommendations.csv'),
    compustat_file: Path = Path('Compustat Fundamentals.csv')
) -> Optional[pd.DataFrame]:
    """
    Build the complete panel dataset by merging all data sources.
    
    Returns:
        Complete panel DataFrame or None if critical data missing
    """
    logger.info("Starting panel dataset construction...")
    
    # 1. Load short interest data
    short_interest = merge_csvs_safely(data_folder)
    if short_interest is None:
        logger.error("Failed to load short interest data")
        return None
    
    # Rename columns for consistency
    short_interest_clean = short_interest.rename(columns=SHORT_INTEREST_COLUMNS)
    
    # 2. Load CRSP data
    crsp = load_and_prepare_crsp(crsp_file)
    if crsp is None:
        logger.error("Failed to load CRSP data")
        return None
    
    # 3. Merge CRSP with short interest
    panel_data = merge_with_tolerance(
        crsp,
        short_interest_clean,
        on_column='date',
        by_column='TSYMBOL',
        data_type="short interest"
    )
    
    # 4. Load and merge recommendations
    recommendations = load_and_aggregate_recommendations(ibes_file)
    if recommendations is not None:
        panel_data = merge_with_tolerance(
            panel_data,
            recommendations,
            on_column='date',
            by_column='CUSIP',
            tolerance=RECOMMENDATION_TOLERANCE,
            data_type="recommendations"
        )
    else:
        logger.warning("Skipping recommendations merge due to load failure")
    
    # 5. Load and merge fundamentals
    fundamentals = load_and_prepare_fundamentals(compustat_file)
    if fundamentals is not None:
        panel_data = merge_with_tolerance(
            panel_data,
            fundamentals,
            on_column='date',
            by_column='CUSIP',
            tolerance=FUNDAMENTALS_TOLERANCE,
            data_type="fundamentals"
        )
    else:
        logger.warning("Skipping fundamentals merge due to load failure")
    
    logger.info(f"Panel dataset construction complete: {len(panel_data):,} rows, {len(panel_data.columns)} columns")
    return panel_data


def main(
    data_folder: Path = Path('Short Interest Data'),
    crsp_file: Path = Path('CRSP Market Data 2.csv'),
    ibes_file: Path = Path('IBES Recommendations.csv'),
    compustat_file: Path = Path('Compustat Fundamentals.csv')
):
    """Main execution function."""
    try:
        # Build the complete panel dataset
        panel_df = build_panel_dataset(data_folder, crsp_file, ibes_file, compustat_file)
        
        if panel_df is not None:
            print("\n" + "="*50)
            print("PANEL DATASET SUMMARY")
            print("="*50)
            print(f"Shape: {panel_df.shape}")
            print(f"Date range: {panel_df['date'].min()} to {panel_df['date'].max()}")
            print(f"Unique securities: {panel_df['TSYMBOL'].nunique():,}")
            print("\nFirst few rows:")
            print(panel_df.head())
            
            # Optional: Save to file
            # panel_df.to_csv('panel_dataset.csv', index=False)
            # logger.info("Dataset saved to panel_dataset.csv")
            
            return panel_df
        else:
            logger.error("Failed to build panel dataset")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        return None


if __name__ == "__main__":
    
    merged_df = main()
