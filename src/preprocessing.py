"""Data preprocessing module for loan approval dataset."""
import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import load_data
from .features import engineer_features

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    logger.info('Cleaning data: shape before %s', df.shape)
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    if 'loan_status' in df.columns:
        df = df.dropna(subset=['loan_status'])
    logger.info('Cleaning data: shape after %s', df.shape)
    return df


def main() -> None:
    """Main entry point for preprocessing module."""
    parser = argparse.ArgumentParser(description='Preprocess employee loan data')
    parser.add_argument('--input', type=Path, required=True, help='Path to raw CSV file')
    parser.add_argument('--output', type=Path, required=True, help='Path to save processed CSV')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    try:
        df = load_data(args.input)
        clean = clean_data(df)
        processed = engineer_features(clean)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(args.output, index=False)
        logger.info('Saved processed data to %s', args.output)
    except Exception as e:
        logger.error('Preprocessing failed: %s', e)
        raise


if __name__ == '__main__':
    main()
