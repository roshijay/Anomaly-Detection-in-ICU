import pandas as pd
import numpy as np


def add_rolling_features(df, vitals, window=6):
    """
    Add rolling window features (mean, std, min, max, trend) for each vital sign.
    
    Args:
        df: DataFrame sorted by Patient_ID and Hour
        vitals: list of column names to compute rolling features for
        window: number of hours to look back
    
    Returns:
        DataFrame with new rolling feature columns added
    """
    df = df.sort_values(['Patient_ID', 'Hour']).copy()
    
    for col in vitals:
        grouped = df.groupby('Patient_ID')[col]
        
        df[f'{col}_rolling_mean'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{col}_rolling_std'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f'{col}_rolling_min'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f'{col}_rolling_max'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df[f'{col}_trend'] = grouped.transform(
            lambda x: x - x.shift(window)
        )
    
    return df