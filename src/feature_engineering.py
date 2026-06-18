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

def add_lab_features(df, labs):
    """
    Add features for sparse lab values: forward-filled last known value
    plus a binary flag indicating whether the lab was recently measured.
    
    Args:
        df: DataFrame sorted by Patient_ID and Hour
        labs: list of lab column names (e.g. Lactate, Creatinine, WBC)
    
    Returns:
        DataFrame with new lab feature columns added
    """
    df = df.sort_values(['Patient_ID', 'Hour']).copy()
    
    for col in labs:
        grouped = df.groupby('Patient_ID')[col]
        
        # Was this lab measured at this hour? (before any fill)
        df[f'{col}_was_measured'] = grouped.transform(lambda x: x.notna().astype(int))
        
        # Carry forward the last known value per patient
        df[f'{col}_last_known'] = grouped.transform(lambda x: x.ffill())
        
        # How many hours since this lab was last measured?
        df[f'{col}_hours_since_measured'] = grouped.transform(
            lambda x: x.notna().astype(int).cumsum()
        )
        df[f'{col}_hours_since_measured'] = df.groupby(
            ['Patient_ID', f'{col}_hours_since_measured']
        ).cumcount()
    
    return df