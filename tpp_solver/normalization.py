"""Pure intensity-normalization helpers (no Streamlit dependency)."""
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.stats.mstats import winsorize

def normalize_by_median(values):
    """
    Normalize values by dividing by the median.
    Robust to outliers and loading differences.
    
    Args:
        values: numpy array of values to normalize
        
    Returns:
        numpy array of normalized values
    """
    if len(values) == 0:
        return values
        
    # Handle NaN values
    values = np.nan_to_num(values, nan=0.0)
    
    # Get positive values for median calculation
    positive_values = values[values > 0]
    if len(positive_values) == 0:
        return values
        
    median = np.median(positive_values)
    if median <= 0:
        return values
        
    # Normalize while preserving zeros
    normalized = np.zeros_like(values)
    nonzero_mask = values > 0
    normalized[nonzero_mask] = values[nonzero_mask] / median
    
    return normalized

def normalize_by_robust_zscore(values):
    """
    Normalize using robust Z-score (median and MAD).
    Highly resistant to outliers.
    
    Args:
        values: numpy array of values to normalize
        
    Returns:
        numpy array of normalized values
    """
    if len(values) == 0:
        return values
        
    # Handle NaN values
    values = np.nan_to_num(values, nan=0.0)
    
    median = np.median(values)
    mad = median_abs_deviation(values, nan_policy='omit')
    
    if mad <= 0:
        return values
        
    # 1.4826 makes MAD consistent with std dev for normal distribution
    return (values - median) / (1.4826 * mad)

def normalize_by_quantile(values):
    """
    Quantile normalization to make distributions identical.
    Good for removing systematic biases.
    
    Args:
        values: numpy array of values to normalize
        
    Returns:
        numpy array of normalized values
    """
    if len(values) == 0:
        return values
        
    # Handle NaN values
    values = np.nan_to_num(values, nan=0.0)
    
    # Handle case where all values are the same
    if np.all(values == values[0]):
        return values
        
    sorted_values = np.sort(values)
    ranks = pd.Series(values).rank(method='average', na_option='keep')
    
    # Preserve zeros in output
    normalized = np.zeros_like(values)
    nonzero_mask = values > 0
    if np.any(nonzero_mask):
        normalized[nonzero_mask] = np.interp(
            ranks[nonzero_mask],
            np.arange(1, len(values[nonzero_mask]) + 1),
            sorted_values[nonzero_mask]
        )
    
    return normalized

def normalize_by_winsorization(values, limits=(0.05, 0.95)):
    """
    Winsorize extreme values to reduce impact of outliers.
    Preserves data structure while limiting extreme values.
    
    Args:
        values: numpy array of values to normalize
        limits: tuple of (lower, upper) percentile limits
        
    Returns:
        numpy array of normalized values
    """
    if len(values) == 0:
        return values
        
    # Handle NaN values
    values = np.nan_to_num(values, nan=0.0)
    
    # Preserve zeros
    nonzero_mask = values > 0
    if not np.any(nonzero_mask):
        return values
        
    # Only winsorize non-zero values
    nonzero_values = values[nonzero_mask]
    winsorized_nonzero = winsorize(nonzero_values, limits=limits)
    
    # Put winsorized values back
    normalized = values.copy()
    normalized[nonzero_mask] = winsorized_nonzero
    
    return normalized
