"""
Activity-based features for CSISD.

Defines:
    - compute_activity(...)
    - compute_activity_features(...)
    - compute_activity_threshold(...)
    - compute_activity_mask(...)
"""

import numpy as np
from .derivatives import compute_derivatives


def _moving_average(x, window_size):
    """
    Simple centered moving average.
    For edges, keep NaN where there is not enough context.
    """
    if window_size is None or window_size <= 1:
        return x

    x = np.asarray(x, float)
    n = len(x)
    y = np.full_like(x, np.nan, dtype=float)

    # only use finite values
    mask = np.isfinite(x)
    if not np.any(mask):
        return y

    # convolution with ones, then normalized
    kernel = np.ones(window_size, dtype=float)
    valid = np.convolve(mask.astype(float), kernel, mode="same")
    smoothed = np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        y = smoothed / valid
    # on positions with no valid samples, keep NaN
    y[valid == 0] = np.nan
    return y


def compute_activity(pitch_cents, time_sec, alpha=0.5, smooth_window=None):
    """
    Compute a scalar 'activity' contour from pitch_cents:

        activity[n] = |d1[n]| + alpha * |d2[n]|

    where:
        d1 : first derivative (cents/s)
        d2 : second derivative (cents/s^2)

    Optionally smooth the activity with a small moving average.

    Parameters
    ----------
    pitch_cents : array-like
    time_sec : array-like
    alpha : float
        Weight for the second derivative term.
    smooth_window : int or None
        If not None and >1, apply a moving average of this window size.

    Returns
    -------
    activity : np.ndarray
        Activity values (same length as pitch_cents).
    d1 : np.ndarray
    d2 : np.ndarray
    """
    derivs = compute_derivatives(pitch_cents, time_sec)
    d1 = derivs["deriv1"]
    d2 = derivs["deriv2"]

    # activity scalar
    with np.errstate(invalid="ignore"):
        activity = np.abs(d1) + alpha * np.abs(d2)

    if smooth_window is not None and smooth_window > 1:
        activity = _moving_average(activity, smooth_window)

    return activity, d1, d2


def compute_activity_features(activity):
    """
    Simple summary stats of an activity contour:

        - activity_mean
        - activity_std
        - activity_max
        - activity_p90  (90th percentile)

    Parameters
    ----------
    activity : array-like

    Returns
    -------
    dict
    """
    a = np.asarray(activity, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            "activity_mean": np.nan,
            "activity_std": np.nan,
            "activity_max": np.nan,
            "activity_p90": np.nan,
        }

    activity_mean = float(np.mean(a))
    activity_std  = float(np.std(a))
    activity_max  = float(np.max(a))
    activity_p90  = float(np.percentile(a, 90.0))

    return {
        "activity_mean": activity_mean,
        "activity_std": activity_std,
        "activity_max": activity_max,
        "activity_p90": activity_p90,
    }


def compute_activity_threshold(activity, quantile=0.2):
    """
    Compute a threshold for 'active' vs 'stable' based on a quantile.

    Parameters
    ----------
    activity : array-like
    quantile : float
        e.g. 0.2 for the 20th percentile.

    Returns
    -------
    float
        Threshold value.
    """
    a = np.asarray(activity, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    q = float(np.clip(quantile, 0.0, 1.0))
    return float(np.percentile(a, q * 100.0))


def compute_activity_mask(activity, threshold):
    """
    Binary mask of 'active' positions:

        mask[n] = activity[n] > threshold

    Parameters
    ----------
    activity : array-like
    threshold : float

    Returns
    -------
    np.ndarray of bool
    """
    a = np.asarray(activity, float)
    with np.errstate(invalid="ignore"):
        mask = a > threshold
    # ensure boolean, NaN -> False
    mask = np.where(np.isfinite(a), mask, False)
    return mask




# fes un zip entre dues llistes random
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
zipped = zip(list1, list2)
print(list(zipped))






'''
import polars as pl
import pandas as pd
from pathlib import Path
from src.annotations.utils import time_str_to_sec


def load_annotation_tsv(
        file_path: Path | str | None,
        recording_id: str | None = None,
        engine='polars',
        sep='\t',
        annotation_type: str = 'svara',
        column_names=None
):
    """

    """
    if file_path is None and recording_id is None:
        raise ValueError("Either file_path or recording_id must be provided.")

    if file_path is not None:

        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        ext = file_path.suffix.lower()


        if engine == 'polars':

            if ext == '.parquet':
                df = pl.read_parquet(file_path)

            elif ext == '.tsv':
                if column_names is None:
                    df = pl.read_csv(file_path, separator=sep, has_header=True)

                else:
                    df = pl.read_csv(file_path, separator=sep, has_header=False)
                    df = df.rename({old: new for old, new in zip(df.columns, column_names)})


            else:
                raise ValueError(f"Unsupported file extension: {ext}")


        elif engine == 'pandas':

            if ext == '.parquet':
                df = pd.read_parquet(file_path)

            elif ext == '.tsv':
                if column_names is None:
                    df = pd.read_csv(file_path, sep=sep, header=0)

                else:
                    df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)

            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            raise ValueError("Engine must be 'polars' or 'pandas'.")
    
    else:
        
        if isinstance(recording_id, str):

            dir_path = Path('data' / 'corpus' / recording_id / 'annotations')
            path_end = f'ann_{annotation_type}.tsv'

            for file in dir_path.iterdir().rglob('*_ann_*'):
                if str(file).endswith(path_end):
                
                    file_path = Path(file)
                    if engine == 'polars':

                        if ext == '.parquet':
                            df = pl.read_parquet(file_path)

                        elif ext == '.tsv':
                            if column_names is None:
                                df = pl.read_csv(file_path, separator=sep, has_header=True)

                            else:
                                df = pl.read_csv(file_path, separator=sep, has_header=False)
                                df = df.rename({old: new for old, new in zip(df.columns, column_names)})


                        else:
                            raise ValueError(f"Unsupported file extension: {ext}")


                    elif engine == 'pandas':
                    
                        if ext == '.parquet':
                            df = pd.read_parquet(file_path)

                        elif ext == '.tsv':
                            if column_names is None:
                                df = pd.read_csv(file_path, sep=sep, header=0)

                            else:
                                df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)

                        else:
                            raise ValueError(f"Unsupported file extension: {ext}")

                    else:
                        raise ValueError("Engine must be 'polars' or 'pandas'.")
                    

    return df



'''








'''import polars as pl
import pandas as pd
from pathlib import Path
from src.annotations.utils import time_str_to_sec


def load_annotation_tsv(
        file_path: Path | str | None,
        recording_id: str | None = None,
        engine='polars',
        sep='\t',
        annotation_type: str = 'svara_marks',
):
    """

    """
    if file_path is None and recording_id is None:
        raise ValueError("Either file_path or recording_id must be provided.")

    if file_path is not None:

        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        ext = file_path.suffix.lower()


        if engine == 'polars':

            if ext == '.parquet':
                df = pl.read_parquet(file_path)
                df['Begin time'] = time_str_to_sec(df['Begin Time'])
                df['End time'] = time_str_to_sec(df['End Time'])

            elif ext == '.tsv':
                if column_names is None:
                    df = pl.read_csv(file_path, separator=sep, has_header=True)
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)
                else:
                    df = pl.read_csv(file_path, separator=sep, has_header=False)
                    df = df.rename({old: new for old, new in zip(df.columns, column_names)})
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)

            else:
                raise ValueError(f"Unsupported file extension: {ext}")


        elif engine == 'pandas':

            if ext == '.parquet':
                df = pd.read_parquet(file_path)
                df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                df['End time'] = df['End Time'].apply(time_str_to_sec)

            elif ext == '.tsv':
                if column_names is None:
                    df = pd.read_csv(file_path, sep=sep, header=0)
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)
                else:
                    df = pd.read_csv(file_path, sep=sep, header=None, names=column_names)
                    df['Begin time'] = df['Begin Time'].apply(time_str_to_sec)
                    df['End time'] = df['End Time'].apply(time_str_to_sec)

            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        else:
            raise ValueError("Engine must be 'polars' or 'pandas'.")
    
    else:
        
        if isinstance(recording_id, str):

            dir_path = Path('data' / 'corpus' / recording_id / 'annotations')

            for file in dir_path.iterdir():
                for  tag in str(file).split('_'):
                    if tag == 'ann':
                        
                        break


    return df'''