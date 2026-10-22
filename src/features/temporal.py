
import numpy as np
import pandas as pd


def create_features(ndvi_series, ndmi_series):
    """Create time-series features from vegetation indices."""
    df = pd.DataFrame({
        "ndvi": np.asarray(ndvi_series, dtype=float),
        "ndmi": np.asarray(ndmi_series, dtype=float),
    })

    df["ndvi_change"] = df["ndvi"].diff()
    df["ndmi_change"] = df["ndmi"].diff()
    df["ndvi_rolling_mean"] = df["ndvi"].rolling(
        window=3, min_periods=1
    ).mean()

    return df
