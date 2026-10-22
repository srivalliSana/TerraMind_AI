
import numpy as np


def clean_raster(data, nodata=None):
    """Replace invalid and NoData pixels with NaN."""
    data = data.astype(np.float32)

    invalid = ~np.isfinite(data)

    if nodata is not None:
        invalid |= data == nodata

    data[invalid] = np.nan
    return data
