import numpy as np


def calculate_ndvi(nir, red):
    """Calculate NDVI from Near-Infrared and Red reflectance."""
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)

    denominator = nir + red
    ndvi = np.full(nir.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(nir) & np.isfinite(red) & (denominator != 0)
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]

    return ndvi

def calculate_ndmi(nir, swir):
    """Calculate NDMI using Near-Infrared and SWIR reflectance."""
    nir = np.asarray(nir, dtype=np.float32)
    swir = np.asarray(swir, dtype=np.float32)

    denominator = nir + swir
    ndmi = np.full(nir.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(nir) & np.isfinite(swir) & (denominator != 0)
    ndmi[valid] = (nir[valid] - swir[valid]) / denominator[valid]

    return ndmi

if __name__ == "__main__":
    nir = np.array([0.8, 0.6, 0.3])
    red = np.array([0.2, 0.3, 0.2])
    swir = np.array([0.1, 0.2, 0.15])

    print("NDVI:", calculate_ndvi(nir, red))
    print("NDMI:", calculate_ndmi(nir, swir))
