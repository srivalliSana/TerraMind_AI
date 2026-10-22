
import os
import earthaccess
import rasterio


def search_hls_data(start_date, end_date, count=5):
    return earthaccess.search_data(
        short_name="HLSL30",
        temporal=(start_date, end_date),
        count=count,
    )


def download_hls_bands(granule, output_dir="data/raw"):
    """Download Red, NIR and SWIR bands for one HLS granule."""
    os.makedirs(output_dir, exist_ok=True)

    links = granule.data_links()
    bands = [".B04.", ".B05.", ".B06.", ".B07."]

    selected = [
        url for url in links
        if any(band in url.rsplit("/", 1)[-1] for band in bands)
    ]

    if not selected:
        raise ValueError("No required HLS band files found.")

    downloaded = earthaccess.download(
        selected,
        local_path=output_dir
    )

    return downloaded


def load_raster(file_path):
    """Read a single-band GeoTIFF raster."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    return data, profile, nodata
