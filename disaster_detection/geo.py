"""Pixel -> longitude/latitude helpers for xBD geotransforms and GeoTIFF tags."""
import numpy as np


def apply_gdal_transform(transform, col, row):
    x0, dx, rx, y0, ry, dy = transform
    return x0 + col * dx + row * rx, y0 + col * ry + row * dy


def geotiff_transform(path):
    """Return (GDAL-style transform, EPSG code) from a GeoTIFF's tags, or (None, None)."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        if 33922 not in tags or 33550 not in tags:
            return None, None
        tie = tags[33922].value  # (i, j, k, x, y, z)
        scale = tags[33550].value  # (sx, sy, sz)
        epsg = None
        if 34735 in tags:
            keys = tags[34735].value
            for n in range(1, int(keys[3]) + 1):
                key_id, _, _, value = keys[4 * n:4 * n + 4]
                if key_id in (3072, 2048) and value not in (0, 32767):  # projected / geographic CRS
                    epsg = int(value)
                    if key_id == 3072:
                        break
        transform = (tie[3] - tie[0] * scale[0], scale[0], 0.0, tie[4] + tie[1] * scale[1], 0.0, -scale[1])
        return transform, epsg


def to_lonlat(x, y, epsg):
    """Convert coordinates in `epsg` to WGS84 lon/lat (identity for EPSG:4326)."""
    if epsg in (None, 4326):
        return x, y
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y)
    return lon, lat


def pixel_lonlat(transform, epsg, cols, rows):
    """Vectorised pixel (col, row) -> (lon, lat)."""
    cols, rows = np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)
    x, y = apply_gdal_transform(transform, cols, rows)
    return to_lonlat(x, y, epsg)
