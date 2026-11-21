#!/usr/bin/env python3
"""Spatiotemporal interpolation of topographic change between two DEMs.

Minimal GDAL/NumPy/SciPy implementation of the method described in:
    Hayakawa, Y.S.; Wang, L. Spatiotemporal interpolation of topographic changes
    from terrestrial LIDAR observations: a controlled experiment with rolling
    objects on a slope. Remote Sensing (under review; update this reference).

Workflow
    1. Cut and fill between the DEM at t0 and the DEM at t_end gives the change
       bodies (erosion body = object at t0, deposition body = object at t_end)
       and the bed (immobile surface).
    2. The cells of both bodies become points (x, y, t) with the body height as
       attribute. A 3D Delaunay tessellation is built in (x, y, t) space and the
       height at any (x, y, t) is obtained with the tetrahedral shape function
       (barycentric coordinates, identical to the volume-ratio form).
    3. Optional physical reality constraints, applied at every time step:
       (i) range: change is set to zero outside a polygon/mask M;
       (ii) mass balance: the volume of the change body is forced to an expected
            value by equal redistribution (iterated until both constraints hold).
    4. The interpolated change body is added to the bed.

Copyright (c) 2023-2026 Wang Luyao and Yuichi S. Hayakawa. MIT License.
"""
import argparse
import os
import numpy as np
from scipy.spatial import Delaunay

try:  # GDAL Python bindings (preferred)
    from osgeo import gdal
    gdal.UseExceptions()
    _BACKEND = "gdal"
except ImportError:  # fallback, also GDAL-based
    gdal = None
    _BACKEND = "rasterio"


# ----------------------------------------------------------------------------
# raster I/O
# ----------------------------------------------------------------------------
def read_dem(path):
    """Return (array float64 with NaN for nodata, geotransform, projection_wkt)."""
    if gdal is not None:
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray().astype("float64")
        nd = band.GetNoDataValue()
        if nd is not None:
            a[a == nd] = np.nan
        return a, ds.GetGeoTransform(), ds.GetProjection()
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float64")
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        t = ds.transform
        gt = (t.c, t.a, t.b, t.f, t.d, t.e)
        return a, gt, ds.crs.to_wkt() if ds.crs else ""


def write_dem(path, array, gt, wkt="", nodata=-9999.0):
    a = np.where(np.isfinite(array), array, nodata).astype("float32")
    if gdal is not None:
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, a.shape[1], a.shape[0], 1, gdal.GDT_Float32)
        ds.SetGeoTransform(gt)
        if wkt:
            ds.SetProjection(wkt)
        b = ds.GetRasterBand(1)
        b.SetNoDataValue(nodata)
        b.WriteArray(a)
        ds.FlushCache()
        ds = None
        return
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                       count=1, dtype="float32", nodata=nodata,
                       transform=Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]),
                       crs=wkt or None) as ds:
        ds.write(a, 1)


def rasterize_polygon(vector_path, shape, gt):
    """Rasterize a polygon layer (any OGR format) to a boolean mask. Needs GDAL."""
    if gdal is None:
        raise RuntimeError("Vector range polygons need the GDAL Python bindings; "
                           "use --range-mask (raster) or --range-hull instead.")
    from osgeo import ogr
    src = ogr.Open(vector_path)
    mem = gdal.GetDriverByName("MEM").Create("", shape[1], shape[0], 1, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    gdal.RasterizeLayer(mem, [1], src.GetLayer(0), burn_values=[1])
    return mem.GetRasterBand(1).ReadAsArray() > 0


# ----------------------------------------------------------------------------
# method
# ----------------------------------------------------------------------------
def cell_centers(shape, gt):
    rows, cols = np.indices(shape)
    x = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
    y = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]
    return x, y


def extract_bodies(dem0, dem1, threshold=0.005):
    """Cut and fill. Returns erosion body, deposition body (heights >= 0) and bed.

    erosion body   : dem0 - dem1 > threshold  (object position at t0)
    deposition body: dem1 - dem0 > threshold  (object position at t_end)
    bed            : (dem0 + dem1 - erosion - deposition) / 2
    Differences within +-threshold are treated as noise (level of detection).
    """
    valid = np.isfinite(dem0) & np.isfinite(dem1)
    d = np.where(valid, dem0 - dem1, 0.0)
    ero = np.where(d > threshold, d, 0.0)
    dep = np.where(d < -threshold, -d, 0.0)
    bed = np.where(valid, (dem0 + dem1 - ero - dep) / 2.0, np.nan)
    return ero, dep, bed, valid


def tetra_interpolate(ero, dep, x, y, t_end, times, time_scale=1.0):
    """Height of the change body at each time in `times` (list) -> array (n, rows, cols).

    Points: eroded cells at t = 0 and deposited cells at t = t_end, with body
    height as attribute. Cells outside the convex hull in (x, y, t) get 0.
    """
    i0 = np.flatnonzero(ero.ravel() > 0)
    i1 = np.flatnonzero(dep.ravel() > 0)
    if i0.size == 0 or i1.size == 0:
        raise ValueError("No change body found; lower the threshold?")
    xf, yf = x.ravel(), y.ravel()
    pts = np.vstack([
        np.column_stack([xf[i0], yf[i0], np.zeros(i0.size)]),
        np.column_stack([xf[i1], yf[i1], np.full(i1.size, float(t_end) * time_scale)]),
    ])
    z = np.concatenate([ero.ravel()[i0], dep.ravel()[i1]])
    tri = Delaunay(pts)
    out = np.zeros((len(times),) + ero.shape)
    n = ero.size
    for k, t in enumerate(times):
        q = np.column_stack([xf, yf, np.full(n, float(t) * time_scale)])
        s = tri.find_simplex(q)
        ok = s >= 0
        T = tri.transform[s[ok]]                       # (m, 4, 3)
        b3 = np.einsum("mij,mj->mi", T[:, :3, :], q[ok] - T[:, 3, :])
        bary = np.column_stack([b3, 1.0 - b3.sum(axis=1)])   # N1..N4 (shape functions)
        h = np.zeros(n)
        h[ok] = np.einsum("mi,mi->m", bary, z[tri.simplices[s[ok]]])
        out[k] = np.clip(h, 0.0, None).reshape(ero.shape)
    return out


def hull_mask(ero, dep, x, y):
    """Convex hull of both change bodies as a stand-in for the range polygon M."""
    idx = np.flatnonzero((ero + dep).ravel() > 0)
    hull = Delaunay(np.column_stack([x.ravel()[idx], y.ravel()[idx]]))
    q = np.column_stack([x.ravel(), y.ravel()])
    return (hull.find_simplex(q) >= 0).reshape(ero.shape)


def apply_constraints(h, range_mask, v_expected, cell_area, error=1e-3,
                      dissipation=0.0, max_iter=1000):
    """Range constraint + mass balance by equal redistribution (Eq. 5 of the paper).

    h_j' = h_j - (V_i - V_sum) / (n_A * dS) on the active change cells, then
    negative values are cut and the range is re-applied; repeated until
    |V_i - V_sum| <= error + dissipation or max_iter is reached.
    """
    h = np.where(range_mask, h, 0.0)
    for _ in range(max_iter):
        v = h.sum() * cell_area
        diff = v - v_expected
        if abs(diff) <= error + dissipation:
            break
        active = h > 0
        if not active.any():
            break
        h = np.where(active, h - diff / (active.sum() * cell_area), h)
        h = np.where(range_mask, np.clip(h, 0.0, None), 0.0)
    return h


def run(dem0, dem1, gt, t_end, times=None, threshold=0.005, constrain=True,
        range_mask=None, expected_volume=None, error=1e-3, time_scale=1.0):
    """Interpolate DEMs for `times` (default 1..t_end-1). Returns dict of arrays."""
    ero, dep, bed, valid = extract_bodies(dem0, dem1, threshold)
    x, y = cell_centers(dem0.shape, gt)
    area = abs(gt[1] * gt[5] - gt[2] * gt[4])
    times = list(range(1, int(t_end))) if times is None else list(times)
    h = tetra_interpolate(ero, dep, x, y, t_end, times, time_scale)
    if constrain:
        if range_mask is None:
            range_mask = hull_mask(ero, dep, x, y)
        v_exp = ero.sum() * area if expected_volume is None else expected_volume
        h = np.stack([apply_constraints(hk, range_mask, v_exp, area, error) for hk in h])
    vol = h.reshape(len(times), -1).sum(axis=1) * area
    return {"times": times, "dem": bed[None] + h, "change": h, "volume": vol,
            "bed": bed, "erosion": ero, "deposition": dep}


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dem_t0")
    ap.add_argument("dem_tend")
    ap.add_argument("outdir")
    ap.add_argument("--t-end", type=float, default=38, help="time index of the second DEM (default 38)")
    ap.add_argument("--steps", type=int, default=None,
                    help="number of intermediate steps (default t_end-1, i.e. t=1..t_end-1)")
    ap.add_argument("--threshold", type=float, default=0.005, help="cut-and-fill threshold in DEM units (default 0.005)")
    ap.add_argument("--no-constraints", action="store_true", help="plain tetrahedral interpolation")
    ap.add_argument("--range-mask", help="raster; non-zero = inside range polygon M")
    ap.add_argument("--range-polygon", help="vector polygon for M (needs GDAL Python bindings)")
    ap.add_argument("--expected-volume", type=float, help="expected volume of the change body (default: volume of the t0 body)")
    ap.add_argument("--error", type=float, default=1e-3, help="mass balance tolerance (default 0.001 m3)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="scale of t relative to x,y in the tessellation (default 1)")
    a = ap.parse_args(argv)

    dem0, gt, wkt = read_dem(a.dem_t0)
    dem1, gt1, _ = read_dem(a.dem_tend)
    if dem0.shape != dem1.shape or not np.allclose(gt, gt1):
        raise SystemExit("The two DEMs must share the same grid (shape and geotransform).")
    mask = None
    if a.range_mask:
        mask = read_dem(a.range_mask)[0] > 0
    elif a.range_polygon:
        mask = rasterize_polygon(a.range_polygon, dem0.shape, gt)
    times = None
    if a.steps:
        times = list(np.linspace(0, a.t_end, a.steps + 2)[1:-1])
    r = run(dem0, dem1, gt, a.t_end, times, a.threshold, not a.no_constraints,
            mask, a.expected_volume, a.error, a.time_scale)
    os.makedirs(a.outdir, exist_ok=True)
    for t, dem, v in zip(r["times"], r["dem"], r["volume"]):
        write_dem(os.path.join(a.outdir, f"output_t{t:.2f}.tif"), dem, gt, wkt)
        print(f"t={t:6.2f}  change-body volume = {v:.4f} m3")
    print(f"backend: {_BACKEND}")


if __name__ == "__main__":
    main()
