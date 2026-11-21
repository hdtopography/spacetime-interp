# spacetime-interp

Spatiotemporal interpolation of topographic change between two DEMs, using a 3D (x, y, t) Delaunay tetrahedral mesh and the tetrahedral shape function, with optional physical reality constraints (spatial range and mass balance).

This is a minimal GDAL/NumPy/SciPy implementation of the method in:

> Hayakawa, Y.S.; Wang, L. Spatiotemporal interpolation of topographic changes from terrestrial LIDAR observations: a controlled experiment with rolling objects on a slope. *Remote Sensing* (under review; reference and DOI to be updated).

The original implementation was a Windows GUI tool (C#, ArcObjects) developed by Wang Luyao as part of her master's research (Hokkaido University, 2023). This repository re-implements the method without any proprietary dependency.

## Method

1. Cut and fill between the DEM at t0 and the DEM at the last epoch gives the erosion body (object at t0), the deposition body (object at the last epoch) and the bed.
2. Body cells become points (x, y, t) with body height as attribute; a 3D Delaunay tessellation is built and the height at any (x, y, t) is computed with the tetrahedral shape function.
3. Constraints (optional), applied at every time step: (i) change is set to zero outside the range polygon M; (ii) the volume of the change body is forced to the expected value by equal redistribution, iterated until both constraints hold.
4. The interpolated change body is added to the bed.

## Install

```
pip install numpy scipy
pip install GDAL      # needs a matching system GDAL; or: conda install -c conda-forge gdal
```
If the GDAL Python bindings are missing, `rasterio` is used for raster I/O instead.

## Use

```
python spacetime_interp.py dem_t0.tif dem_t38.tif out_dir --t-end 38 \
    --threshold 0.005 --expected-volume 0.4 --range-polygon area.shp
```

- Both DEMs must share the same grid. Output: `out_dir/output_t<k>.tif` for k = 1 ... t_end-1 (bed + change body).
- `--no-constraints` gives the plain tetrahedral interpolation.
- Range polygon M: `--range-polygon` (vector, needs GDAL), `--range-mask` (raster, non-zero = inside). Without either, the convex hull of both bodies is used as a stand-in.
- `--expected-volume` defaults to the volume of the t0 body.
- `--error` (default 0.001 m3) is the mass balance tolerance.

Python API: `spacetime_interp.run(dem0, dem1, geotransform, t_end, ...)`. A synthetic example is in `examples/demo.py`.

## Notes and limitations

- Heights are in DEM units (metres assumed for volumes). Time is a unitless index scaled by `--time-scale` in the tessellation (default 1, as in the original tool).
- The interpolation is linear in space-time: a moving object is smeared along its path, which the constraints only partly correct.
- The bed is `(DEM0 + DEM_end - erosion - deposition) / 2`.

## License

MIT, see `LICENSE`.
