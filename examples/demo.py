"""Synthetic demo: one bump moves 3 m down a slope between t0 and t10."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spacetime_interp as si

res = 0.1
rows, cols = 60, 30
gt = (0.0, res, 0.0, rows * res, 0.0, -res)
yy, xx = np.indices((rows, cols)) * res
bed = 0.3 * (rows * res - yy)                       # planar slope
def bump(cy, cx):
    return 0.4 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 0.2 ** 2))
dem0 = bed + bump(1.0, 1.5)
dem1 = bed + bump(4.0, 1.5)
os.makedirs("demo_out", exist_ok=True)
si.write_dem("demo_out/dem_t0.tif", dem0, gt)
si.write_dem("demo_out/dem_t10.tif", dem1, gt)
for con in (False, True):
    r = si.run(dem0, dem1, gt, 10, threshold=0.01, constrain=con)
    print("constrained" if con else "unconstrained", "volumes:", np.round(r["volume"], 3))
