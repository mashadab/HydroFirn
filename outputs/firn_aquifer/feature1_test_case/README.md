# Feature I: initial test case

This uses the zoomed panel-B domain with its 2 km buffer. Original reconstructed cell heights are retained exactly, and every other cell (including internal holes) has initial saturated thickness **h0 = 0 m**. The zeros are a modeling assumption for the test case, not measurements of dry firn. Source reconstruction files are unchanged.

Grid: 69 columns × 88 rows at 250 m in EPSG:32624. Initial positive footprint: 43.25 km² (692 cells); assumed-zero cells: 5380.

- `feature1_initial_height_250m.tif`: north-up height raster in metres; zero is valid data and no cells are NoData.
- `feature1_initial_height.npz`: `h0_m[y,x]`, `reconstructed_patch_mask[y,x]`, cell-centre coordinates and grid spacing. Both NPZ axes increase east/north; row 0 is the southern row. `x_m`/`y_m` are local coordinates relative to `utm_origin_m`; `x_utm_m`/`y_utm_m` are absolute coordinates.
- `metadata.json`: bounds, assumptions, provenance and validation.

```python
import numpy as np
with np.load('feature1_initial_height.npz') as data:
    h0 = data['h0_m']
    patch_mask = data['reconstructed_patch_mask']
    x, y = data['x_m'], data['y_m']
```

Only the initial thickness field is initialized here. No expansion simulation, boundary conditions or extrapolation of aquifer-base elevations is included. Reproduce from the repository root with:

```sh
/opt/anaconda3/bin/python3 src/initialize_feature1_test_case.py
```
