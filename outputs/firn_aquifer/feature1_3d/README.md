# Helheim Feature I: reconstructed 3D geometry

This is a **2012–2014 composite, supported subset**, not a measured full aquifer
or a dated initial condition. It is built from existing CFA3D_Application data.

## Result

- Horizontal grid: 250 m sampling, 139 × 191 nodes.
- Active footprint: 43.250 km² in 14 supported components.
- Bulk aquifer material volume: 0.2919 km³, including ice matrix.
- Mesh thickness: 3.91–23.46 m.
- Volume mesh: 8,304 triangular-prism wedges in 6 sigma layers.
- No porosity has been assigned; bulk volume must not be interpreted as liquid-water volume.

## Files and coordinates

- `feature1_geometry_3d.html`: self-contained interactive preview; 15× vertical
  exaggeration initially, with a true-scale button and optional source points.
- `feature1_geometry_preview.png`: static 3D view and source-support maps.
- `feature1_geometry.npz`: gridded surface/top/base, thickness/depth, masks,
  support distances, connected components, and boundary/volume mesh arrays.
- `feature1_aquifer_volume.vtu`: ParaView-compatible volume mesh, with thickness,
  observation-distance and component fields.
- `feature1_aquifer_boundary.obj`: closed boundary triangles at true coordinate scale.
- `metadata.json`, `validation.json`: methods, source hashes, assumptions and checks.
- `*_source_points.csv`, `*_spatial_cv.csv`: traceable inputs and interpolation diagnostics.

All mesh x/y coordinates are **local east/north metres**. Add the origin
(462000.000, 7313000.000) m to recover EPSG:32624 UTM coordinates.
Mesh z coordinates are **metres above the WGS84 ellipsoid**, not local thickness
or orthometric heights. Grid arrays have shape (ny, nx); active_cells and
component_id have shape (ny-1, nx-1). Coordinates increase eastward and northward.
Indices in NPZ/VTU are zero-based; OBJ indices are one-based. NPZ loads without pickle.

## Construction and limits

1. Read 4,578 Feature I thickness points from Chu Dataset S1, worksheet
   `aq_hw_f1`. Preserve original Excel row IDs. There is no per-point date column.
2. Read 2012/2014 Miège points, retaining 1,104 finite, positive-depth
   records within 15 km of Chu tracks. Exact repeated coordinates use median values.
3. Interpolate thickness and depth separately with piecewise-linear Delaunay
   interpolation in UTM coordinates. There is no extrapolation outside either hull.
4. Sample the existing GIMP DEM bilinearly. Preserve nodata. Calculate top = surface
   minus water-table depth; base = top minus thickness. GIMP is nominally 2007;
   the mixed epoch is explicit, without an invented correction.
5. Require distance to **each** observation set ≤ 5000 m and
   interpolation triangle maximum edge ≤ 10000 m in **each** set.
   A cell enters the mesh only when all four corners pass and thickness is
   ≥ 3.9 m. This screening does not identify a physical margin.
6. Triangulate and extrude each accepted cell. Retain every component. Separate
   diagonal-only vertex fans to prevent nonmanifold contact. Close reconstruction
   cuts with vertical side walls. Those walls are artificial modeling boundaries.

Missing areas are unspecified, not evidence of physical absence. Grid spacing
does not imply equivalent observation resolution. The full published Feature I
area is not forced. The broad Miège KML footprint is not used as a Feature I mask.
Montgomery's later 2015–2016 transect is not imposed on the 2012–2014 composite.

The study-wide 3.9 m thickness uncertainty is a screening/sensitivity scale,
not a local standard deviation. On the **fixed accepted footprint**, subtracting
or adding 3.9 m gives bulk volumes 0.1232–
0.4606 km³. This is not a confidence
interval and excludes depth, DEM, temporal, boundary and interpolation uncertainty.

## Validation

Surface order is checked; every mesh edge has two consistently oriented incident
faces; faces have nonzero area. Closed-mesh volume agrees with the independently
integrated piecewise-linear thickness to relative error 1.12e-14.
Five-fold 2 km spatial-block withholding gives thickness RMSE 6.15 m
on 3077/4578 predictable points and water-table
depth RMSE 2.11 m on 764/1104 points.
These are interpolation diagnostics, not external field validation. Source ELEV
minus GIMP has RMSE 2.03 m; source ELEV is
retained for comparison but is not assigned a new vertical datum or used to shift DEM.

## Sources and reproduction

- Chu, Schroeder & Siegfried (2018), thickness Dataset S1: https://doi.org/10.1029/2018GL079751
- Miège et al. (2016), water-table depth: https://doi.org/10.1002/2016JF003869
- Miège (2018), data archive: https://doi.org/10.18739/A2TM72225
- NASA GIMP DEM: https://nsidc.org/data/nsidc-0645/versions/1

From HydroFirn, with numpy/scipy/matplotlib/pyproj/rasterio/plotly installed:

```sh
python src/reconstruct_helheim_feature1.py
```

Use `--help` to change spacing, support limits, thickness cutoff, sigma layers,
input paths or output directory. A time-dependent model additionally needs
material properties, thermal state, recharge and explicit boundary conditions.
