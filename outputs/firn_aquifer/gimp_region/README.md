# GIMP surface elevation: full Feature I region

This map covers the full Feature I observation extent and the Montgomery 2016
seismic stations, with a 5 km buffer. It covers 44.25 ×
57.36 km, including unsurveyed terrain between tracks.

- `feature1_gimp_surface_overview.png`: elevation map and 3D surface overview.
- `feature1_gimp_surface_3d.html`: self-contained interactive 3D surface; true-scale button.
- `feature1_gimp_surface_30m.tif`: elevation raster in EPSG:32624, 30 m grid,
  WGS84 ellipsoid heights in metres. Bilinearly reprojected from the local
  NASA GIMP EPSG:3413 mosaic. No geoid or epoch correction is applied.
- `metadata.json`: exact bounds, sources, resolution and rendering details.

Nominal DEM epoch is 2007. Finite elevation coverage is 100.00%.
Surface elevations span 1359.9–1983.3 m above the WGS84 ellipsoid.
The DEM describes surface terrain, not the aquifer top or base. Radar and seismic
locations are overlaid for reference and do not establish an aquifer boundary.
The 3D preview uses 240 m sampling and 15× vertical exaggeration initially.
Source: https://nsidc.org/data/nsidc-0645/versions/1

Reproduce from HydroFirn with the scientific Python environment:
`python src/plot_gimp_feature1_region.py`
