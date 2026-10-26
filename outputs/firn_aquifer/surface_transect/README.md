# Surface transect beyond Site 12

The transect samples the local NASA GIMP DEM at 30 m intervals along a straight
continuation of the Montgomery et al. (2017) survey alignment. The first
extension is 20 km beyond nominal Site 12; change `--extend-km` to adjust it.

## Reproduce

From the HydroFirn repository, with NumPy, SciPy, rasterio, pyproj and
Matplotlib installed:

```sh
python src/extract_montgomery_surface_transect.py --extend-km 20 --spacing-m 30 --plot
```

The DEM defaults to the existing mosaic in the neighboring project:
`meltwater-percolation/Infiltration/JPL/Greenland_surface_elevation/combined_greenland_tif.vrt`.
Use `--dem`, `--table`, `--geometry`, or `--output` to override file locations.
Source DEM files are read without changes.

## Alignment and distance

The 2016 latitude/longitude coordinates from Table 1 are transformed to UTM
zone 24N (EPSG:32624). A least-squares direction is fitted through the exact
Site 12(a) position at its published D = 0.41 km. The vector is normalized to
unit length, preserving the published distance as the model coordinate.
The nominal D = 0 position lies 410 m farther along that line, downglacier.

Positive D runs inland; negative D extends beyond nominal Site 12. The
survey segment ends at D = 18.35 km. The output spans -20 to 18.35 km.

The station coordinates are not perfectly collinear. The cross-track RMS
residual is about 115 m; the full position RMS residual at published D is
about 123 m. Individual residuals are included in the comparison file.
This is the requested straight continuation of the survey, not a traced
glacier centerline or velocity-derived ice-flow path.

## Elevations

The local mosaic is EPSG:3413 with 30 m posting. Elevations are sampled
bilinearly using pixel centers. Nodata remains missing; it is not replaced
by a nearby elevation. Exact D = 0 and all 2016 station D values are also
included in the sample series.

GIMP's nominal epoch is 2007 and heights refer to the WGS84 ellipsoid.
Documentation: [GIMP DEM](https://byrd.osu.edu/research/groups/glacier-dynamics/data/gimpdem).
No geoid or epoch correction, or fitted elevation offset, is applied.
Table 1's elevations are retained as reported rather than being assigned a
new vertical datum. At the actual station coordinates, this DEM and Table 1
differ by about 1.4 m RMS (3.25 m maximum absolute difference).

The current continuation ends 20 km beyond nominal Site 12. Endpoint
elevations and the minimum elevation along this section are recorded in
`transect_metadata.json`.

## Files

- `montgomery_surface_transect.csv`: signed D, geographic coordinates,
  EPSG:3413 coordinates, sampled surface elevation and section labels.
- `montgomery_surface_and_base_transect.csv`: the same surface samples plus
  model base elevation, separation below the DEM, and explicit base-source
  labels. `base_is_assumed` is 1 for the continuation and 0 for the extracted
  station-interpolation region.
- `montgomery_surface_base_and_aquifer_transect.csv`: adds the aquifer top,
  thickness, surface/top heights above the same local base, and provenance.
  `aquifer_geometry_defined` is 1 from D = 0 to 18.35 km. Top and thickness
  are blank for D < 0; this denotes unspecified geometry, not physical absence.
- `montgomery_surface_transect.geojson`: WGS84 line, actual survey station
  points, and nominal D = 0 point, suitable for GIS import.
- `station_DEM_comparison.csv`: actual station coordinates and elevations,
  DEM samples at actual and fitted positions, and alignment residuals.
- `transect_metadata.json`: source, methods, CRS, epoch, fit and comparisons.
- PNG/SVG: three panels with terrain map, absolute elevations, and height
  above the local base. Both profiles use reversed signed distance.
  The aquifer is filled between its top and base only through D = 0.
  Extracted boundaries are solid and assumed continuations are dashed.
  The map includes true north and a geodesic 5 km scale bar.

## Assumed base continuation

The base is gap-filled beyond Site 12(a), at D = 0.41 km, using

```text
z_base(D) = z_base_12a + z_surface_DEM(D) - z_surface_DEM_12a
```

This preserves the extracted base elevation at the join and gives the base
exactly the same local slope as the surface in the assumed region. It also
fills the 410 m between Site 12(a) and nominal D = 0. Beyond the join, the
base is a modeling assumption rather than an observation.

The extracted anchor base is 1487 m in the paper's reported elevation
convention, from E = 1518 m minus a graphical base depth of 31 m. The DEM
surface at that coordinate is approximately 1520.05 m. Consequently the
effective constant DEM-to-base separation is approximately 33.05 m. This
includes the 2.05 m DEM-versus-reported surface difference at the anchor;
it is not a new observed aquifer depth or a formal vertical-datum correction.

Within the measured range, base elevations remain linear interpolations of
the extracted station values. The original surface-only CSV and observed
aquifer initialization are preserved.

## Aquifer through D = 0 and panel c

Between D = 0.41 and 18.35 km, the aquifer top is a linear interpolation of
the Figure 4 top elevations in `montgomery2016_geometry_stations.json`.
The top and base are graphical estimates rounded to 0.5 m. At Site 5, the
Figure 4 top/base thickness (8.5 m) is used consistently; Table 1's conflicting
H = 17 m is retained in the original source table.

For the short gap from Site 12(a) to nominal D = 0, the top follows the
surface-parallel base with constant Site 12(a) thickness:

```text
z_top(D) = z_base(D) + 20.5 m,  0 <= D < 0.41 km
```

This joins continuously to the extracted top at Site 12(a). It is an
explicit assumption over 410 m. The aquifer top and saturated thickness
are left unspecified beyond D = 0, while the assumed base and DEM surface
continue along the full 20 km extension.

Panel c transforms the absolute elevations in panel b using
`height(D) = elevation(D) - z_base(D)`. The base is therefore 0 m, the
aquifer top is the local aquifer thickness, and the surface is the full
surface-to-base separation. Heights increase upward. This is a coordinate
transformation of the existing mixed-source geometry; it does not apply
a new vertical-datum or epoch correction.

## Sources

Howat, Negrete and Smith (2014), *The Greenland Ice Mapping Project (GIMP)
land classification and surface elevation data sets*, The Cryosphere 8,
1509-1518, [doi:10.5194/tc-8-1509-2014](https://doi.org/10.5194/tc-8-1509-2014).

Montgomery et al. (2017), Figure 4 and Table 1,
[doi:10.3389/feart.2017.00010](https://doi.org/10.3389/feart.2017.00010).
