# Interpolated firn-aquifer model inputs

The main CSV contains latitude, longitude, x, base elevation, slope in degrees,
and aquifer height (vertical saturated thickness), plus coordinate, derivative,
and source fields. It has 1,286 rows at target 30 m spacing, with exact survey
stations, the origin, and both endpoints included.

## Coordinates and zero-height condition

This export uses **x = -1000 D**, in metres. Nominal Site 12 is x = 0.
Negative x is inland, with the observed survey ending at x = -18,350 m.
Positive x is downglacier, extending to x = 20,000 m.

This is the opposite sign to `x_D_m = 1000 D` in the earlier observed-only
initializer. `D_km` is retained in this export to make conversion explicit.
The earlier files and figures are unchanged.

Latitude and longitude lie on the previously fitted straight transect.
They are interpolated in x between its control locations. The original
rounded GPS station positions are slightly off that line and remain in
`../surface_transect/station_DEM_comparison.csv`.

Aquifer height is interpolated between the extracted Figure 4 thicknesses
for x <= -410 m. From x = -410 m to x = 0 it remains 20.5 m, following the
previous Site 12(a) gap-filling assumption. **For every x > 0 it is exactly
zero**, as requested. H(0) = 20.5 m, so there is a prescribed discontinuity
at the origin; no interpolation across that discontinuity creates a taper.

## Interpolation and slope

For a scalar field f between control positions x_i and x_(i+1):

```text
t = (x - x_i) / (x_(i+1) - x_i)
f(x) = (1-t) f_i + t f_(i+1)
```

There are 692 base controls: the 11 extracted station bases, all saved
DEM-derived assumed-base points below Site 12(a), and the included origin.
The observed station base elevations are preserved. The continuation uses
the existing surface-parallel base and does not fit a new overall slope.
No field is extrapolated outside [-18350, 20000] m.

Slope is derived from base elevations, not directly measured. Its sign is
**positive downhill in the +x direction**, so uphill segments have negative
angles. Two slope fields serve different purposes:

- `slope_deg`: the requested linearly interpolated angle estimates. At each
  control, the slope estimate is the arithmetic mean of its two adjacent
  segment angles. Endpoints use their single adjacent segment angle. These
  nodal estimates are then interpolated linearly in x.
- `geometric_slope_deg`: the exact slope of the piecewise linear base,
  `degrees(atan(-dz_base/dx))`. It is constant within each straight segment
  and can jump at a control point. At a control, the downstream/right-hand
  segment is used; the final endpoint uses the last segment. Use this field
  when the model slope must agree with the base geometry.

Interpolating base elevations and slope angles independently does not make
the interpolated angle the exact derivative of the base. `base_dz_dx` also
provides the signed geometric gradient in m/m (negative downhill).

## Files

- `montgomery_interpolated_model_inputs.csv`: requested inputs on the grid.
- `montgomery_interpolated_model_inputs.npz`: the same named arrays for NumPy.
  It can be loaded with `numpy.load(..., allow_pickle=False)`.
- `montgomery_interpolation_controls.csv`: controls, nodal slope estimates,
  and source labels used by the interpolation.
- `interpolation_metadata.json`: units, methods, assumptions, and provenance.
- `interpolation_validation.json`: numerical checks and coordinate errors.
- `montgomery_interpolated_inputs.png` / `.svg`: base, slope, and height profiles.

Primary CSV columns are `x_m`, `latitude_deg_north`, `longitude_deg_east`,
`base_elevation_m`, `slope_deg`, and `aquifer_height_m`. Additional fields
give D, the geometric slope/gradient, aquifer-top elevation, and source labels.
Where aquifer height is zero, aquifer-top elevation coincides with the base.

## Reproduce and query arbitrary positions

From the HydroFirn repository:

```sh
python src/interpolate_montgomery_transect.py --spacing-m 30 --plot
```

To export a custom grid, change `--spacing-m`. To export explicit queries:

```sh
python src/interpolate_montgomery_transect.py --x-m -1000 0 1000 --output /tmp/firn-query
```

For model code, evaluate any finite scalar or 1-D array of x positions:

```python
from src.interpolate_montgomery_transect import TransectInterpolator

transect = TransectInterpolator()
p = transect.evaluate([-1000.0, 0.0, 1000.0])
latitude = p['latitude_deg_north']
longitude = p['longitude_deg_east']
base = p['base_elevation_m']
slope_interpolated = p['slope_deg']
slope_geometric = p['geometric_slope_deg']
height = p['aquifer_height_m']
```

The interpolator requires NumPy and the standard library. Only plotting
requires Matplotlib. No source DEM or source extraction is changed.

## Source conventions

Montgomery et al. (2017), Figure 4 and Table 1,
[doi:10.3389/feart.2017.00010](https://doi.org/10.3389/feart.2017.00010),
as recorded in `../montgomery2016_geometry_stations.json`.
Graphical depths were rounded to 0.5 m. Site 5 uses Figure 4's 8.5 m
thickness consistently; the conflicting Table 1 H = 17 m remains in the
original source table.

The downstream base comes from the existing assumed surface-parallel
continuation using the local NASA GIMP DEM. Source/DEM vertical references
and epochs are preserved; interpolation adds no datum or epoch correction.
See `../surface_transect/README.md` for the original DEM and alignment details.
