# Montgomery 2016 aquifer initialization

This initializes observed geometry and a saturated-region mask for the 2016
Helheim transect. Source: Montgomery et al. (2017), Figure 4 and Table 1,
page 6, DOI [10.3389/feart.2017.00010](https://doi.org/10.3389/feart.2017.00010).

## Reproduce

From the HydroFirn repository, with NumPy installed:

```sh
python src/initialize_montgomery2016.py
```

Add `--plot` with Matplotlib installed to regenerate the PNG and SVG.
Defaults: target horizontal cells of 100 m, vertical cells of 0.5 m, and a
10 m geometric buffer below the lowest observed aquifer base. `--dx`, `--dy`,
and `--below-base` change these settings. No transient simulation runs.

## Geometry

The station input is `../montgomery2016_geometry_stations.json`, which
retains source station identities, Table 1 thicknesses, and graphical picks.
Horizontal position is `x = 1000 D`, without shifting the Site 12 origin.
Measured 2016 coverage is D = 0.41-18.35 km. Nothing is extrapolated to D = 0.

Surface elevation is Table 1 E. The aquifer top and base are graphical
readings of Figure 4's dashed boundaries, rounded to 0.5 m. Both boundaries
use the same local vertical references and depth-axis calibration.
Station elevations are E minus the corresponding depth; boundaries are
linearly interpolated between stations.

At Site 5, the initialized thickness is 8.5 m, consistent with Figure 4.
Table 1's different H = 17 m remains preserved in the station input.
Digitization rounding is separate from the original measurement uncertainty.

The surface rises 252 m across the observed domain, while the base rises
254.5 m. A constant depth of 27.82 m approximates the site bases with an RMS
deviation of 2.09 m. The actual initialization retains varying base depth.

## Saved NPZ arrays

`montgomery2016_initial_geometry.npz` contains a rectangular Cartesian grid.
Two-dimensional arrays have shape `(Ny, Nx)` with x increasing inland and
solver y increasing downward. Absolute elevation is `Z = reference - Y`.

- `x_faces_m`, `y_faces_m`, `x_centers_m`, `y_centers_m`: grid coordinates.
- `elevation_reference_m_asl`: fixed elevation of the y = 0 datum.
- `X_m`, `Y_down_m`, `Z_elevation_m_asl`: cell-center coordinates.
- `surface_z_m`, `water_table_z_m`, `base_z_m`: elevations at column centers.
- `region`: 0 air, 1 unsaturated firn, 2 aquifer, 3 below the aquifer base.
- `water_saturation_initial`: 1 in aquifer cells, 0 in overlying firn cells,
  NaN in air and below-base cells. Dry overlying firn is an explicit modeling
  assumption; residual water has not been inferred from the geometry.
- `aquifer_pressure_head_initial_m`: water-table elevation minus cell
  elevation in aquifer cells, NaN elsewhere. This imposes vertical
  hydrostatic pressure, with horizontal flow permitted by head gradients.
- `aquifer_geometric_cell_fraction`: geometric fraction of each vertical
  cell interval occupied by the aquifer at the column center. This is not
  porosity or liquid-water volume fraction, and is not substituted for
  water saturation in the cell-center mask.

For the existing HydroFirn cell ordering, map any `(Ny, Nx)` array with
`field.T.reshape(-1, 1)`, also available as the initializer's `solver_column`.

## Before time stepping

This is a geometry and hydraulic-mask initialization. Temperature, porosity,
permeability, and solver boundary conditions are still needed to create a
fully specified thermohydrologic state. No material properties are assigned
to the geometric below-base region. Conserved composition C and enthalpy H
are therefore not generated from unsupported temperature/porosity values.

The grid and air/basal masks need to be connected to the chosen solver's
active domain and boundary handling before a time-dependent run.
