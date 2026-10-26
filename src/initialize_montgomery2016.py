"""Initialize the observed 2016 firn-aquifer geometry and a hydraulic mask.

Run from the repository root:
    python src/initialize_montgomery2016.py --plot

Only the Python standard library and NumPy are required; --plot also needs
Matplotlib. The input station JSON records the source and digitization method.
No transient solver, temperature, permeability, or porosity is prescribed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs' / 'firn_aquifer' / 'initialization_2016'
DEFAULT_STATIONS = ROOT / 'outputs' / 'firn_aquifer' / 'montgomery2016_geometry_stations.json'


def load_stations(path=DEFAULT_STATIONS):
    """Read and validate the 11 matched 2016 profiles in increasing D."""
    payload = json.loads(Path(path).read_text())
    records = sorted(payload['stations'], key=lambda r: r['x_m'])
    if len(records) < 2 or any(r['year'] != 2016 for r in records):
        raise ValueError('At least two 2016 profiles are required.')
    x = np.array([r['x_m'] for r in records], dtype=float)
    surface = np.array([r['surface_elevation_m_asl'] for r in records])
    top = np.array([r['top_elevation_est_m_asl'] for r in records])
    base = np.array([r['base_elevation_est_m_asl'] for r in records])
    if not np.isfinite(np.r_[x, surface, top, base]).all():
        raise ValueError('Coordinates and elevations must be finite.')
    if np.any(np.diff(x) <= 0) or np.any(base >= top) or np.any(top >= surface):
        raise ValueError('Require increasing x and base < top < surface.')
    return records


def sample_boundaries(x_m, stations):
    """Linearly interpolate elevations within the observed domain only."""
    x = np.asarray(x_m, dtype=float)
    knots = np.array([r['x_m'] for r in stations])
    if not np.isfinite(x).all() or np.any(x < knots[0]) or np.any(x > knots[-1]):
        raise ValueError(f'x must lie within observed [{knots[0]}, {knots[-1]}] m.')
    return {
        name: np.interp(x, knots, [r[key] for r in stations])
        for name, key in (
            ('surface_z_m', 'surface_elevation_m_asl'),
            ('water_table_z_m', 'top_elevation_est_m_asl'),
            ('base_z_m', 'base_elevation_est_m_asl')
        )
    }


def solver_column(field):
    """Map a (Ny,Nx) field to HydroFirn's y-fast (N,1) cell ordering."""
    return np.asarray(field).T.reshape(-1, 1)


def initialize(stations, dx_m=100.0, dy_m=0.5, below_base_m=10.0):
    """Build geometry, region labels, saturation and hydrostatic pressure head.

    Absolute elevation z is positive upward. Solver y is positive downward
    from the highest observed surface, so z = elevation_reference - y.
    Region labels: 0=air; 1=unsaturated firn; 2=aquifer; 3=below aquifer base.
    S_w=1 in the aquifer, S_w=0 above it is an explicit dry-firn assumption;
    air/below-base saturation is undefined (NaN). No thermal state is implied.
    """
    if not np.isfinite([dx_m, dy_m, below_base_m]).all():
        raise ValueError('Grid dimensions must be finite.')
    if min(dx_m, dy_m) <= 0 or below_base_m < 0:
        raise ValueError('Positive grid sizes and a nonnegative basal buffer are required.')
    xmin, xmax = stations[0]['x_m'], stations[-1]['x_m']
    reference = max(r['surface_elevation_m_asl'] for r in stations)
    zmin = min(r['base_elevation_est_m_asl'] for r in stations) - below_base_m
    nx = int(np.ceil((xmax - xmin) / dx_m))
    ny = int(np.ceil((reference - zmin) / dy_m))
    if nx * ny > 10_000_000:
        raise ValueError('Requested initialization exceeds 10 million cells.')
    xf = np.linspace(xmin, xmax, nx + 1)
    yf = np.linspace(0, reference - zmin, ny + 1)
    xc, yc = (xf[:-1]+xf[1:])/2, (yf[:-1]+yf[1:])/2
    X, Y = np.meshgrid(xc, yc)
    Z = reference - Y
    bnd = sample_boundaries(xc, stations)
    surface, top, base = (bnd[k][None, :] for k in ('surface_z_m','water_table_z_m','base_z_m'))
    region = np.zeros(Z.shape, dtype=np.uint8)
    region[(Z <= surface) & (Z > top)] = 1
    region[(Z <= top) & (Z >= base)] = 2
    region[Z < base] = 3
    sw = np.full(Z.shape, np.nan)
    sw[region == 1] = 0.0
    sw[region == 2] = 1.0
    pressure = np.where(region == 2, top - Z, np.nan)
    # Geometric vertical overlap, distinct from the cell-center S_w assignment.
    cell_z_top = reference - yf[:-1, None]
    cell_z_bottom = reference - yf[1:, None]
    fill = np.maximum(0, np.minimum(cell_z_top, top)-np.maximum(cell_z_bottom, base)) / np.diff(yf)[:,None]
    fields = dict(
        x_faces_m=xf, y_faces_m=yf, x_centers_m=xc, y_centers_m=yc,
        elevation_reference_m_asl=np.array(reference), X_m=X, Y_down_m=Y,
        Z_elevation_m_asl=Z, region=region, water_saturation_initial=sw,
        aquifer_pressure_head_initial_m=pressure, aquifer_geometric_cell_fraction=fill,
        **bnd
    )
    return fields


def summarize(stations, fields):
    x = np.array([r['x_m'] for r in stations])
    surface = np.array([r['surface_elevation_m_asl'] for r in stations])
    base = np.array([r['base_elevation_est_m_asl'] for r in stations])
    depth = surface-base
    return {
        'source':'Montgomery et al. (2017), Figure 4 and Table 1; DOI 10.3389/feart.2017.00010',
        'year':2016, 'station_count':len(stations),
        'observed_D_range_km':[float(x[0]/1000), float(x[-1]/1000)],
        'surface_rise_m':float(surface[-1]-surface[0]),
        'base_rise_m':float(base[-1]-base[0]),
        'base_depth_range_m':[float(depth.min()),float(depth.max())],
        'mean_base_depth_m':float(depth.mean()),
        'surface_fit_slope_m_per_km':float(np.polyfit(x/1000,surface,1)[0]),
        'base_fit_slope_m_per_km':float(np.polyfit(x/1000,base,1)[0]),
        'constant_depth_approximation_rmse_m':float(np.sqrt(np.mean((depth-depth.mean())**2))),
        'grid_shape_Ny_Nx':list(fields['region'].shape),
        'actual_dx_m':float(np.diff(fields['x_faces_m'])[0]),
        'actual_dy_m':float(np.diff(fields['y_faces_m'])[0]),
        'assumptions':[
            'Piecewise linear interpolation; no extrapolation beyond measured 2016 D coverage.',
            'Figure 4 top/base used consistently, including Site 5; original Table 1 H retained in station data.',
            'Depth estimates rounded to 0.5 m; source measurement uncertainty is additional.',
            'S_w=1 in aquifer and S_w=0 above: dry overlying firn is an initialization assumption.',
            'Hydrostatic vertical pressure head within aquifer; horizontal equilibrium is not imposed.',
            'Below-base and above-surface labels are geometric regions, not assigned material properties.',
            'Temperature, porosity, permeability and solver boundary conditions remain unspecified.'
        ]
    }


def plot_geometry(stations, fields, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xf = fields['x_faces_m']
    x = np.sort(np.r_[xf, [r['x_m'] for r in stations]])
    bnd = sample_boundaries(x, stations)
    s, t, b = (bnd[k] for k in ('surface_z_m','water_table_z_m','base_z_m'))
    station_x = np.array([r['x_m']/1000 for r in stations])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True, gridspec_kw={'height_ratios':[1.4,1]})
    ax = axes[0]
    ax.fill_between(x/1000,b.min()-10,b,color='#DFE5E9',label='Below aquifer base')
    ax.fill_between(x/1000,b,t,color='#51A9D4',label='Saturated firn aquifer')
    ax.fill_between(x/1000,t,s,color='#F1F4F6',label='Overlying firn')
    ax.plot(x/1000,s,color='#4A535B',lw=1.7,label='Surface')
    ax.plot(x/1000,t,color='#168BA4',lw=1.3,label='Aquifer top')
    ax.plot(x/1000,b,color='#154C7A',lw=1.4,label='Aquifer base')
    ax.scatter(station_x,[r['base_elevation_est_m_asl'] for r in stations],s=22,color='#154C7A',zorder=5)
    for r in stations:
        ax.annotate(r['site'],(r['x_m']/1000,r['surface_elevation_m_asl']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_ylabel('Elevation (m a.s.l.)')
    ax.set_ylim(b.min()-10,s.max()+22)
    ax.legend(loc='upper left',ncol=2,fontsize=8,frameon=False)
    ax.set_title('2016 Helheim firn aquifer: initialized geometry',loc='left',fontsize=14,pad=18)
    ax = axes[1]
    dtop, dbase = s-t, s-b
    ax.fill_between(x/1000,dtop,dbase,color='#51A9D4')
    ax.plot(x/1000,dtop,color='#168BA4',lw=1.4,label='Depth to aquifer top')
    ax.plot(x/1000,dbase,color='#154C7A',lw=1.4,label='Depth to aquifer base')
    ax.scatter(station_x,[r['base_depth_est_m'] for r in stations],s=22,color='#154C7A',zorder=5)
    ax.set_ylim(36,0)
    ax.set_ylabel('Depth below local surface (m)')
    ax.set_xlabel('Distance D from Site 12 (km); increases inland / upglacier')
    ax.legend(loc='upper right',fontsize=9,frameon=False)
    for ax in axes:
        ax.set_xlim(x.min()/1000,x.max()/1000)
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.text(.08,.02,'Source: Montgomery et al. (2017), Fig. 4 and Table 1. Graphical depths rounded to 0.5 m; linear interpolation.',fontsize=9,color='#47515A')
    fig.tight_layout(rect=(0,.055,1,1),h_pad=2)
    fig.savefig(output/'montgomery2016_initialized_geometry.png',dpi=180)
    fig.savefig(output/'montgomery2016_initialized_geometry.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stations',type=Path,default=DEFAULT_STATIONS)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--dx',type=float,default=100,help='Target horizontal cell size (m).')
    parser.add_argument('--dy',type=float,default=.5,help='Target downward vertical cell size (m).')
    parser.add_argument('--below-base',type=float,default=10,help='Geometric buffer below minimum aquifer base (m).')
    parser.add_argument('--plot',action='store_true')
    args = parser.parse_args()
    stations = load_stations(args.stations)
    fields = initialize(stations,args.dx,args.dy,args.below_base)
    summary = summarize(stations,fields)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'montgomery2016_initial_geometry.npz',**fields)
    (args.output/'initialization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.plot:
        plot_geometry(stations,fields,args.output)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
