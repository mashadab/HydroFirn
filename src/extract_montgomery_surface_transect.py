"""Sample the local GIMP DEM along the Montgomery survey and its continuation.

Default: 20 km beyond nominal Site 12, at 30 m sampling, using the 2016
survey coordinates to fit a straight alignment anchored at Site 12(a).
The extracted aquifer base is continued beyond Site 12(a) using the same
local slope as the DEM surface, as an explicit gap-filling assumption.
The aquifer top is interpolated between extracted stations and continued
only across the 410 m gap to D = 0, preserving the Site 12(a) thickness.
Dependencies: NumPy, SciPy, rasterio, pyproj; Matplotlib for --plot.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.windows import Window, from_bounds
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = ROOT.parent / 'meltwater-percolation/Infiltration/JPL/Greenland_surface_elevation/combined_greenland_tif.vrt'
DEFAULT_TABLE = ROOT / 'outputs/firn_aquifer/montgomery2017_table1.csv'
DEFAULT_GEOMETRY = ROOT / 'outputs/firn_aquifer/montgomery2016_geometry_stations.json'
DEFAULT_OUTPUT = ROOT / 'outputs/firn_aquifer/surface_transect'
GIMP_DOC = 'https://byrd.osu.edu/research/groups/glacier-dynamics/data/gimpdem'


def load_survey(path):
    with Path(path).open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['year'] == '2016']
    rows.sort(key=lambda r: float(r['distance_from_site12_km']))
    if len(rows) < 2:
        raise ValueError('Need at least two survey stations.')
    return rows


def fit_alignment(rows):
    """Fit an upglacier direction in local UTM 24N, fixing Site 12(a).

    Published D remains the horizontal model coordinate. Its origin is
    obtained by continuing 410 m downglacier from the measured Site 12(a).
    The fitted vector is normalized so one D metre is one local map metre.
    """
    forward = Transformer.from_crs(4326, 32624, always_xy=True)
    inverse = Transformer.from_crs(32624, 4326, always_xy=True)
    lon = np.array([float(r['longitude_deg_east']) for r in rows])
    lat = np.array([float(r['latitude_deg_north']) for r in rows])
    x, y = forward.transform(lon, lat)
    xy = np.column_stack((x, y))
    d = np.array([float(r['distance_from_site12_km'])*1000 for r in rows])
    if np.any(np.diff(d) <= 0) or rows[0]['site'] != '12 (a)':
        raise ValueError('Require unique increasing 2016 D with Site 12(a) first.')
    delta = d-d[0]
    fit = np.sum(delta[:,None]*(xy-xy[0]),axis=0)/np.sum(delta**2)
    unit = fit/np.linalg.norm(fit)
    origin = xy[0]-unit*d[0]
    perpendicular = np.array([-unit[1],unit[0]])
    predicted = origin+d[:,None]*unit
    return dict(
        D_m=d, station_xy=xy, predicted_xy=predicted, origin=origin, unit=unit,
        cross_track_m=(xy-origin)@perpendicular,
        coordinate_residual_m=np.linalg.norm(xy-predicted,axis=1),
        fit_scale=float(np.linalg.norm(fit)), inverse=inverse
    )


def bilinear_sample(dataset, x, y):
    """Sample pixel-centered elevations with no interpolation across nodata."""
    x, y = np.asarray(x), np.asarray(y)
    inv = ~dataset.transform
    col, row = inv*(x,y)
    col, row = col-.5, row-.5
    left, top = max(0,int(np.floor(col.min()))-1), max(0,int(np.floor(row.min()))-1)
    right = min(dataset.width,int(np.ceil(col.max()))+3)
    bottom = min(dataset.height,int(np.ceil(row.max()))+3)
    if right <= left or bottom <= top:
        return np.full(x.shape,np.nan)
    data = dataset.read(1,window=Window(left,top,right-left,bottom-top),masked=True).astype(float).filled(np.nan)
    return map_coordinates(data,[row-top,col-left],order=1,mode='constant',cval=np.nan,prefilter=False)


def build_transect(dataset, rows, alignment, extend_km, spacing_m):
    if not np.isfinite([extend_km,spacing_m]).all() or extend_km <= 0 or spacing_m <= 0:
        raise ValueError('Extension and spacing must be finite and positive.')
    if extend_km*1000/spacing_m > 1_000_000:
        raise ValueError('Requested transect is too large.')
    dmin, dmax = -1000*extend_km, alignment['D_m'].max()
    d = np.unique(np.r_[np.arange(dmin,0,spacing_m),np.arange(0,dmax,spacing_m),0,alignment['D_m'],dmax])
    xy = alignment['origin']+d[:,None]*alignment['unit']
    lon,lat = alignment['inverse'].transform(xy[:,0],xy[:,1])
    to_dem = Transformer.from_crs(32624,dataset.crs,always_xy=True)
    xd,yd = to_dem.transform(xy[:,0],xy[:,1])
    elevation = bilinear_sample(dataset,xd,yd)
    actual_lon = [float(r['longitude_deg_east']) for r in rows]
    actual_lat = [float(r['latitude_deg_north']) for r in rows]
    to_dem_geo = Transformer.from_crs(4326,dataset.crs,always_xy=True)
    sx,sy = to_dem_geo.transform(actual_lon,actual_lat)
    actual_elev = bilinear_sample(dataset,sx,sy)
    px,py = to_dem.transform(alignment['predicted_xy'][:,0],alignment['predicted_xy'][:,1])
    fitted_elev = bilinear_sample(dataset,px,py)
    records = []
    for dist,lo,la,xx,yy,z in zip(d,lon,lat,xd,yd,elevation):
        records.append({
            'D_km':float(dist/1000), 'x_D_m':float(dist),
            'longitude_deg_east':float(lo), 'latitude_deg_north':float(la),
            'dem_x_epsg3413_m':float(xx), 'dem_y_epsg3413_m':float(yy),
            'surface_elevation_wgs84_ellipsoid_m':float(z) if np.isfinite(z) else None,
            'section':'downglacier_extension' if dist < 0 else ('between_nominal_origin_and_12a' if dist < alignment['D_m'][0] else 'survey_alignment')
        })
    validation = []
    for r,z,zfit,err,cross in zip(rows,actual_elev,fitted_elev,alignment['coordinate_residual_m'],alignment['cross_track_m']):
        validation.append({
            'site':r['site'], 'D_km':float(r['distance_from_site12_km']),
            'longitude_deg_east':float(r['longitude_deg_east']),
            'latitude_deg_north':float(r['latitude_deg_north']),
            'table1_E_m_as_reported':float(r['surface_elevation_m_asl']),
            'dem_at_actual_station_ellipsoid_m':float(z),
            'dem_minus_table1_E_m':float(z)-float(r['surface_elevation_m_asl']),
            'dem_on_straight_line_at_same_D_ellipsoid_m':float(zfit),
            'straight_line_position_residual_m':float(err), 'cross_track_residual_m':float(cross)
        })
    geod = Geod(ellps='WGS84')
    zero_lon,zero_lat = alignment['inverse'].transform(*alignment['origin'])
    downstream_lon,downstream_lat = alignment['inverse'].transform(*(alignment['origin']-1000*alignment['unit']))
    azimuth = geod.inv(zero_lon,zero_lat,downstream_lon,downstream_lat)[0]%360
    valid = np.isfinite(elevation)
    below = (d < 0)&valid
    z0 = float(elevation[np.flatnonzero(d==0)[0]])
    delta = actual_elev-np.array([float(r['surface_elevation_m_asl']) for r in rows])
    summary = {
        'DEM_file':str(Path(dataset.name).resolve()),'DEM_product':'NASA GIMP 30 m local mosaic',
        'DEM_CRS':str(dataset.crs),'DEM_posting_m':list(dataset.res),
        'vertical_reference':'height above WGS84 ellipsoid; no geoid correction applied',
        'nominal_DEM_epoch':2007,'documentation':GIMP_DOC,
        'survey_source':'Montgomery et al. (2017), Table 1, 2016 sites; DOI 10.3389/feart.2017.00010',
        'alignment_method':'UTM 24N least-squares direction through Site 12(a); normalized to unit length, preserving published D',
        'local_alignment_CRS':'EPSG:32624','anchor_site':'12 (a)', 'anchor_D_km':float(alignment['D_m'][0]/1000),
        'nominal_D0_longitude':float(zero_lon),'nominal_D0_latitude':float(zero_lat),
        'downglacier_azimuth_deg':float(azimuth),
        'fitted_scale_before_normalization':alignment['fit_scale'],
        'cross_track_RMS_m':float(np.sqrt(np.mean(alignment['cross_track_m']**2))),
        'coordinate_fit_RMS_m':float(np.sqrt(np.mean(alignment['coordinate_residual_m']**2))),
        'extension_km':float(extend_km),'D_range_km':[float(dmin/1000),float(dmax/1000)],
        'sampling_spacing_target_m':float(spacing_m),'sample_count':len(d),'valid_sample_count':int(valid.sum()),
        'surface_at_nominal_D0_ellipsoid_m':z0,
        'surface_at_extension_endpoint_ellipsoid_m':float(elevation[0]) if valid[0] else None,
        'extension_minimum_surface_ellipsoid_m':float(elevation[below].min()),
        'extension_minimum_D_km':float(d[below][np.argmin(elevation[below])]/1000),
        'station_DEM_minus_table_E_mean_m':float(delta.mean()),
        'station_DEM_minus_table_E_RMS_m':float(np.sqrt(np.mean(delta**2))),
        'station_DEM_minus_table_E_max_abs_m':float(np.abs(delta).max()),
        'interpretation':[
            'This is a straight continuation of the survey alignment, not a traced ice-flow path.',
            'Negative D extends beyond nominal Site 12, positive D points inland.',
            'Published station coordinates have 0.001-degree precision and are not exactly collinear; residuals are exported.',
            'The GIMP DEM has nominal 2007 epoch; the survey is from 2016. No epoch or vertical-datum shift is fitted.',
            'Only the surface is extended. Aquifer top/base beyond the observed stations are not inferred.'
        ]
    }
    return records,validation,summary


def write_csv(path,records):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def add_parallel_base(records, geometry_path, summary):
    """Join the observed base to a surface-parallel continuation at Site 12.

    Observed/interpolated base elevations remain in the paper's reported
    elevation convention. Only DEM elevation differences are used beyond
    the final observed station; no vertical-datum correction is inferred.
    """
    geometry=json.loads(Path(geometry_path).read_text())
    stations=sorted(geometry['stations'],key=lambda r:r['D_km'])
    station_d=np.array([s['D_km'] for s in stations],dtype=float)
    station_b=np.array([s['base_elevation_est_m_asl'] for s in stations],dtype=float)
    d=np.array([r['D_km'] for r in records],dtype=float)
    surface=np.array([r['surface_elevation_wgs84_ellipsoid_m'] for r in records],dtype=float)
    if not np.isfinite(station_b).all() or np.any(np.diff(station_d)<=0):
        raise ValueError('Base stations must be finite and ordered by unique D.')
    if d.max()>station_d[-1]+1e-10:
        raise ValueError('No upglacier base extrapolation is defined.')
    anchor=stations[0]
    anchor_indices=np.flatnonzero(np.isclose(d,station_d[0],rtol=0,atol=1e-10))
    if len(anchor_indices)!=1 or not np.isfinite(surface[anchor_indices[0]]):
        raise ValueError('A valid DEM sample at the Site 12 base anchor is required.')
    anchor_surface=float(surface[anchor_indices[0]])
    anchor_base=float(station_b[0])
    assumed=d<station_d[0]
    base=np.interp(d,station_d,station_b)
    base[assumed]=anchor_base+surface[assumed]-anchor_surface
    output=[]
    for r,z,b,is_assumed in zip(records,surface,base,assumed):
        output.append(dict(r,
            base_elevation_model_m=float(b) if np.isfinite(b) else None,
            base_depth_below_dem_surface_m=float(z-b) if np.isfinite(z-b) else None,
            base_is_assumed=int(is_assumed),
            base_model_source='surface_parallel_assumption' if is_assumed else 'figure4_table1_interpolation'))
    summary['base_model']={
        'equation':'z_base(D) = z_base_anchor + z_surface_DEM(D) - z_surface_DEM(anchor), D < anchor',
        'assumption':'Constant surface-to-base separation; base has the same local slope as the DEM surface.',
        'source_geometry_file':str(Path(geometry_path).resolve()),
        'anchor_site':anchor['site'],'anchor_D_km':float(station_d[0]),
        'anchor_base_elevation_m_as_reported':anchor_base,
        'anchor_surface_DEM_ellipsoid_m':anchor_surface,
        'anchor_surface_table1_m_as_reported':float(anchor['surface_elevation_m_asl']),
        'anchor_published_figure4_base_depth_m':float(anchor['base_depth_est_m']),
        'effective_constant_depth_below_DEM_m':anchor_surface-anchor_base,
        'anchor_DEM_minus_reported_surface_m':anchor_surface-float(anchor['surface_elevation_m_asl']),
        'assumed_base_at_extension_endpoint_m':float(base[0]) if np.isfinite(base[0]) else None,
        'observed_base_stations':[{'site':s['site'],'D_km':float(s['D_km']),
            'base_elevation_m':float(s['base_elevation_est_m_asl'])} for s in stations],
        'notes':[
            'The constant DEM-to-base separation includes the DEM/Table 1 surface difference at the anchor.',
            'This preserves the extracted base elevation and makes the join continuous.',
            'The assumed segment includes the 0 to 0.41 km gap between nominal Site 12 and Site 12(a).',
            'No aquifer-top elevation or saturated thickness is extrapolated.'
        ]
    }
    summary['interpretation'][-1]='An assumed surface-parallel base is extended beyond Site 12(a); aquifer top and saturated thickness remain unspecified there.'
    return output


def add_aquifer(records, geometry_path, summary):
    """Add the extracted top through D = 0 and height above the same base.

    Keep the extracted absolute elevations between survey stations. Across
    0 <= D < 0.41 km, continue the top parallel to the existing base, with
    the Site 12(a) thickness. No aquifer geometry is defined for D < 0.
    """
    geometry=json.loads(Path(geometry_path).read_text())
    stations=sorted(geometry['stations'],key=lambda r:r['D_km'])
    station_d=np.array([s['D_km'] for s in stations],dtype=float)
    station_top=np.array([s['top_elevation_est_m_asl'] for s in stations],dtype=float)
    station_base=np.array([s['base_elevation_est_m_asl'] for s in stations],dtype=float)
    if (not np.isfinite(np.r_[station_d,station_top,station_base]).all()
            or np.any(np.diff(station_d)<=0) or np.any(station_top<=station_base)):
        raise ValueError('Aquifer stations must be finite, ordered, and have positive thickness.')
    d=np.array([r['D_km'] for r in records],dtype=float)
    surface=np.array([r['surface_elevation_wgs84_ellipsoid_m'] for r in records],dtype=float)
    base=np.array([r['base_elevation_model_m'] for r in records],dtype=float)
    if d.max()>station_d[-1]+1e-10:
        raise ValueError('No upglacier aquifer extrapolation is defined.')
    defined=(d>=0)&(d<=station_d[-1])
    gap=defined&(d<station_d[0])
    top=np.full(d.shape,np.nan)
    top[defined]=np.interp(d[defined],station_d,station_top)
    anchor_thickness=float(station_top[0]-station_base[0])
    top[gap]=base[gap]+anchor_thickness
    thickness=top-base
    if np.any(thickness[defined]<=0):
        raise ValueError('Interpolated aquifer thickness must remain positive.')
    output=[]
    for r,z,b,t,h,is_defined,is_gap in zip(records,surface,base,top,thickness,defined,gap):
        output.append(dict(r,
            surface_height_above_base_m=float(z-b) if np.isfinite(z-b) else None,
            aquifer_geometry_defined=int(is_defined),
            aquifer_top_elevation_model_m=float(t) if np.isfinite(t) else None,
            aquifer_top_height_above_base_m=float(h) if np.isfinite(h) else None,
            aquifer_thickness_model_m=float(h) if np.isfinite(h) else None,
            aquifer_top_is_assumed=int(is_gap) if is_defined else None,
            aquifer_geometry_source=('site12a_constant_thickness_gap_fill' if is_gap
                else 'figure4_table1_interpolation' if is_defined else '')))
    summary['aquifer_model']={
        'source_geometry_file':str(Path(geometry_path).resolve()),
        'D_range_km':[0.,float(station_d[-1])],
        'top_method':'Linear interpolation of extracted Figure 4 top elevations between stations.',
        'gap_D_range_km':[0.,float(station_d[0])],
        'gap_method':'Top parallels the surface-parallel base with constant Site 12(a) thickness.',
        'gap_thickness_m':anchor_thickness,
        'height_coordinate':'height = elevation - local model base elevation; base = 0 m',
        'observed_stations':[{'site':s['site'],'D_km':float(s['D_km']),
            'surface_elevation_m':float(s['surface_elevation_m_asl']),
            'top_elevation_m':float(s['top_elevation_est_m_asl']),
            'base_elevation_m':float(s['base_elevation_est_m_asl']),
            'thickness_m':float(s['top_elevation_est_m_asl']-s['base_elevation_est_m_asl'])}
            for s in stations],
        'notes':[
            geometry['precision'], geometry['site5_choice'],
            'Top and base are preserved in the source elevation convention; DEM heights are not datum corrected.',
            'The 410 m gap to D = 0 is assumed, not observed.',
            'Aquifer geometry is undefined for D < 0; this does not establish its physical absence.',
            'Panel c subtracts the same local base from every elevation in panel b.'
        ]
    }
    summary['base_model']['notes'][-1]='Aquifer top is gap-filled only to D = 0; saturated thickness is unspecified for D < 0.'
    summary['interpretation'][-1]='The base continues surface-parallel below Site 12(a); aquifer top is filled only to D = 0 at constant Site 12(a) thickness.'
    return output


def write_geojson(path,records,validation,summary):
    features=[{'type':'Feature','properties':{'name':'Straight survey alignment and downglacier continuation','D_min_km':records[0]['D_km'],'D_max_km':records[-1]['D_km']},'geometry':{'type':'LineString','coordinates':[[r['longitude_deg_east'],r['latitude_deg_north']] for r in records]}}]
    for r in validation:
        features.append({'type':'Feature','properties':r,'geometry':{'type':'Point','coordinates':[r['longitude_deg_east'],r['latitude_deg_north']]}})
    features.append({'type':'Feature','properties':{'name':'Nominal Site 12 D=0','surface_elevation_wgs84_ellipsoid_m':summary['surface_at_nominal_D0_ellipsoid_m']},'geometry':{'type':'Point','coordinates':[summary['nominal_D0_longitude'],summary['nominal_D0_latitude']]}})
    Path(path).write_text(json.dumps({'type':'FeatureCollection','features':features},indent=2,allow_nan=False)+'\n')


def plot_transect(dataset,records,validation,summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from rasterio.enums import Resampling
    d=np.array([r['D_km'] for r in records])
    z=np.array([r['surface_elevation_wgs84_ellipsoid_m'] if r['surface_elevation_wgs84_ellipsoid_m'] is not None else np.nan for r in records])
    x=np.array([r['dem_x_epsg3413_m'] for r in records]); y=np.array([r['dem_y_epsg3413_m'] for r in records])
    bounds=(x.min()-5000,y.min()-6000,x.max()+5000,y.max()+6000)
    window=from_bounds(*bounds,transform=dataset.transform).round_offsets().round_lengths()
    ratio=max(window.width/1300,window.height/600,1)
    data=dataset.read(1,window=window,out_shape=(max(2,int(window.height/ratio)),max(2,int(window.width/ratio))),masked=True,resampling=Resampling.bilinear).astype(float).filled(np.nan)
    actual_bounds=rasterio.windows.bounds(window,dataset.transform)
    fig,axes=plt.subplots(3,1,figsize=(14,14),gridspec_kw={'height_ratios':[1,1.2,1]})
    fig.suptitle('Helheim firn aquifer: along-transect geometry',x=.075,y=.985,
        ha='left',fontsize=21,fontweight='semibold',color='#203746')
    for panel in axes:
        panel.tick_params(labelsize=12,colors='#344653',length=5)
        for spine in panel.spines.values():
            spine.set_color('#9AAAB5')
            spine.set_linewidth(.8)
    ax=axes[0]
    extent=[actual_bounds[0]/1000,actual_bounds[2]/1000,actual_bounds[1]/1000,actual_bounds[3]/1000]
    shade=LightSource(azdeg=315,altdeg=40).hillshade(np.nan_to_num(data,nan=np.nanmedian(data)),dx=dataset.res[0]*ratio,dy=dataset.res[1]*ratio)
    ax.imshow(shade,extent=extent,cmap='gray',origin='upper',vmin=0,vmax=1)
    elevation=ax.imshow(data,extent=extent,cmap='gist_earth',origin='upper',alpha=.38)
    cbar=fig.colorbar(elevation,cax=ax.inset_axes([1.015,0,.018,1]))
    cbar.set_label('Surface elevation (m, WGS84 ellipsoid)',fontsize=12,labelpad=10)
    cbar.ax.tick_params(labelsize=11)
    for mask,color,label in [(d>=0,'#037E9B','Survey alignment'),(d<=0,'#D86428','Downglacier continuation')]:
        ax.plot(x[mask]/1000,y[mask]/1000,color=color,lw=2.8,label=label)
    trans=Transformer.from_crs(4326,dataset.crs,always_xy=True)
    sx,sy=trans.transform([r['longitude_deg_east'] for r in validation],[r['latitude_deg_north'] for r in validation])
    ax.scatter(np.array(sx)/1000,np.array(sy)/1000,s=22,c='#182B35',zorder=5,label='2016 survey sites')
    for r,xx,yy in zip(validation,sx,sy):
        if r['site'] in ('1','5','8 (a)','12 (a)'):
            ax.annotate(r['site'],(xx/1000,yy/1000),xytext=(0,11),textcoords='offset points',fontsize=11,fontweight='medium',ha='center',color='#142D3B')
    zero=np.flatnonzero(d==0)[0]
    ax.scatter([x[zero]/1000],[y[zero]/1000],marker='*',s=145,c='#EDB84B',edgecolors='#182B35',linewidths=.7,zorder=6,label='Nominal Site 12, D = 0')
    ax.set_xlabel('Easting (km, EPSG:3413)',fontsize=13,labelpad=8)
    ax.set_ylabel('Northing (km)',fontsize=13,labelpad=8)
    ax.set_title(f"a   Survey alignment and {summary['extension_km']:g} km continuation",loc='left',fontsize=15,fontweight='semibold',pad=14,color='#203746')
    map_legend=ax.legend(loc='lower left',fontsize=10.5,framealpha=.95,borderpad=.6,labelspacing=.45)
    map_legend.get_frame().set_edgecolor('#D8E2E8')
    ax.set_aspect('equal')
    # Geographic north at the arrow location, accounting for map convergence.
    xmin,xmax,ymin,ymax=extent
    width,height=xmax-xmin,ymax-ymin
    to_geographic=Transformer.from_crs(dataset.crs,4326,always_xy=True)
    geod=Geod(ellps='WGS84')
    arrow_start=np.array([xmin+.91*width,ymin+.65*height])
    north_lon,north_lat=to_geographic.transform(*(arrow_start*1000))
    north_lon_end,north_lat_end,_=geod.fwd(north_lon,north_lat,0,2000)
    nx,ny=trans.transform(north_lon_end,north_lat_end)
    north_vector=np.array([nx,ny])/1000-arrow_start
    north_vector=north_vector/np.linalg.norm(north_vector)
    arrow_tip=arrow_start+north_vector*(.22*height)
    ax.annotate('',xy=arrow_tip,xytext=arrow_start,
        arrowprops={'arrowstyle':'-|>','color':'#F7FAFC','lw':2.2,'mutation_scale':22},zorder=8)
    ax.text(arrow_tip[0],arrow_tip[1]+.025*height,'N',ha='center',va='bottom',
        fontsize=16,fontweight='semibold',color='#F7FAFC',zorder=9)
    # A 5 km geodesic distance expressed as a horizontal segment on this map.
    scale_x,scale_y=xmin+.77*width,ymin+.11*height
    scale_lon,scale_lat=to_geographic.transform(scale_x*1000,scale_y*1000)
    lo,hi=0.,10.
    for _ in range(32):
        length=(lo+hi)/2
        end_lon,end_lat=to_geographic.transform((scale_x+length)*1000,scale_y*1000)
        ground_distance=geod.inv(scale_lon,scale_lat,end_lon,end_lat)[2]
        if ground_distance<5000:
            lo=length
        else:
            hi=length
    scale_length=(lo+hi)/2
    ax.plot([scale_x,scale_x+scale_length],[scale_y,scale_y],color='#142D3B',lw=2.5,zorder=8)
    for tick_x in (scale_x,scale_x+scale_length):
        ax.plot([tick_x,tick_x],[scale_y-.018*height,scale_y+.018*height],color='#142D3B',lw=1.5,zorder=8)
    ax.text(scale_x+scale_length/2,scale_y-.035*height,'5 km',ha='center',va='top',
        fontsize=13,fontweight='medium',color='#142D3B',zorder=9)
    ax=axes[1]
    ax.set_title('b   Surface and aquifer elevations',loc='left',fontsize=15,fontweight='semibold',pad=14,color='#203746')
    ax.plot(d,z,color='#47545E',lw=2,label='DEM surface',zorder=3)
    ax.plot(d[d<=0],z[d<=0],color='#D86428',lw=2.2,label='DEM extension',zorder=3)
    ax.scatter([r['D_km'] for r in validation],[r['table1_E_m_as_reported'] for r in validation],s=34,c='#037E9B',edgecolors='white',linewidths=.5,zorder=5,label='Survey surface elevations')
    if 'base_model' in summary:
        base=np.array([r['base_elevation_model_m'] for r in records],dtype=float)
        anchor_d=summary['base_model']['anchor_D_km']
        observed=d>=anchor_d; assumed=d<=anchor_d
        ax.fill_between(d,base,z,color='#DCE9EF',alpha=.45,zorder=1)
        ax.plot(d[observed],base[observed],color='#235A83',lw=2,label='Extracted base',zorder=4)
        ax.plot(d[assumed],base[assumed],color='#235A83',lw=2.2,ls=(0,(5,3)),label='Assumed parallel base',zorder=4)
        base_points=summary['base_model']['observed_base_stations']
        ax.scatter([s['D_km'] for s in base_points],[s['base_elevation_m'] for s in base_points],s=27,c='#235A83',edgecolors='white',linewidths=.5,zorder=5)
    if 'aquifer_model' in summary:
        top=np.array([r['aquifer_top_elevation_model_m'] for r in records],dtype=float)
        aquifer=(d>=0)&np.isfinite(top)
        measured=aquifer&(d>=anchor_d)
        gap=aquifer&(d<=anchor_d)
        ax.fill_between(d,base,top,where=aquifer,color='#37A9C4',alpha=.65,
            label='Firn aquifer (to D = 0)',zorder=2)
        ax.plot(d[measured],top[measured],color='#037E9B',lw=1.5,label='Aquifer top',zorder=4)
        ax.plot(d[gap],top[gap],color='#037E9B',lw=1.5,ls=(0,(3,2)),zorder=4)
        ax.plot([0,0],[base[zero],top[zero]],color='#037E9B',lw=1.3,zorder=5)
    ax.axvline(0,color='#8A969F',ls=(0,(3,3)),lw=1.2)
    ax.text(.985,.96,'Downglacier extension: assumed surface-parallel base',transform=ax.transAxes,
        ha='right',va='top',fontsize=11.5,color='#235A83')
    ax.annotate('D = 0',(0,z[zero]),xytext=(8,17),textcoords='offset points',fontsize=12,color='#344653')
    ax.set_xlabel('Signed distance D from nominal Site 12 (km)',fontsize=14,labelpad=10)
    ax.set_ylabel('Elevation (m)',fontsize=14,labelpad=10)
    ax.set_xlim(d.max(),d.min())
    ax.grid(axis='y',alpha=.25,color='#AAB8C2',linewidth=.7)
    ax.spines[['top','right']].set_visible(False)
    profile_legend=ax.legend(loc='lower left',fontsize=10.5,ncol=2,framealpha=.95,
        borderpad=.7,labelspacing=.6,columnspacing=1.4,handlelength=2.6)
    profile_legend.get_frame().set_edgecolor('#D8E2E8')
    ax=axes[2]
    ax.set_title('c   Height above the local aquifer base',loc='left',fontsize=15,fontweight='semibold',pad=14,color='#203746')
    surface_height=z-base
    top_height=top-base
    ax.fill_between(d,0,surface_height,color='#DCE9EF',alpha=.45,zorder=1)
    ax.fill_between(d,0,top_height,where=aquifer,color='#37A9C4',alpha=.65,
        label='Firn aquifer',zorder=2)
    ax.plot(d,surface_height,color='#47545E',lw=2,label='DEM surface',zorder=3)
    ax.plot(d[d<=0],surface_height[d<=0],color='#D86428',lw=2.2,zorder=3)
    ax.plot(d[measured],top_height[measured],color='#037E9B',lw=1.8,label='Aquifer top',zorder=4)
    ax.plot(d[gap],top_height[gap],color='#037E9B',lw=1.8,ls=(0,(3,2)),zorder=4)
    ax.plot([0,0],[0,top_height[zero]],color='#037E9B',lw=1.3,zorder=4)
    station_points=summary['aquifer_model']['observed_stations']
    ax.scatter([s['D_km'] for s in station_points],[s['thickness_m'] for s in station_points],
        s=27,c='#037E9B',edgecolors='white',linewidths=.5,zorder=5)
    ax.plot(d[observed],np.zeros(observed.sum()),color='#235A83',lw=2,zorder=4)
    ax.plot(d[assumed],np.zeros(assumed.sum()),color='#235A83',lw=2.2,ls=(0,(5,3)),zorder=4)
    ax.axvline(0,color='#8A969F',ls=(0,(3,3)),lw=1.2)
    ax.text(.985,.48,'Aquifer shown through D = 0',transform=ax.transAxes,
        ha='right',va='center',fontsize=11.5,color='#235A83')
    ax.text(.985,.065,'Base = 0 m',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=11.5,color='#235A83')
    ax.set_xlim(d.max(),d.min())
    ax.set_ylim(-1,5*np.ceil(np.nanmax(surface_height)/5)+10)
    ax.set_yticks(np.arange(0,ax.get_ylim()[1]+.1,10))
    ax.set_xlabel('Signed distance D from nominal Site 12 (km)',fontsize=14,labelpad=10)
    ax.set_ylabel('Height above the base (m)',fontsize=14,labelpad=10)
    ax.grid(axis='y',alpha=.25,color='#AAB8C2',linewidth=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',fontsize=10.5,ncol=3,frameon=False,
        columnspacing=1.7,handlelength=2.6)
    fig.text(.075,.047,'Aquifer: extracted Figure 4 top and base; the 410 m gap to D = 0 retains the Site 12(a) thickness (20.5 m).',fontsize=10.5,color='#51636F')
    fig.text(.075,.028,'Dashed base: assumed surface-parallel continuation beyond Site 12(a). Panel c uses height = elevation − local base.',fontsize=10.5,color='#51636F')
    fig.text(.075,.009,'Surface: local GIMP DEM (30 m posting, nominal 2007). Extracted boundary elevations retain the paper’s vertical reference.',fontsize=10.5,color='#51636F')
    fig.tight_layout(rect=(0,.078,1,.958),h_pad=3)
    fig.savefig(output/'montgomery_surface_transect.png',dpi=180)
    fig.savefig(output/'montgomery_surface_transect.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem',type=Path,default=DEFAULT_DEM)
    parser.add_argument('--table',type=Path,default=DEFAULT_TABLE)
    parser.add_argument('--geometry',type=Path,default=DEFAULT_GEOMETRY)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--extend-km',type=float,default=20)
    parser.add_argument('--spacing-m',type=float,default=30)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    rows=load_survey(args.table); alignment=fit_alignment(rows)
    with rasterio.open(args.dem) as dem:
        if dem.crs.to_epsg()!=3413:
            raise ValueError('Expected local GIMP DEM in EPSG:3413.')
        records,validation,summary=build_transect(dem,rows,alignment,args.extend_km,args.spacing_m)
        base_records=add_parallel_base(records,args.geometry,summary)
        aquifer_records=add_aquifer(base_records,args.geometry,summary)
        args.output.mkdir(parents=True,exist_ok=True)
        write_csv(args.output/'montgomery_surface_transect.csv',records)
        write_csv(args.output/'montgomery_surface_and_base_transect.csv',base_records)
        write_csv(args.output/'montgomery_surface_base_and_aquifer_transect.csv',aquifer_records)
        write_csv(args.output/'station_DEM_comparison.csv',validation)
        write_geojson(args.output/'montgomery_surface_transect.geojson',records,validation,summary)
        (args.output/'transect_metadata.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        if args.plot:
            plot_transect(dem,aquifer_records,validation,summary,args.output)
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
