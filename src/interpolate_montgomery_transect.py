"""Interpolate model inputs along the saved Montgomery/GIMP transect.

x = -1000 D is in metres, positive downglacier from nominal Site 12.
Aquifer height is saturated thickness and is prescribed as zero for x > 0.
NumPy and the standard library suffice; --plot additionally needs Matplotlib.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSECT = ROOT/'outputs/firn_aquifer/surface_transect/montgomery_surface_base_and_aquifer_transect.csv'
DEFAULT_GEOMETRY = ROOT/'outputs/firn_aquifer/montgomery2016_geometry_stations.json'
DEFAULT_OUTPUT = ROOT/'outputs/firn_aquifer/interpolation'


def write_csv(path, columns):
    names=list(columns)
    with Path(path).open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))


class TransectInterpolator:
    """Evaluate latitude, longitude, base, slope and thickness at arbitrary x.

    Elevation control points are the 11 extracted stations and all saved
    DEM-derived base points below Site 12(a), including D = 0. Coordinates
    belong to the fitted straight transect, rather than off-line GPS points.
    Slope estimates at controls average the adjacent segment angles, then
    are linearly interpolated. The geometric derivative of the piecewise
    linear base is also returned separately. Both angles are positive downhill.
    """

    def __init__(self,transect_path=DEFAULT_TRANSECT,geometry_path=DEFAULT_GEOMETRY):
        self.transect_path=Path(transect_path).resolve()
        self.geometry_path=Path(geometry_path).resolve()
        with self.transect_path.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.geometry=json.loads(self.geometry_path.read_text())
        self.stations=sorted(self.geometry['stations'],key=lambda r:r['D_km'])
        station_d=np.array([s['D_km'] for s in self.stations],dtype=float)
        if len(station_d)<2 or np.any(np.diff(station_d)<=0):
            raise ValueError('At least two stations with unique increasing D are required.')
        self.anchor_x_m=-1000*station_d[0]
        self.anchor_height_m=float(self.stations[0]['thickness_figure4_est_m'])
        d=np.array([float(r['D_km']) for r in rows])
        if not np.isfinite(d).all() or len(np.unique(d))!=len(d):
            raise ValueError('Transect D must be finite and unique.')
        controls=d<=station_d[0]
        for dist in station_d:
            matches=np.flatnonzero(np.isclose(d,dist,rtol=0,atol=1e-10))
            if len(matches)!=1:
                raise ValueError(f'Exactly one transect sample at station D={dist} is required.')
            controls[matches[0]]=True
        selected=[r for r,keep in zip(rows,controls) if keep]
        selected.sort(key=lambda r:-1000*float(r['D_km']))
        self.x=np.array([-1000*float(r['D_km']) for r in selected])
        self.lat=np.array([float(r['latitude_deg_north']) for r in selected])
        self.lon=np.array([float(r['longitude_deg_east']) for r in selected])
        self.base=np.array([float(r['base_elevation_model_m']) for r in selected])
        self.height=np.array([0. if xx>0 else float(r['aquifer_thickness_model_m'])
            for xx,r in zip(self.x,selected)])
        if (not np.isfinite(np.r_[self.x,self.lat,self.lon,self.base,self.height]).all()
                or np.any(np.diff(self.x)<=0) or not self.x[0]<0<self.x[-1]
                or np.count_nonzero(self.x==0)!=1):
            raise ValueError('Finite ordered controls spanning both sides of x=0 are required.')
        if np.any(self.height[self.x<=0]<=0) or np.any(self.height[self.x>0]!=0):
            raise ValueError('Require positive aquifer height through x=0 and zero height for x>0.')
        self.segment_gradient=np.diff(self.base)/np.diff(self.x)
        self.segment_slope=np.degrees(np.arctan(-self.segment_gradient))
        self.slope=np.r_[self.segment_slope[0],
            .5*(self.segment_slope[:-1]+self.segment_slope[1:]),self.segment_slope[-1]]

    def evaluate(self,x_m):
        """Return named 1-D arrays; reject extrapolation and preserve the x=0 jump.

        slope_deg: linearly interpolated nodal angle estimate (positive downhill).
        geometric_slope_deg: exact segment angle of the interpolated base.
        At a control knot the latter uses the downstream/right segment; at
        the final endpoint it uses the last available segment.
        """
        x=np.atleast_1d(np.asarray(x_m,dtype=float))
        if x.ndim!=1 or not np.isfinite(x).all():
            raise ValueError('x must be a finite scalar or one-dimensional array.')
        if np.any(x<self.x[0]) or np.any(x>self.x[-1]):
            raise ValueError(f'x must lie in [{self.x[0]}, {self.x[-1]}] m; extrapolation is disabled.')
        height=np.zeros(x.shape)
        upstream=x<=0
        height[upstream]=np.interp(x[upstream],self.x[self.x<=0],self.height[self.x<=0])
        segment=np.clip(np.searchsorted(self.x,x,side='right')-1,0,len(self.segment_gradient)-1)
        base=np.interp(x,self.x,self.base)
        return dict(
            x_m=x,
            latitude_deg_north=np.interp(x,self.x,self.lat),
            longitude_deg_east=np.interp(x,self.x,self.lon),
            base_elevation_m=base,
            slope_deg=np.interp(x,self.x,self.slope),
            aquifer_height_m=height,
            D_km=-x/1000,
            geometric_slope_deg=self.segment_slope[segment],
            base_dz_dx=self.segment_gradient[segment],
            aquifer_top_elevation_m=base+height,
            base_is_assumed=(x>self.anchor_x_m).astype(int),
            base_source=np.where(x>self.anchor_x_m,'surface_parallel_DEM_assumption','figure4_station_interpolation'),
            aquifer_height_source=np.where(x>0,'zero_prescribed',np.where(
                x>self.anchor_x_m,'site12a_constant_height_gap_fill','figure4_station_interpolation'))
        )

    def regular_grid(self,spacing_m=30.):
        if not np.isfinite(spacing_m) or spacing_m<=0:
            raise ValueError('Spacing must be finite and positive.')
        if (self.x[-1]-self.x[0])/spacing_m>1_000_000:
            raise ValueError('Requested interpolation grid exceeds one million samples.')
        return np.unique(np.r_[-np.arange(0,-self.x[0],spacing_m),
            np.arange(0,self.x[-1],spacing_m),self.x[0],self.x[-1],
            [-1000*s['D_km'] for s in self.stations],0.])

    def metadata(self):
        return dict(
            source_transect=str(self.transect_path),source_geometry=str(self.geometry_path),
            source_citation=self.geometry['source'],
            x_definition='x_m = -1000 * D_km; x=0 at nominal Site 12; x increases downglacier',
            x_range_m=[float(self.x[0]),float(self.x[-1])],
            original_initializer_coordinate='The earlier observed-only initializer uses x_D_m = 1000 D; this export uses the opposite sign.',
            interpolation='Piecewise linear in x; no extrapolation outside the saved transect.',
            location='WGS84 latitude/longitude on the fitted straight transect; not the individual off-line published station coordinates.',
            elevation='Extracted base elevations retained as reported; below Site 12(a), the existing DEM-surface-parallel base is retained.',
            elevation_controls='11 extracted station bases plus all saved assumed-base samples below Site 12(a), including x=0.',
            slope_deg='Positive downhill with increasing x. At each internal base control, mean of adjacent segment angles; endpoints use the nearest segment. These nodal angle estimates are linearly interpolated in x.',
            geometric_slope_deg='atan(-dz_base/dx), in degrees, from each straight base segment. At a knot, downstream/right-sided; final endpoint uses the last segment.',
            slope_note='Slope is derived from digitized/model base elevations, not directly measured. Interpolating slope independently does not make it the exact derivative of the linearly interpolated base. Use geometric_slope_deg when exact geometric consistency is needed.',
            base_dz_dx='Signed rise/run in m/m; negative for a base that descends as x increases.',
            aquifer_height_m='Vertical saturated thickness, linearly interpolated between Figure 4 station thicknesses. The 410 m gap to x=0 retains 20.5 m; x>0 is exactly zero as prescribed.',
            aquifer_at_origin='H(0)=20.5 m, H(x>0)=0: an explicit discontinuity with no artificial taper.',
            aquifer_top_elevation_m='base_elevation_m + aquifer_height_m; coincides with base where prescribed height is zero.',
            gap_x_range_m=[float(self.anchor_x_m),0.],
            control_point_count=len(self.x),
            site5_choice=self.geometry['site5_choice'],
            precision=self.geometry['precision'],
            vertical_reference_note='Existing source/DEM elevation conventions are preserved; no new geoid or epoch correction is made.'
        )


def plot_profiles(model,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.unique(np.r_[model.regular_grid(30),model.x])
    p=model.evaluate(x)
    fig,axes=plt.subplots(3,1,figsize=(13,10.5),sharex=True)
    fig.suptitle('Firn-aquifer inputs along the transect',x=.08,y=.98,ha='left',
        fontsize=21,fontweight='semibold',color='#203746')
    axes[0].plot(x/1000,p['base_elevation_m'],lw=2,color='#235A83')
    station_x=np.array([-1000*s['D_km'] for s in model.stations])
    axes[0].scatter(station_x/1000,[s['base_elevation_est_m_asl'] for s in model.stations],
        s=30,color='#235A83',edgecolors='white',linewidths=.5,zorder=4,label='Extracted stations')
    axes[0].set_ylabel('Base elevation (m)',fontsize=13)
    axes[0].set_title('a   Linearly interpolated base',loc='left',fontsize=15,fontweight='semibold',color='#203746')
    axes[0].legend(frameon=False,fontsize=11)
    axes[1].stairs(model.segment_slope,model.x/1000,color='#A6B4BF',lw=1.2,
        baseline=None,label='Slope of each straight base segment')
    axes[1].plot(x/1000,p['slope_deg'],color='#037E9B',lw=1.8,label='Linearly interpolated slope estimates')
    axes[1].set_ylabel('Slope (degrees)',fontsize=13)
    axes[1].set_title('b   Positive slope is downhill in the +x direction',loc='left',fontsize=15,fontweight='semibold',color='#203746')
    axes[1].legend(frameon=False,fontsize=10.5,loc='upper left')
    upstream=x<=0
    axes[2].fill_between(x[upstream]/1000,0,p['aquifer_height_m'][upstream],color='#37A9C4',alpha=.65)
    axes[2].plot(x[upstream]/1000,p['aquifer_height_m'][upstream],color='#037E9B',lw=2)
    axes[2].plot([0,model.x[-1]/1000],[0,0],color='#D86428',lw=2)
    axes[2].plot([0,0],[0,model.anchor_height_m],color='#037E9B',lw=1.3,ls=':')
    axes[2].scatter([0],[model.anchor_height_m],s=25,color='#037E9B',zorder=4)
    axes[2].scatter([0],[0],s=25,facecolors='white',edgecolors='#D86428',zorder=4)
    axes[2].text(.98,.68,'Aquifer height = 0 for x > 0',transform=axes[2].transAxes,
        ha='right',fontsize=12,color='#D86428')
    axes[2].set_ylabel('Aquifer height (m)',fontsize=13)
    axes[2].set_ylim(-1,25)
    axes[2].set_title('c   Aquifer height (saturated thickness)',loc='left',fontsize=15,fontweight='semibold',color='#203746')
    axes[2].set_xlabel('x (km): positive downglacier from nominal Site 12',fontsize=14,labelpad=8)
    for ax in axes:
        ax.axvline(0,color='#8A969F',ls=(0,(3,3)),lw=1.1)
        ax.grid(axis='y',alpha=.25,color='#AAB8C2',lw=.7)
        ax.tick_params(labelsize=11,colors='#344653')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#9AAAB5')
        ax.set_xlim(model.x[0]/1000,model.x[-1]/1000)
    fig.text(.08,.025,'x = −1000D. The 410 m gap to the origin retains Site 12(a) thickness; zero height for x > 0 is prescribed.',fontsize=10.5,color='#51636F')
    fig.tight_layout(rect=(0,.055,1,.95),h_pad=2)
    fig.savefig(output/'montgomery_interpolated_inputs.png',dpi=180)
    fig.savefig(output/'montgomery_interpolated_inputs.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transect',type=Path,default=DEFAULT_TRANSECT)
    parser.add_argument('--geometry',type=Path,default=DEFAULT_GEOMETRY)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--spacing-m',type=float,default=30)
    parser.add_argument('--x-m',type=float,nargs='+',help='Evaluate these x values instead of the default grid.')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    model=TransectInterpolator(args.transect,args.geometry)
    x=np.asarray(args.x_m) if args.x_m is not None else model.regular_grid(args.spacing_m)
    values=model.evaluate(x)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'montgomery_interpolated_model_inputs.csv',values)
    write_csv(args.output/'montgomery_interpolation_controls.csv',model.evaluate(model.x))
    np.savez_compressed(args.output/'montgomery_interpolated_model_inputs.npz',**values)
    metadata=model.metadata()
    metadata.update(output_sample_count=len(x),
        grid='explicit query x values' if args.x_m is not None else 'regular spacing plus station points, origin and domain endpoints',
        target_spacing_m=args.spacing_m if args.x_m is None else None)
    (args.output/'interpolation_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    if args.plot:
        plot_profiles(model,args.output)
    print(json.dumps({'output':str(args.output.resolve()),'samples':len(x),
        'control_points':len(model.x),'x_range_m':metadata['x_range_m'],
        'H_at_zero_m':float(model.evaluate(0)['aquifer_height_m'][0]),
        'H_for_positive_x':'exactly zero','slope_convention':'positive downhill'},indent=2))


if __name__=='__main__':
    main()
