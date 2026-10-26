"""Map the complete Feature I observation region using the local GIMP DEM."""

from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helheim_geometry_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/helheim_geometry_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import plotly.graph_objects as go

from reconstruct_helheim_feature1 import DEFAULT_DEM, sample_dem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/firn_aquifer/gimp_region"
RECON = ROOT / "outputs/firn_aquifer/feature1_3d"
SPACING = 30.
BUFFER = 5000.
EXAGGERATION = 15.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The full thickness-point extent is used, not only the screened mesh patches.
    with (RECON / "chu_source_points.csv").open(newline="") as f:
        chu = list(csv.DictReader(f))
    lon = np.array([float(r["longitude_deg"]) for r in chu])
    lat = np.array([float(r["latitude_deg"]) for r in chu])
    with (ROOT / "outputs/firn_aquifer/montgomery2017_table1.csv").open(newline="") as f:
        stations = [r for r in csv.DictReader(f) if r["year"] == "2016"]
    forward = Transformer.from_crs(4326, 32624, always_xy=True)
    inverse = Transformer.from_crs(32624, 4326, always_xy=True)
    cx, cy = forward.transform(lon, lat)
    sx, sy = forward.transform(
        [float(r["longitude_deg_east"]) for r in stations],
        [float(r["latitude_deg_north"]) for r in stations])
    sx, sy = np.asarray(sx), np.asarray(sy)
    west = np.floor((min(cx.min(),sx.min())-BUFFER)/SPACING)*SPACING
    east = np.ceil((max(cx.max(),sx.max())+BUFFER)/SPACING)*SPACING
    south = np.floor((min(cy.min(),sy.min())-BUFFER)/SPACING)*SPACING
    north = np.ceil((max(cy.max(),sy.max())+BUFFER)/SPACING)*SPACING
    nx, ny = int(round((east-west)/SPACING)), int(round((north-south)/SPACING))
    transform = from_origin(west,north,SPACING,SPACING)
    Z = np.full((ny,nx),np.nan,dtype=np.float32)
    with rasterio.open(DEFAULT_DEM) as src:
        reproject(source=rasterio.band(src,1),destination=Z,
                  src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
                  dst_transform=transform,dst_crs="EPSG:32624",dst_nodata=np.nan,
                  resampling=Resampling.bilinear,num_threads=2)
        to_source = Transformer.from_crs(4326,src.crs,always_xy=True)
        px,py = to_source.transform([float(r["longitude_deg_east"]) for r in stations],
                                    [float(r["latitude_deg_north"]) for r in stations])
        station_z = sample_dem(src,px,py)
        source_crs = str(src.crs)
    valid = np.isfinite(Z)
    if valid.mean() < .95:
        raise ValueError(f"Only {valid.mean():.1%} DEM coverage: inspect before plotting.")
    with rasterio.open(OUT/"feature1_gimp_surface_30m.tif","w",driver="GTiff",
                       height=ny,width=nx,count=1,dtype="float32",crs="EPSG:32624",
                       transform=transform,nodata=np.nan,compress="deflate",
                       predictor=3,tiled=True) as dst:
        dst.write(Z,1)
        dst.set_band_description(1,"Surface elevation above WGS84 ellipsoid (m)")
        dst.update_tags(DEM_PRODUCT="NASA GIMP DEM",NOMINAL_EPOCH="2007",
                        VERTICAL_REFERENCE="WGS84 ellipsoid",UNITS="m",
                        SOURCE_NATIVE_GRID="30 m EPSG:3413",RESAMPLING="bilinear")
    x = west+(np.arange(nx)+.5)*SPACING
    y = north-(np.arange(ny)+.5)*SPACING
    extent = [west/1000,east/1000,south/1000,north/1000]
    zmin,zmax = float(Z[valid].min()),float(Z[valid].max())
    norm = Normalize(zmin,zmax)
    cmap = plt.get_cmap("viridis")
    fill = np.where(valid,Z,np.nanmedian(Z))
    rgb = LightSource(azdeg=315,altdeg=40).shade(fill,cmap=cmap,norm=norm,
            blend_mode="soft",vert_exag=1,dx=SPACING,dy=SPACING)
    rgb[~valid,3] = 0
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig = plt.figure(figsize=(16,9),facecolor="#f8fafb")
    gs = fig.add_gridspec(1,2,width_ratios=[1,1.12],wspace=.18)
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(rgb,extent=extent,origin="upper")
    step = 5
    # Contours use pixel-center coordinates, matching the raster georeferencing.
    contour_levels = np.arange(np.ceil(zmin/200)*200,zmax,200)
    contours = ax.contour(x[::step]/1000,y[::step]/1000,Z[::step,::step],
                          levels=contour_levels,colors="white",linewidths=.45,alpha=.65)
    ax.clabel(contours,fmt="%d m",fontsize=8,inline=True)
    ax.scatter(cx/1000,cy/1000,s=2.8,c="#152b3c",alpha=.8,linewidths=0,zorder=4)
    order = np.argsort([float(r["distance_from_site12_km"]) for r in stations])
    ax.plot(sx[order]/1000,sy[order]/1000,color="white",lw=3,zorder=5)
    ax.plot(sx[order]/1000,sy[order]/1000,color="#e76735",lw=1.3,zorder=6)
    ax.scatter(sx/1000,sy/1000,s=27,c="#e76735",edgecolors="white",lw=.7,zorder=7)
    for i,r in enumerate(stations):
        if r["site"] in ["1","7","12 (a)"]:
            ax.annotate("Site "+r["site"],(sx[i]/1000,sy[i]/1000),xytext=(0,11),
                        textcoords="offset points",ha="center",fontsize=9,color="#172e3e",
                        bbox=dict(boxstyle="round,pad=.18",fc="white",ec="none",alpha=.9),zorder=8)
    ax.legend(handles=[Line2D([],[],marker=".",ls="none",color="#152b3c",label="Chu Feature I observations"),
                       Line2D([],[],marker="o",color="#e76735",markeredgecolor="white",label="2016 seismic transect")],
              loc="lower left",fontsize=9,framealpha=.93)
    ax.set(xlabel="UTM zone 24N easting (km)",ylabel="UTM zone 24N northing (km)",
           title="Surface elevation and survey locations",xlim=extent[:2],ylim=extent[2:])
    ax.set_aspect("equal")
    bx,by = west/1000+3,south/1000+7.5
    ax.plot([bx,bx+10],[by,by],color="white",lw=4,zorder=8)
    ax.plot([bx,bx+10],[by,by],color="#172e3e",lw=2,zorder=9)
    ax.text(bx+5,by+.6,"10 km",ha="center",fontsize=9,color="white",zorder=10,
            bbox=dict(fc="#172e3e",ec="none",alpha=.75,pad=1.5))
    ax.annotate("N",xy=(.94,.96),xytext=(.94,.87),xycoords="axes fraction",
                ha="center",va="bottom",fontsize=12,color="white",
                arrowprops=dict(arrowstyle="-|>",lw=1.8,color="white"))
    decimation = 8
    xd,yd,zd = x[::decimation],y[::decimation],Z[::decimation,::decimation]
    XX,YY = np.meshgrid(xd,yd)
    ax3 = fig.add_subplot(gs[0,1],projection="3d",facecolor="#f8fafb")
    ax3.plot_surface(XX/1000,YY/1000,zd,cmap=cmap,norm=norm,
                     rcount=len(yd),ccount=len(xd),linewidth=0,antialiased=True,rasterized=True)
    ax3.plot(sx[order]/1000,sy[order]/1000,station_z[order]+3,color="#e76735",lw=2,zorder=10)
    ax3.set(xlabel="Easting (km)",ylabel="Northing (km)",zlabel="Elevation (m)",
            title="Regional surface in 3D · vertical scale 15×")
    ax3.set_box_aspect((east-west,north-south,EXAGGERATION*(zmax-zmin)))
    ax3.view_init(elev=30,azim=-125)
    ax3.tick_params(labelsize=9,pad=1)
    for axis in [ax3.xaxis,ax3.yaxis,ax3.zaxis]:
        axis.set_major_locator(MaxNLocator(4))
        axis.get_major_formatter().set_useOffset(False)
    cax = fig.add_axes([.76,.13,.18,.023])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
                 orientation="horizontal",label="Elevation (m, WGS84 ellipsoid)")
    fig.suptitle("GIMP surface elevation · full Feature I region",x=.065,y=.97,ha="left",
                 fontsize=23,fontweight="bold",color="#17384b")
    fig.text(.065,.919,f"Full observation extent + 5 km buffer · {(east-west)/1000:.1f} × {(north-south)/1000:.1f} km · GIMP 30 m DEM",
             fontsize=12,color="#526773")
    fig.text(.065,.045,"Nominal DEM epoch: 2007. Elevations reference the WGS84 ellipsoid; no geoid or epoch correction applied.\n"
             "Colored terrain is the ice/firn surface. Observation tracks do not define the aquifer boundary. 3D rendering sampled at 240 m.",
             fontsize=10,color="#526773",linespacing=1.6)
    fig.subplots_adjust(left=.065,right=.95,bottom=.16,top=.87)
    fig.savefig(OUT/"feature1_gimp_surface_overview.png",dpi=180)
    plt.close(fig)
    # The browser preview is fully self-contained and starts with a declared scale.
    fig3 = go.Figure(go.Surface(x=xd/1000,y=yd/1000,z=zd,colorscale="Viridis",
            cmin=zmin,cmax=zmax,colorbar=dict(title="Elevation (m)"),
            hovertemplate="Easting %{x:.2f} km<br>Northing %{y:.2f} km<br>Elevation %{z:.1f} m<extra>GIMP surface</extra>"))
    fig3.add_trace(go.Scatter3d(x=sx[order]/1000,y=sy[order]/1000,z=station_z[order]+3,
            mode="lines+markers",marker=dict(size=4,color="#e76735"),line=dict(color="#e76735",width=5),
            text=[stations[i]["site"] for i in order],name="2016 seismic stations",
            hovertemplate="Site %{text}<extra>Position on GIMP surface</extra>"))
    spans = np.array([east-west,north-south,zmax-zmin])
    ratios = dict(x=spans[0]/spans[:2].max(),y=spans[1]/spans[:2].max(),
                  z=EXAGGERATION*spans[2]/spans[:2].max())
    true = {**ratios,"z":ratios["z"]/EXAGGERATION}
    fig3.update_layout(title="GIMP surface elevation | Feature I region<br><sup>Nominal 2007 · WGS84 ellipsoid heights · rendering sampled at 240 m</sup>",
        scene=dict(xaxis_title="UTM easting (km)",yaxis_title="UTM northing (km)",zaxis_title="Elevation (m)",
                   aspectmode="manual",aspectratio=ratios,camera=dict(eye=dict(x=-1.5,y=-1.7,z=1.2))),
        updatemenus=[dict(type="buttons",direction="right",x=.03,y=.03,buttons=[
            dict(label="15× vertical",method="relayout",args=[{"scene.aspectratio":ratios}]),
            dict(label="True scale",method="relayout",args=[{"scene.aspectratio":true}])])],
        paper_bgcolor="#f8fafb",font=dict(family="Arial",color="#17384b"),
        margin=dict(l=0,r=0,t=90,b=30),legend=dict(x=.02,y=.87))
    fig3.write_html(OUT/"feature1_gimp_surface_3d.html",include_plotlyjs=True,full_html=True)
    corners_lon,corners_lat = inverse.transform([west,east,east,west],[south,south,north,north])
    metadata = dict(region="Full Chu Dataset S1 Feature I observation extent and 2016 seismic stations, plus 5 km buffer",
        source=str(DEFAULT_DEM.resolve()),source_crs=source_crs,output_crs="EPSG:32624",
        vertical_reference="Metres above WGS84 ellipsoid; no geoid correction",nominal_dem_epoch=2007,
        documentation="https://nsidc.org/data/nsidc-0645/versions/1",
        bounds_utm_m=dict(west=west,east=east,south=south,north=north),
        corner_longitude_deg=list(corners_lon),corner_latitude_deg=list(corners_lat),
        raster_shape_y_x=[ny,nx],grid_spacing_m=SPACING,source_posting_m=30,
        resampling="bilinear",valid_fraction=float(valid.mean()),
        surface_elevation_min_m=zmin,surface_elevation_max_m=zmax,
        extent_width_km=(east-west)/1000,extent_height_km=(north-south)/1000,
        observation_count=len(chu),seismic_station_count=len(stations),
        plot_3d_sampling_m=decimation*SPACING,default_vertical_exaggeration=EXAGGERATION,
        station_marker_visual_lift_m=3,notes=[
          "This is surface terrain, not aquifer top or base.",
          "Survey markers are geographic references from later years, not contemporaneous 2007 observations.",
          "The raster retains metre coordinates and true vertical scale; exaggeration applies only to display.",
          "A 30 m grid is a posting/resampling scale, not a claim of 30 m vertical accuracy."])
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (OUT/"README.md").write_text(f"""# GIMP surface elevation: full Feature I region

This map covers the full Feature I observation extent and the Montgomery 2016
seismic stations, with a 5 km buffer. It covers {metadata['extent_width_km']:.2f} ×
{metadata['extent_height_km']:.2f} km, including unsurveyed terrain between tracks.

- `feature1_gimp_surface_overview.png`: elevation map and 3D surface overview.
- `feature1_gimp_surface_3d.html`: self-contained interactive 3D surface; true-scale button.
- `feature1_gimp_surface_30m.tif`: elevation raster in EPSG:32624, 30 m grid,
  WGS84 ellipsoid heights in metres. Bilinearly reprojected from the local
  NASA GIMP EPSG:3413 mosaic. No geoid or epoch correction is applied.
- `metadata.json`: exact bounds, sources, resolution and rendering details.

Nominal DEM epoch is 2007. Finite elevation coverage is {valid.mean():.2%}.
Surface elevations span {zmin:.1f}–{zmax:.1f} m above the WGS84 ellipsoid.
The DEM describes surface terrain, not the aquifer top or base. Radar and seismic
locations are overlaid for reference and do not establish an aquifer boundary.
The 3D preview uses 240 m sampling and 15× vertical exaggeration initially.
Source: https://nsidc.org/data/nsidc-0645/versions/1

Reproduce from HydroFirn with the scientific Python environment:
`python src/plot_gimp_feature1_region.py`
""")
    # Verify the saved geospatial artifact, including missing-data handling.
    with rasterio.open(OUT/"feature1_gimp_surface_30m.tif") as check:
        recovered = check.read(1)
        assert check.crs.to_epsg() == 32624
        assert check.transform == transform
        assert np.array_equal(recovered,Z,equal_nan=True)
        assert check.tags()["VERTICAL_REFERENCE"] == "WGS84 ellipsoid"
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
