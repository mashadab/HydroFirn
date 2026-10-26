"""Show radar-derived aquifer thickness and the existing mesh over the GIMP ROI.

Aquifer height means saturated thickness h = water-table elevation - base
elevation, in metres. This script does not change or extend the reconstruction.
"""

from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helheim_geometry_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/helheim_geometry_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/firn_aquifer/gimp_region"
RECON = ROOT/"outputs/firn_aquifer/feature1_3d"


def main():
    source_grid = RECON/"feature1_geometry.npz"
    source_points = RECON/"chu_source_points.csv"
    source_dem = OUT/"feature1_gimp_surface_30m.tif"
    fields = np.load(source_grid,allow_pickle=False)
    meta = json.loads((RECON/"metadata.json").read_text())
    x,y = fields["x_m"],fields["y_m"]
    origin = fields["utm_origin_m"]
    active = fields["active_cells"]
    h = fields["aquifer_thickness_m"]
    top,base = fields["water_table_z_m"],fields["base_z_m"]
    supported = fields["supported_nodes"]
    assert np.allclose(h[supported],(top-base)[supported],atol=1.e-10,rtol=0)
    # Area mean of the two planar triangles used in each existing mesh quad.
    cell_h = (2*h[:-1,:-1]+h[:-1,1:]+2*h[1:,1:]+h[1:,:-1])/6
    cell_h = np.where(active,cell_h,np.nan)
    spacing = float(x[1]-x[0])
    assert np.allclose(np.diff(x),spacing) and np.allclose(np.diff(y),spacing)
    area = active.sum()*spacing**2/1.e6
    volume = float(np.nansum(cell_h)*spacing**2)
    assert np.isclose(area,meta["summary"]["mesh_area_km2"])
    assert np.isclose(volume,meta["summary"]["mesh_volume_km3"]*1.e9,rtol=1.e-10)

    with source_points.open(newline="") as f:
        records = list(csv.DictReader(f))
    lon = np.array([float(r["longitude_deg"]) for r in records])
    lat = np.array([float(r["latitude_deg"]) for r in records])
    point_h = np.array([float(r["thickness_m"]) for r in records])
    to_utm = Transformer.from_crs(4326,32624,always_xy=True)
    px,py = to_utm.transform(lon,lat)
    with (ROOT/"outputs/firn_aquifer/montgomery2017_table1.csv").open(newline="") as f:
        stations = [r for r in csv.DictReader(f) if r["year"] == "2016"]
    sx,sy = to_utm.transform([float(r["longitude_deg_east"]) for r in stations],
                            [float(r["latitude_deg_north"]) for r in stations])
    sx,sy = np.asarray(sx)/1000,np.asarray(sy)/1000
    order = np.argsort([float(r["distance_from_site12_km"]) for r in stations])
    with rasterio.open(source_dem) as src:
        background = src.read(1,masked=True)
        bounds,crs = src.bounds,src.crs
        bg_x = src.transform.c+(np.arange(src.width)+.5)*src.transform.a
        bg_y = src.transform.f+(np.arange(src.height)+.5)*src.transform.e
    extent = [bounds.left/1000,bounds.right/1000,bounds.bottom/1000,bounds.top/1000]
    h_extent = [(origin[0]+x[0])/1000,(origin[0]+x[-1])/1000,
                (origin[1]+y[0])/1000,(origin[1]+y[-1])/1000]
    # Keep the true 250 m mesh-cell resolution; no apparent 30 m thickness detail.
    out_transform = from_origin(origin[0]+x[0],origin[1]+y[-1],spacing,spacing)
    raster_values = np.flipud(cell_h).astype(np.float32)
    raster_path = OUT/"feature1_aquifer_height_250m.tif"
    with rasterio.open(raster_path,"w",driver="GTiff",height=cell_h.shape[0],width=cell_h.shape[1],
                       count=1,dtype="float32",crs=crs,transform=out_transform,
                       nodata=np.nan,compress="deflate",predictor=3,tiled=True) as dst:
        dst.write(raster_values,1)
        dst.set_band_description(1,"Aquifer height: saturated thickness, mesh-cell area mean (m)")
        dst.update_tags(UNITS="m",QUANTITY="water-table elevation minus aquifer-base elevation",
                        PERIOD="2012-2014 composite",GRID_SPACING="250 m; not observational resolution",
                        UNSUPPORTED="nodata; not physical absence",SOURCE=str(source_grid.resolve()),
                        CELL_MEAN="Area mean of two piecewise-planar triangles of the existing mesh quad")

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(13,9),sharex=True,sharey=True,facecolor="#f8fafb")
    limit = float(np.ceil(max(point_h.max(),np.nanmax(cell_h))/5)*5)
    norm = Normalize(0,limit)
    for ax in axes:
        ax.imshow(background,extent=extent,origin="upper",cmap="Greys_r",alpha=.23,
                  interpolation="bilinear")
        contours = ax.contour(bg_x[::8]/1000,bg_y[::8]/1000,background[::8,::8],
                              levels=np.arange(1400,2000,200),colors="#71828a",linewidths=.45,alpha=.4)
        ax.clabel(contours,fmt="%d m",fontsize=8)
        ax.plot(sx[order],sy[order],color="white",lw=3,zorder=6)
        ax.plot(sx[order],sy[order],color="#172e3e",lw=1.1,zorder=7)
        ax.scatter(sx,sy,s=16,c="#172e3e",edgecolors="white",lw=.6,zorder=8)
        for i,r in enumerate(stations):
            if r["site"] in ["1","12 (a)"]:
                ax.annotate("Site "+r["site"],(sx[i],sy[i]),xytext=(0,10),
                            textcoords="offset points",ha="center",fontsize=9,
                            bbox=dict(boxstyle="round,pad=.15",fc="white",ec="none",alpha=.9),zorder=9)
        ax.set(xlabel="UTM zone 24N easting (km)",xlim=extent[:2],ylim=extent[2:])
        ax.set_aspect("equal")
    axes[0].scatter(px/1000,py/1000,c=point_h,s=7,cmap="viridis",norm=norm,
                    linewidths=0,zorder=5,rasterized=True)
    axes[0].set_title("(a) Radar-derived thickness along survey tracks")
    im = axes[1].imshow(cell_h,extent=h_extent,origin="lower",cmap="viridis",norm=norm,
                        interpolation="nearest",zorder=4,rasterized=True)
    axes[1].set_title(f"(b) Current reconstruction · {area:.2f} km²")
    axes[0].set_ylabel("UTM zone 24N northing (km)")
    axes[0].legend(handles=[Line2D([],[],marker="o",color="#172e3e",markeredgecolor="white",
                                  markersize=5,lw=1,label="2016 seismic reference")],
                   loc="lower left",fontsize=9,framealpha=.95)
    cax = fig.add_axes([.27,.125,.48,.023])
    fig.colorbar(im,cax=cax,orientation="horizontal",ticks=np.arange(0,limit+1,5),
                 label="Aquifer height / saturated thickness (m)")
    fig.suptitle("Aquifer height · same Feature I region",x=.075,y=.97,ha="left",
                 fontsize=22,fontweight="bold",color="#17384b")
    fig.text(.075,.921,r"$h=z_{\mathrm{water\ table}}-z_{\mathrm{base}}$ · 2012–2014 composite · same bounds as the GIMP slope panels",
             fontsize=12,color="#526773")
    fig.text(.075,.022,
             "Left: Chu radar-attenuation thickness estimates. Right: existing 250 m mesh; both thickness and depth must pass support filters.\n"
             "Pale background = GIMP terrain context; uncolored areas are unspecified. The current mesh does not cover the full aquifer.",
             fontsize=9.5,color="#526773",linespacing=1.6)
    fig.subplots_adjust(left=.075,right=.97,bottom=.22,top=.86,wspace=.10)
    fig.savefig(OUT/"feature1_aquifer_height.png",dpi=200)
    plt.close(fig)
    # Verify exported values, row orientation and volume independently of the plot.
    with rasterio.open(raster_path) as check:
        recovered = check.read(1)
        assert np.array_equal(recovered,raster_values,equal_nan=True)
        assert np.isclose(np.nansum(recovered.astype(float))*spacing**2,volume,rtol=1.e-7)
        assert np.isclose(check.bounds.left,origin[0]+x[0])
        assert np.isclose(check.bounds.top,origin[1]+y[-1])
        assert check.crs.to_epsg() == 32624
    summary = dict(quantity="Saturated thickness h, not absolute elevation or depth below the surface",
        equations="h = z_water_table - z_base",period="2012-2014 composite",
        source_grid=str(source_grid.resolve()),source_grid_sha256=hashlib.sha256(source_grid.read_bytes()).hexdigest(),
        source_points=str(source_points.resolve()),source_paper="https://doi.org/10.1029/2018GL079751",
        map_extent_utm_m=list(bounds),map_crs=str(crs),same_extent_as="feature1_gimp_xy_slopes.png",
        source_point_count=len(point_h),source_point_thickness_range_m=[float(point_h.min()),float(point_h.max())],
        reconstructed_area_km2=float(area),grid_spacing_m=spacing,
        reconstructed_cell_mean_thickness_range_m=[float(np.nanmin(cell_h)),float(np.nanmax(cell_h))],
        reconstructed_bulk_material_volume_m3=volume,
        existing_reconstruction_changed=False,missing_values="Unspecified, not zero thickness or physical absence",
        raster_cell_values="Area means of the two existing piecewise-planar mesh triangles",
        display_color_limits_m=[0,limit],
        validation={"h_equals_top_minus_base":"passed","area_and_volume_match_existing_mesh":"passed",
                    "saved_values_georeferencing_and_north_up_orientation":"passed"})
    (OUT/"feature1_aquifer_height_metadata.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
