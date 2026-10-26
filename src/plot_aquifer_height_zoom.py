"""Standalone zoom of aquifer-height panel B, preserving the existing field."""

from pathlib import Path
import argparse
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
from rasterio.windows import from_bounds, bounds as window_bounds


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/firn_aquifer/gimp_region"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding-km",type=float,default=2.)
    args = parser.parse_args()
    if not np.isfinite(args.padding_km) or args.padding_km <= 0:
        raise ValueError("Padding must be positive and finite.")
    source = OUT/"feature1_aquifer_height_250m.tif"
    source_meta = json.loads((OUT/"feature1_aquifer_height_metadata.json").read_text())
    with rasterio.open(source) as src:
        h = src.read(1,masked=True).astype(float).filled(np.nan)
        transform,crs,h_bounds = src.transform,src.crs,src.bounds
    rows,cols = np.nonzero(np.isfinite(h))
    left = transform.c+cols.min()*transform.a
    right = transform.c+(cols.max()+1)*transform.a
    north = transform.f+rows.min()*transform.e
    south = transform.f+(rows.max()+1)*transform.e
    pad = args.padding_km*1000
    view = dict(west=left-pad,east=right+pad,south=south-pad,north=north+pad)
    with rasterio.open(OUT/"feature1_gimp_surface_30m.tif") as dem:
        view = dict(west=max(view["west"],dem.bounds.left),east=min(view["east"],dem.bounds.right),
                    south=max(view["south"],dem.bounds.bottom),north=min(view["north"],dem.bounds.top))
        window = from_bounds(view["west"],view["south"],view["east"],view["north"],dem.transform)
        window = window.round_offsets().round_lengths()
        z = dem.read(1,window=window,masked=True)
        z_transform = dem.window_transform(window)
        zw,zs,ze,zn = window_bounds(window,dem.transform)
    # Every finite cell from the prior panel remains visible in the new view.
    assert view["west"] <= left and view["east"] >= right
    assert view["south"] <= south and view["north"] >= north
    xx = z_transform.c+(np.arange(z.shape[1])+.5)*z_transform.a
    yy = z_transform.f+(np.arange(z.shape[0])+.5)*z_transform.e
    with (ROOT/"outputs/firn_aquifer/montgomery2017_table1.csv").open(newline="") as f:
        stations = [r for r in csv.DictReader(f) if r["year"] == "2016"]
    forward = Transformer.from_crs(4326,crs,always_xy=True)
    sx,sy = forward.transform([float(r["longitude_deg_east"]) for r in stations],
                              [float(r["latitude_deg_north"]) for r in stations])
    sx,sy = np.asarray(sx),np.asarray(sy)
    order = np.argsort([float(r["distance_from_site12_km"]) for r in stations])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,ax = plt.subplots(figsize=(9,10),facecolor="#f8fafb")
    ax.imshow(z,extent=[zw/1000,ze/1000,zs/1000,zn/1000],origin="upper",
              cmap="Greys_r",alpha=.23,interpolation="bilinear")
    contours = ax.contour(xx[::4]/1000,yy[::4]/1000,z[::4,::4],
              levels=np.arange(1400,2000,100),colors="#71828a",linewidths=.6,alpha=.45)
    ax.clabel(contours,fmt="%d m",fontsize=9)
    vmin,vmax = source_meta["display_color_limits_m"]
    im = ax.imshow(h,extent=[h_bounds.left/1000,h_bounds.right/1000,h_bounds.bottom/1000,h_bounds.top/1000],
                   origin="upper",cmap="viridis",norm=Normalize(vmin,vmax),
                   interpolation="nearest",zorder=4)
    ax.plot(sx[order]/1000,sy[order]/1000,color="white",lw=3,zorder=6)
    ax.plot(sx[order]/1000,sy[order]/1000,color="#172e3e",lw=1.1,zorder=7)
    ax.scatter(sx/1000,sy/1000,s=22,c="#172e3e",edgecolors="white",lw=.7,zorder=8)
    for i,r in enumerate(stations):
        if (r["site"] in ["1","7","12 (a)"] and view["west"] <= sx[i] <= view["east"]
                and view["south"] <= sy[i] <= view["north"]):
            ax.annotate("Site "+r["site"],(sx[i]/1000,sy[i]/1000),xytext=(0,11),
                        textcoords="offset points",ha="center",fontsize=10,
                        bbox=dict(boxstyle="round,pad=.18",fc="white",ec="none",alpha=.94),zorder=9)
    ax.set(xlim=(view["west"]/1000,view["east"]/1000),ylim=(view["south"]/1000,view["north"]/1000),
           xlabel="UTM zone 24N easting (km)",ylabel="UTM zone 24N northing (km)")
    ax.set_aspect("equal")
    ax.legend(handles=[Line2D([],[],marker="o",color="#172e3e",markeredgecolor="white",
                              markersize=5,lw=1,label="2016 seismic reference")],
              loc="lower left",fontsize=9,framealpha=.95)
    cax = fig.add_axes([.83,.23,.027,.59])
    fig.colorbar(im,cax=cax,ticks=np.arange(vmin,vmax+1,5),
                 label="Aquifer height / saturated thickness (m)")
    fig.suptitle("Aquifer height · zoom of panel B",x=.10,y=.97,ha="left",
                 fontsize=21,fontweight="bold",color="#17384b")
    fig.text(.10,.923,"2012–2014 composite · 250 m grid · 43.25 km² reconstructed footprint",
             fontsize=10.5,color="#526773")
    fig.text(.10,.045,
             f"All reconstructed patches shown with a {args.padding_km:g} km surrounding buffer. Same 0–{vmax:g} m color scale.\n"
             "Uncolored terrain is unspecified; the surrounding view leaves room to compare later expansion.",
             fontsize=9.5,color="#526773",linespacing=1.6)
    fig.subplots_adjust(left=.13,right=.79,bottom=.14,top=.88)
    target = OUT/"feature1_aquifer_height_zoom.png"
    fig.savefig(target,dpi=220)
    plt.close(fig)
    metadata = dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output=str(target.resolve()),quantity="Saturated thickness h, metres",crs=str(crs),
        view_bounds_utm_m=view,buffer_m=pad,all_prior_reconstruction_cells_visible=True,
        color_limits_m=[vmin,vmax],resampling_of_height="none; nearest-neighbor display of original cells",
        field_changed=False,simulation_performed=False)
    (OUT/"feature1_aquifer_height_zoom_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
