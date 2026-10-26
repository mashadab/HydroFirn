"""Calculate and plot signed east/north GIMP surface-slope angles."""

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from pyproj import Transformer
import rasterio


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/firn_aquifer/gimp_region"
SOURCE = OUT / "feature1_gimp_surface_30m.tif"


def directional_slopes(z, transform):
    """Central differences in physical metre coordinates, then arctangent.

    For a north-up raster, row spacing transform.e is NEGATIVE. Passing that
    signed spacing is necessary to get slope toward increasing northing.
    The outer pixel rim is not evaluated with one-sided derivatives.
    """
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise ValueError("Require an unrotated, north-up projected raster.")
    z = np.asarray(z, dtype=np.float64)
    dz_dy, dz_dx = np.gradient(z, transform.e, transform.a, edge_order=2)
    finite = np.isfinite(z)
    good = np.zeros(z.shape, dtype=bool)
    good[1:-1,1:-1] = (finite[1:-1,1:-1] & finite[1:-1,:-2] & finite[1:-1,2:]
                       & finite[:-2,1:-1] & finite[2:,1:-1])
    theta_x = np.where(good, np.degrees(np.arctan(dz_dx)), np.nan)
    theta_y = np.where(good, np.degrees(np.arctan(dz_dy)), np.nan)
    return theta_x, theta_y, good


def stats(values):
    v = values[np.isfinite(values)]
    return dict(min=float(v.min()),max=float(v.max()),mean=float(v.mean()),
                median=float(np.median(v)),p05=float(np.percentile(v,5)),
                p95=float(np.percentile(v,95)))


def main():
    with rasterio.open(SOURCE) as src:
        z = src.read(1,masked=True).astype(np.float64).filled(np.nan)
        transform,crs,profile,bounds = src.transform,src.crs,src.profile.copy(),src.bounds
    if crs.to_epsg() != 32624:
        raise ValueError("Expected the existing EPSG:32624 surface raster.")
    theta_x,theta_y,good = directional_slopes(z,transform)

    # Analytical inclined plane checks both units and the raster northing sign.
    x = np.arange(9)*transform.a
    y = np.arange(8)*transform.e
    plane = 100+np.tan(np.radians(-3))*x[None,:]+np.tan(np.radians(2))*y[:,None]
    test_x,test_y,test_good = directional_slopes(plane,transform)
    assert np.allclose(test_x[test_good],-3,atol=1.e-10)
    assert np.allclose(test_y[test_good],2,atol=1.e-10)
    # Independent three-point checks on the actual DEM, including sign.
    for j,i in [(200,200),(z.shape[0]//2,z.shape[1]//2),(z.shape[0]-201,z.shape[1]-201)]:
        direct_x = np.degrees(np.arctan((z[j,i+1]-z[j,i-1])/(2*transform.a)))
        direct_y = np.degrees(np.arctan((z[j+1,i]-z[j-1,i])/(2*transform.e)))
        assert np.isclose(theta_x[j,i],direct_x,atol=1.e-12)
        assert np.isclose(theta_y[j,i],direct_y,atol=1.e-12)

    # Display limits alone saturate a small tail; exported values are unclipped.
    combined = np.abs(np.concatenate([theta_x[good],theta_y[good]]))
    limit = float(max(1,np.ceil(np.percentile(combined,99.5))))
    norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    OUT.mkdir(parents=True,exist_ok=True)
    target = OUT / "feature1_gimp_xy_slopes_degrees.tif"
    profile.update(count=2,dtype="float32",nodata=np.nan,compress="deflate",predictor=3)
    with rasterio.open(target,"w",**profile) as dst:
        dst.write(theta_x.astype(np.float32),1)
        dst.write(theta_y.astype(np.float32),2)
        dst.set_band_description(1,"Signed x/easting slope angle (degrees)")
        dst.set_band_description(2,"Signed y/northing slope angle (degrees)")
        dst.update_tags(UNITS="degrees",X_POSITIVE="east",Y_POSITIVE="north",
                        ANGLE_SIGN="positive when elevation rises toward increasing coordinate",
                        X_FORMULA="atan(dz/dx) * 180/pi",Y_FORMULA="atan(dz/dy) * 180/pi",
                        METHOD="central differences on 30 m raster, no smoothing; outer pixel rim nodata",
                        VERTICAL_REFERENCE="source surface heights above WGS84 ellipsoid",
                        NOMINAL_DEM_EPOCH="2007",SOURCE=str(SOURCE.resolve()))

    with (ROOT / "outputs/firn_aquifer/montgomery2017_table1.csv").open(newline="") as f:
        stations = [r for r in csv.DictReader(f) if r["year"] == "2016"]
    order = np.argsort([float(r["distance_from_site12_km"]) for r in stations])
    forward = Transformer.from_crs(4326,crs,always_xy=True)
    sx,sy = forward.transform([float(r["longitude_deg_east"]) for r in stations],
                              [float(r["latitude_deg_north"]) for r in stations])
    sx,sy = np.asarray(sx)/1000,np.asarray(sy)/1000
    extent = [bounds.left/1000,bounds.right/1000,bounds.bottom/1000,bounds.top/1000]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(13,9),sharex=True,sharey=True,facecolor="#f8fafb")
    for ax,field,title in zip(axes,[theta_x,theta_y],
        [r"(a)  x slope: $\theta_x=\tan^{-1}(\partial z/\partial x)$",
         r"(b)  y slope: $\theta_y=\tan^{-1}(\partial z/\partial y)$"]):
        im = ax.imshow(field,extent=extent,origin="upper",cmap="RdBu_r",norm=norm,
                       interpolation="nearest",rasterized=True)
        ax.plot(sx[order],sy[order],color="white",lw=3,zorder=4)
        ax.plot(sx[order],sy[order],color="#172e3e",lw=1.1,zorder=5)
        ax.scatter(sx,sy,s=16,c="#172e3e",edgecolors="white",lw=.6,zorder=6)
        for i,r in enumerate(stations):
            if r["site"] in ["1","12 (a)"]:
                ax.annotate("Site "+r["site"],(sx[i],sy[i]),xytext=(0,10),
                            textcoords="offset points",ha="center",fontsize=9,
                            bbox=dict(boxstyle="round,pad=.15",fc="white",ec="none",alpha=.9),zorder=7)
        ax.set(xlabel="UTM zone 24N easting (km)",title=title,
               xlim=extent[:2],ylim=extent[2:])
        ax.set_aspect("equal")
    axes[0].set_ylabel("UTM zone 24N northing (km)")
    axes[0].legend(handles=[Line2D([],[],marker="o",color="#172e3e",markeredgecolor="white",
                                  markersize=5,lw=1,label="2016 seismic stations")],
                   loc="lower left",fontsize=9,framealpha=.95)
    cax = fig.add_axes([.27,.125,.48,.023])
    fig.colorbar(im,cax=cax,orientation="horizontal",extend="both",
                 ticks=np.arange(-limit,limit+.5,1),label="Signed directional slope (degrees)")
    fig.suptitle("GIMP surface slopes · full Feature I region",x=.075,y=.97,ha="left",
                 fontsize=22,fontweight="bold",color="#17384b")
    fig.text(.075,.921,"x increases eastward; y increases northward · 30 m grid · no smoothing",
             fontsize=12,color="#526773")
    fig.text(.075,.022,
             f"Positive = elevation rises toward east (x) or north (y). Shared color range ±{limit:g}°; larger values saturate.\n"
             "Nominal GIMP epoch: 2007 · derivatives of WGS84 ellipsoid heights · outer 30 m pixel rim omitted.",
             fontsize=10,color="#526773",linespacing=1.6)
    fig.subplots_adjust(left=.075,right=.97,bottom=.22,top=.86,wspace=.10)
    fig.savefig(OUT/"feature1_gimp_xy_slopes.png",dpi=200)
    plt.close(fig)
    metadata = dict(source=str(SOURCE.resolve()),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        output=str(target.resolve()),crs=str(crs),grid_spacing_m=[transform.a,abs(transform.e)],
        formulas={"x":"degrees(atan(dz/dx))","y":"degrees(atan(dz/dy))"},
        coordinate_directions={"x":"UTM grid east","y":"UTM grid north"},
        angle_sign="positive when elevation rises toward increasing coordinate",
        method="Second-order centered finite differences over 60 m; no smoothing; one-pixel exterior rim left nodata",
        input_elevation_reference="WGS84 ellipsoid",nominal_dem_epoch=2007,
        full_extent_utm_m=list(bounds),valid_pixel_count=int(good.sum()),
        excluded_pixel_count=int(good.size-good.sum()),
        x_slope_degrees=stats(theta_x),y_slope_degrees=stats(theta_y),
        display_symmetric_limit_degrees=limit,
        x_fraction_color_saturated=float((np.abs(theta_x[good])>limit).mean()),
        y_fraction_color_saturated=float((np.abs(theta_y[good])>limit).mean()),
        export_values_clipped=False,
        validation={"known_minus3_degree_east_plus2_degree_north_plane":"passed",
                    "independent_actual_DEM_central_differences":"passed",
                    "outer_rim_is_nodata":"passed"})
    with rasterio.open(target) as check:
        assert check.transform == transform and check.crs == crs and check.count == 2
        assert np.array_equal(check.read(1),theta_x.astype(np.float32),equal_nan=True)
        assert np.array_equal(check.read(2),theta_y.astype(np.float32),equal_nan=True)
        assert np.isnan(check.read(1)[0]).all() and np.isnan(check.read(2)[:,-1]).all()
    metadata["validation"]["saved_two_band_raster_values_and_georeferencing"] = "passed"
    (OUT/"feature1_gimp_xy_slopes_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
