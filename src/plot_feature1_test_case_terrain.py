"""Show GIMP elevation and signed x/y slopes over the exact test-case view."""

from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helheim_geometry_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/helheim_geometry_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, bounds as window_bounds, from_bounds


ROOT = Path(__file__).resolve().parents[1]
REGION = ROOT / "outputs/firn_aquifer/gimp_region"
OUT = ROOT / "outputs/firn_aquifer/feature1_test_case"


def summary(values):
    return dict(min=float(np.nanmin(values)), max=float(np.nanmax(values)),
                median=float(np.nanmedian(values)))


def main():
    case = json.loads((OUT / "metadata.json").read_text())
    view = case["bounds_utm_m"]
    dem_path = REGION / "feature1_gimp_surface_30m.tif"
    slopes_path = REGION / "feature1_gimp_xy_slopes_degrees.tif"
    slope_meta = json.loads((REGION / "feature1_gimp_xy_slopes_metadata.json").read_text())
    dem_hash = hashlib.sha256(dem_path.read_bytes()).hexdigest()
    assert dem_hash == slope_meta["source_sha256"]
    with rasterio.open(dem_path) as src:
        raw = from_bounds(view["west"], view["south"], view["east"], view["north"], src.transform)
        c0, r0 = int(np.floor(raw.col_off)), int(np.floor(raw.row_off))
        c1 = int(np.ceil(raw.col_off + raw.width))
        r1 = int(np.ceil(raw.row_off + raw.height))
        window = Window(c0, r0, c1 - c0, r1 - r0)
        z = src.read(1, window=window, masked=True).astype(float).filled(np.nan)
        raster_bounds = window_bounds(window, src.transform)
        transform, crs = src.transform, src.crs
    with rasterio.open(slopes_path) as src:
        assert src.crs == crs and src.transform == transform
        theta = src.read(window=window, masked=True).astype(float).filled(np.nan)
    assert np.isfinite(z).all() and np.isfinite(theta).all()
    for row, col in [(100, 100), (z.shape[0] // 2, z.shape[1] // 2),
                     (z.shape[0] - 101, z.shape[1] - 101)]:
        direct_x = np.degrees(np.arctan((z[row, col + 1] - z[row, col - 1]) / (2 * transform.a)))
        direct_y = np.degrees(np.arctan((z[row + 1, col] - z[row - 1, col]) / (2 * transform.e)))
        assert np.isclose(theta[0, row, col], direct_x, atol=2e-7)
        assert np.isclose(theta[1, row, col], direct_y, atol=2e-7)

    with rasterio.open(OUT / "feature1_initial_height_250m.tif") as src:
        assert src.crs == crs
        wet = (src.read(1) > 0).astype(np.uint8)
        boundaries = []
        for polygon, value in shapes(wet, mask=wet.astype(bool), transform=src.transform):
            if value == 1:
                boundaries.extend(np.asarray(ring) / 1000 for ring in polygon["coordinates"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 9.5), sharex=True, sharey=True,
                              facecolor="white")
    west, south, east, north = raster_bounds
    extent = [west / 1000, east / 1000, south / 1000, north / 1000]
    slope_limit = 3.
    fields = [z, theta[0], theta[1]]
    norms = [Normalize(1500, 1800), TwoSlopeNorm(vmin=-slope_limit, vcenter=0, vmax=slope_limit)]
    images = []
    titles = ["(a)  Surface elevation", "(b)  x slope · eastward", "(c)  y slope · northward"]
    for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        im = ax.imshow(field, extent=extent, origin="upper", interpolation="nearest",
                       cmap="cividis" if i == 0 else "RdBu_r", norm=norms[0 if i == 0 else 1])
        images.append(im)
        outlines = LineCollection(boundaries, colors="#263c48", linewidths=.65,
                                  path_effects=[pe.Stroke(linewidth=1.5, foreground="white"), pe.Normal()])
        ax.add_collection(outlines)
        ax.set(xlim=(view["west"] / 1000, view["east"] / 1000),
               ylim=(view["south"] / 1000, view["north"] / 1000),
               xlabel="UTM easting (km)")
        ax.set_title(title, loc="left", fontsize=13, pad=12, color="#17384b", fontweight="medium")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=10, colors="#465963", width=.7)
        ax.xaxis.label.set_color("#465963")
        for spine in ax.spines.values():
            spine.set_color("#9aa8b1")
            spine.set_linewidth(.7)
    axes[0].set_ylabel("UTM zone 24N northing (km)", color="#465963")
    fig.suptitle("GIMP surface elevation and slopes", x=.07, y=.975, ha="left",
                 fontsize=24, fontweight="bold", color="#17384b")
    fig.text(.07, .921, "Same zoomed test-case domain · 30 m GIMP data · initial aquifer footprint outlined",
             fontsize=12, color="#526773")
    fig.subplots_adjust(left=.07, right=.98, bottom=.235, top=.85, wspace=.08)
    # Align each horizontal scale with the actual map positions after equal aspect.
    fig.canvas.draw()
    box0, box1, box2 = [ax.get_position() for ax in axes]
    elevation_ax = fig.add_axes([box0.x0, .14, box0.width, .018])
    slope_ax = fig.add_axes([box1.x0, .14, box2.x1 - box1.x0, .018])
    elevation_cb = fig.colorbar(images[0], cax=elevation_ax, orientation="horizontal",
                                ticks=[1500, 1600, 1700, 1800])
    slope_cb = fig.colorbar(images[1], cax=slope_ax, orientation="horizontal", extend="min",
                            ticks=np.arange(-3, 4, 1), extendfrac=.025)
    for cb in [elevation_cb, slope_cb]:
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=3, width=.7, colors="#465963", labelsize=10, pad=5)
    elevation_ax.set_title("Elevation (m, WGS84 ellipsoid)", loc="left", fontsize=11,
                           color="#17384b", pad=10)
    slope_ax.set_title("Signed directional slope (°)", loc="left", fontsize=11,
                       color="#17384b", pad=10)
    fig.legend(handles=[Line2D([], [], color="#263c48", lw=.8, label="Initial aquifer footprint")],
               loc="upper left", bbox_to_anchor=(.065, .086), frameon=False, fontsize=10)
    fig.text(.07, .018,
             "Positive slope = elevation rises eastward (x) or northward (y). "
             "Angles = arctan(dz/dx), arctan(dz/dy), in degrees.\n"
             "Centred differences on the unsmoothed 30 m DEM · GIMP nominal epoch 2007 · "
             "slope colours saturate below −3°; values are unchanged.",
             fontsize=10, color="#526773", linespacing=1.6)
    image = OUT / "feature1_elevation_xy_slopes.png"
    fig.savefig(image, dpi=220)
    plt.close(fig)
    metadata = dict(source_elevation=str(dem_path.resolve()), source_slopes=str(slopes_path.resolve()),
        source_elevation_sha256=dem_hash, output=str(image.resolve()), crs=str(crs),
        view_bounds_utm_m=view, native_crop_bounds_utm_m=dict(zip(["west", "south", "east", "north"], raster_bounds)),
        grid_spacing_m=30, crop_shape_yx=list(z.shape),
        display_note="Native 30 m cells cover the view; axes clip to the exact test-case extent. No resampling.",
        elevation_reference="WGS84 ellipsoid", nominal_dem_epoch=2007,
        slope_formulas={"x": "degrees(arctan(dz/dx))", "y": "degrees(arctan(dz/dy))"},
        positive_directions={"x": "UTM grid east", "y": "UTM grid north"},
        slope_method=slope_meta["method"], elevation_m=summary(z),
        x_slope_degrees=summary(theta[0]), y_slope_degrees=summary(theta[1]),
        elevation_color_limits_m=[1500, 1800], slope_color_limits_degrees=[-3, 3],
        x_fraction_color_saturated=float((np.abs(theta[0]) > slope_limit).mean()),
        y_fraction_color_saturated=float((np.abs(theta[1]) > slope_limit).mean()),
        input_data_changed=False, validation={"slope_source_dem_hash_matches": True,
            "direct_central_difference_checks": True, "all_view_pixels_finite": True,
            "exact_test_case_view_bounds": True})
    (OUT / "feature1_elevation_xy_slopes_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ["output", "elevation_m", "x_slope_degrees",
        "y_slope_degrees", "validation"]}, indent=2))


if __name__ == "__main__":
    main()
