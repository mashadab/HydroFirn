"""Make the panel-B test case, retaining patch heights and setting all else to zero.

The source reconstruction is unchanged. Zero outside its supported cells is an
initial-condition assumption for this test case, not an observation of dry firn.
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
from matplotlib.patches import Patch
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/firn_aquifer/gimp_region"
OUT = ROOT / "outputs/firn_aquifer/feature1_test_case"
DISPLAY_COLOR_LIMITS_M = [0., 25.]
DISPLAY_COLORMAP = "YlGnBu"
DRY_COLOR = "#eef2f5"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_preview(h0, crs, view, color_limits, wet_area_km2):
    """Show positive heights on a continuous scale, with a separate zero class."""
    with rasterio.open(INPUT / "feature1_gimp_surface_30m.tif") as dem:
        window = from_bounds(view["west"], view["south"], view["east"],
                             view["north"], dem.transform)
        window = window.round_offsets().round_lengths()
        z = dem.read(1, window=window, masked=True)
        z_transform = dem.window_transform(window)
    xx = z_transform.c + (np.arange(z.shape[1]) + .5) * z_transform.a
    yy = z_transform.f + (np.arange(z.shape[0]) + .5) * z_transform.e
    with (ROOT / "outputs/firn_aquifer/montgomery2017_table1.csv").open(newline="") as f:
        stations = [row for row in csv.DictReader(f) if row["year"] == "2016"]
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    sx, sy = forward.transform([float(row["longitude_deg_east"]) for row in stations],
                               [float(row["latitude_deg_north"]) for row in stations])
    sx, sy = np.asarray(sx), np.asarray(sy)
    order = np.argsort([float(row["distance_from_site12_km"]) for row in stations])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 11), facecolor="white")
    extent = [view["west"] / 1000, view["east"] / 1000,
              view["south"] / 1000, view["north"] / 1000]
    ax.set_facecolor(DRY_COLOR)
    # Mask zeros only for display; the actual test-case field remains unchanged.
    im = ax.imshow(np.ma.masked_equal(h0, 0), extent=extent, origin="upper", cmap=DISPLAY_COLORMAP,
                   norm=Normalize(*color_limits), interpolation="nearest")
    contours = ax.contour(xx[::4] / 1000, yy[::4] / 1000, z[::4, ::4],
                         levels=np.arange(1400, 2000, 100), colors="#7c8993",
                         linewidths=.7, alpha=.65)
    ax.clabel(contours, fmt="%d m", fontsize=9)
    ax.plot(sx[order] / 1000, sy[order] / 1000, color="white", lw=3, zorder=6)
    ax.plot(sx[order] / 1000, sy[order] / 1000, color="#172e3e", lw=1.1, zorder=7)
    ax.scatter(sx / 1000, sy / 1000, s=22, c="#172e3e", edgecolors="white", lw=.7, zorder=8)
    for i, row in enumerate(stations):
        if (row["site"] in ["1", "7", "12 (a)"] and
                view["west"] <= sx[i] <= view["east"] and
                view["south"] <= sy[i] <= view["north"]):
            ax.annotate("Site " + row["site"], (sx[i] / 1000, sy[i] / 1000),
                        xytext=(0, 11), textcoords="offset points", ha="center", fontsize=10,
                        bbox=dict(boxstyle="round,pad=.18", fc="white", ec="none", alpha=.94),
                        zorder=9)
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="UTM zone 24N easting (km)",
           ylabel="UTM zone 24N northing (km)")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#9aa8b1")
        spine.set_linewidth(.7)
    ax.tick_params(colors="#465963", labelsize=10)
    ax.xaxis.label.set_color("#465963")
    ax.yaxis.label.set_color("#465963")
    cax = fig.add_axes([.19, .15, .67, .018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=np.arange(color_limits[0], color_limits[1] + 1, 5))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=.7, colors="#465963", labelsize=10, pad=5)
    cax.set_title("Aquifer height h₀ (m)", loc="left", fontsize=11, color="#17384b", pad=10)
    fig.legend(handles=[Patch(facecolor=DRY_COLOR, edgecolor="#a6b1ba",
                              label="h₀ = 0 m · assumed dry"),
                        Line2D([], [], marker="o", color="#172e3e", markeredgecolor="white",
                               markersize=5, lw=1, label="2016 seismic reference")],
               bbox_to_anchor=(.18, .092), loc="upper left", ncol=2,
               fontsize=9, frameon=False, handlelength=1.8, columnspacing=2.)
    fig.suptitle("Firn aquifer · initial test case", x=.16, y=.98, ha="left",
                 fontsize=20, fontweight="bold", color="#17384b")
    fig.text(.16, .945,
             f"Panel B domain · 250 m grid · {h0.shape[1]} × {h0.shape[0]} cells · "
             f"{wet_area_km2:.2f} km² initial wet area", fontsize=10, color="#526773")
    fig.text(.19, .025,
             "Reconstructed patch heights retained; all other cells initialized to zero.\n"
             "Grey contours show GIMP surface elevation.",
             fontsize=9, color="#526773", linespacing=1.6)
    fig.subplots_adjust(left=.16, right=.89, bottom=.25, top=.91)
    path = OUT / "feature1_initial_height.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    source = INPUT / "feature1_aquifer_height_250m.tif"
    source_hash = sha256(source)
    zoom_path = INPUT / "feature1_aquifer_height_zoom_metadata.json"
    zoom = json.loads(zoom_path.read_text())
    view = zoom["view_bounds_utm_m"]
    with rasterio.open(source) as src:
        crs = src.crs
        dx, dy = src.res
        assert dx == dy == 250 and src.transform.b == src.transform.d == 0
        window = from_bounds(view["west"], view["south"], view["east"],
                             view["north"], src.transform)
        raw_window = np.array([window.col_off, window.row_off, window.width, window.height])
        assert np.allclose(raw_window, np.rint(raw_window), atol=1e-9, rtol=0)
        window = Window(*np.rint(raw_window).astype(int))
        # Boundless reading fills the part of the buffer outside the source grid.
        patch_h = src.read(1, window=window, boundless=True, masked=True).filled(np.nan)
        source_h = src.read(1, masked=True).filled(np.nan)
    mask = np.isfinite(patch_h)
    assert np.all(patch_h[mask] > 0)
    assert mask.sum() == np.isfinite(source_h).sum()
    # The same aligned cells are copied exactly; there is no interpolation.
    assert np.array_equal(patch_h[mask], source_h[np.isfinite(source_h)])
    h0 = np.where(mask, patch_h, np.float32(0)).astype(np.float32)
    transform = from_origin(view["west"], view["north"], dx, dy)
    assert np.all(np.isfinite(h0)) and np.all(h0 >= 0)
    assert np.all(h0[~mask] == 0) and np.array_equal(h0[mask], patch_h[mask])
    assert not np.any(h0[[0, -1], :]) and not np.any(h0[:, [0, -1]])
    area = dx * dy
    wet_area_km2 = float(mask.sum() * area / 1e6)
    bulk_volume = float(h0.sum(dtype=np.float64) * area)
    assert bulk_volume == float(np.nansum(source_h, dtype=np.float64) * area)

    OUT.mkdir(parents=True, exist_ok=True)
    tif = OUT / "feature1_initial_height_250m.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=h0.shape[0], width=h0.shape[1],
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=None, compress="deflate", predictor=3) as dst:
        dst.write(h0, 1)
        dst.set_band_description(1, "Initial saturated aquifer thickness h0 (m)")
        dst.update_tags(quantity="initial saturated thickness", units="m", role="test_case",
                        outside_patch_assumption="h0=0 m (assumed dry)",
                        source_sha256=source_hash, zero_is_valid_data="true")
    # NPZ rows increase northward, matching the existing geometry's convention.
    # Raster rows, by contrast, increase southward as usual for a north-up GeoTIFF.
    x = (np.arange(h0.shape[1]) + .5) * dx
    y = (np.arange(h0.shape[0]) + .5) * dy
    npz = OUT / "feature1_initial_height.npz"
    np.savez_compressed(npz, h0_m=np.flipud(h0), reconstructed_patch_mask=np.flipud(mask),
                        x_m=x, y_m=y, x_utm_m=view["west"] + x,
                        y_utm_m=view["south"] + y,
                        utm_origin_m=np.array([view["west"], view["south"]]),
                        dx_m=dx, dy_m=dy, epsg=crs.to_epsg())
    image = make_preview(h0, crs, view, DISPLAY_COLOR_LIMITS_M, wet_area_km2)

    with rasterio.open(tif) as check:
        assert check.nodata is None
        assert np.array_equal(check.read(1), h0)
        assert np.all(check.read_masks(1) == 255)
        assert tuple(check.bounds) == (view["west"], view["south"], view["east"], view["north"])
    with np.load(npz) as check:
        assert np.array_equal(np.flipud(check["h0_m"]), h0)
        assert np.array_equal(np.flipud(check["reconstructed_patch_mask"]), mask)
    assert sha256(source) == source_hash
    metadata = dict(
        name="Feature I panel B: zero-outside initial test case",
        quantity="initial saturated aquifer thickness h0, metres",
        source=str(source.resolve()), source_sha256=source_hash,
        source_zoom_metadata=str(zoom_path.resolve()), source_reconstruction_modified=False,
        assumption="Retain reconstructed patch heights; set h0=0 in every other domain cell, including holes and buffer.",
        interpretation_of_zero="Assumed dry initial condition for a test case; not observed absence of water.",
        crs=str(crs), bounds_utm_m=view, resolution_m=[dx, dy],
        shape_yx=list(h0.shape), domain_area_km2=float(h0.size * area / 1e6),
        reconstructed_positive_cells=int(mask.sum()), assumed_zero_cells=int((~mask).sum()),
        initial_wet_area_km2=wet_area_km2, initial_bulk_aquifer_volume_m3=bulk_volume,
        volume_interpretation="Bulk saturated-layer volume; water volume requires porosity.",
        retained_positive_height_range_m=[float(h0[mask].min()), float(h0[mask].max())],
        color_limits_m=DISPLAY_COLOR_LIMITS_M, display_colormap=DISPLAY_COLORMAP,
        zero_display_color=DRY_COLOR,
        zero_display="Separate pale class; masked only in the positive-height display overlay.",
        resampling="none; exact aligned cell copy",
        geotiff=dict(path=str(tif.resolve()), zero_is_valid_data=True, nodata=None,
                     row_order="north to south"),
        npz=dict(path=str(npz.resolve()), field="h0_m", axes="[y, x]",
                 row_order="south to north", coordinates="Cell centres; x_m/y_m relative to utm_origin_m.",
                 provenance="reconstructed_patch_mask: true=retained patch, false=assumed zero"),
        preview=str(image.resolve()), simulation_performed=False,
        validation=dict(all_cells_finite=True, all_outside_cells_exactly_zero=True,
                        patch_values_exactly_preserved=True, all_source_patches_retained=True,
                        initial_bulk_volume_preserved=True, outermost_cells_zero=True,
                        geotiff_and_npz_round_trip_passed=True))
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Feature I: initial test case\n\n"
        "This uses the zoomed panel-B domain with its 2 km buffer. Original reconstructed "
        "cell heights are retained exactly, and every other cell (including internal holes) "
        "has initial saturated thickness **h0 = 0 m**. The zeros are a modeling assumption "
        "for the test case, not measurements of dry firn. Source reconstruction files are unchanged.\n\n"
        f"Grid: {h0.shape[1]} columns × {h0.shape[0]} rows at 250 m in EPSG:32624. "
        f"Initial positive footprint: {wet_area_km2:.2f} km² ({mask.sum()} cells); "
        f"assumed-zero cells: {(~mask).sum()}.\n\n"
        "- `feature1_initial_height_250m.tif`: north-up height raster in metres; "
        "zero is valid data and no cells are NoData.\n"
        "- `feature1_initial_height.npz`: `h0_m[y,x]`, `reconstructed_patch_mask[y,x]`, "
        "cell-centre coordinates and grid spacing. Both NPZ axes increase east/north; "
        "row 0 is the southern row. `x_m`/`y_m` are local coordinates relative to "
        "`utm_origin_m`; `x_utm_m`/`y_utm_m` are absolute coordinates.\n"
        "- `metadata.json`: bounds, assumptions, provenance and validation.\n\n"
        "```python\nimport numpy as np\n"
        "with np.load('feature1_initial_height.npz') as data:\n"
        "    h0 = data['h0_m']\n    patch_mask = data['reconstructed_patch_mask']\n"
        "    x, y = data['x_m'], data['y_m']\n```\n\n"
        "Only the initial thickness field is initialized here. No expansion simulation, "
        "boundary conditions or extrapolation of aquifer-base elevations is included. "
        "Reproduce from the repository root with:\n\n"
        "```sh\n/opt/anaconda3/bin/python3 src/initialize_feature1_test_case.py\n```\n")
    print(json.dumps({k: metadata[k] for k in ["name", "shape_yx", "domain_area_km2",
        "reconstructed_positive_cells", "assumed_zero_cells", "initial_wet_area_km2",
        "initial_bulk_aquifer_volume_m3", "validation"]}, indent=2))


if __name__ == "__main__":
    main()
