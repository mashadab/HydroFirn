"""Plot initial Feature I height, GIMP elevation, and x/y slopes in a 2x2 layout."""

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
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib import font_manager
from matplotlib.text import Text
import matplotlib.patheffects as pe
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, bounds as window_bounds, from_bounds


ROOT = Path(__file__).resolve().parents[1]
REGION = ROOT / "outputs/firn_aquifer/gimp_region"
OUT = ROOT / "outputs/firn_aquifer/feature1_test_case"
DRY_COLOR = "#eef2f5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    case = json.loads((OUT / "metadata.json").read_text())
    view = case["bounds_utm_m"]
    paths = [OUT / "feature1_initial_height_250m.tif",
             REGION / "feature1_gimp_surface_30m.tif",
             REGION / "feature1_gimp_xy_slopes_degrees.tif"]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    slope_meta = json.loads((REGION / "feature1_gimp_xy_slopes_metadata.json").read_text())
    assert hashes[str(paths[1].resolve())] == slope_meta["source_sha256"]
    with rasterio.open(paths[0]) as src:
        h0, crs = src.read(1), src.crs
        assert tuple(src.bounds) == (view["west"], view["south"], view["east"], view["north"])
        assert np.isfinite(h0).all() and (h0 >= 0).all()
        wet = (h0 > 0).astype(np.uint8)
        boundaries = []
        for polygon, value in shapes(wet, mask=wet.astype(bool), transform=src.transform):
            if value == 1:
                boundaries.extend(np.asarray(ring) / 1000 for ring in polygon["coordinates"])
    with rasterio.open(paths[1]) as src:
        assert src.crs == crs
        raw = from_bounds(view["west"], view["south"], view["east"], view["north"], src.transform)
        c0, r0 = int(np.floor(raw.col_off)), int(np.floor(raw.row_off))
        c1 = int(np.ceil(raw.col_off + raw.width))
        r1 = int(np.ceil(raw.row_off + raw.height))
        window = Window(c0, r0, c1 - c0, r1 - r0)
        z = src.read(1, window=window, masked=True).astype(float).filled(np.nan)
        transform = src.transform
        west, south, east, north = window_bounds(window, transform)
    with rasterio.open(paths[2]) as src:
        assert src.crs == crs and src.transform == transform
        theta = src.read(window=window, masked=True).astype(float).filled(np.nan)
    assert np.isfinite(z).all() and np.isfinite(theta).all()
    terrain_extent = [west / 1000, east / 1000, south / 1000, north / 1000]
    view_extent = [view["west"] / 1000, view["east"] / 1000,
                   view["south"] / 1000, view["north"] / 1000]

    font_family = "DejaVu Sans"
    font_manager.findfont(font_manager.FontProperties(family=font_family), fallback_to_default=False)
    plt.rcParams.update({"font.family": font_family, "font.size": 12,
                         "axes.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 17), facecolor="white")
    map_width = .37
    map_height = map_width * 12 / 17 * (view["north"] - view["south"]) / (view["east"] - view["west"])
    axes = [fig.add_axes([x, y, map_width, map_height])
            for y in [.59, .155] for x in [.105, .565]]
    fields = [np.ma.masked_equal(h0, 0), z, theta[0], theta[1]]
    norms = [Normalize(0, 25), Normalize(1500, 1800),
             TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3), TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)]
    cmaps = ["YlGnBu", "cividis", "RdBu_r", "RdBu_r"]
    titles = ["(a)  Aquifer height", "(b)  Surface elevation",
              "(c)  x slope", "(d)  y slope"]
    images = []
    for i, (ax, field, title, norm, cmap) in enumerate(zip(axes, fields, titles, norms, cmaps)):
        ax.set_facecolor(DRY_COLOR)
        images.append(ax.imshow(field, extent=view_extent if i == 0 else terrain_extent,
                                origin="upper", interpolation="nearest", cmap=cmap, norm=norm))
        if i > 0:
            ax.add_collection(LineCollection(boundaries, colors="#263c48", linewidths=.65,
                path_effects=[pe.Stroke(linewidth=1.5, foreground="white"), pe.Normal()]))
        ax.set(xlim=view_extent[:2], ylim=view_extent[2:])
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=15, fontweight="semibold",
                     pad=14, color="#203746")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.tick_params(colors="#344653", width=.7, labelsize=12, labelleft=i % 2 == 0)
        if i % 2 == 0:
            ax.set_ylabel("UTM northing (km)", color="black", labelpad=9)
        if i >= 2:
            ax.set_xlabel("UTM easting (km)", color="black", labelpad=8)
        for spine in ax.spines.values():
            spine.set_color("#9aa8b1")
            spine.set_linewidth(.7)

    fig.suptitle("Feature I", x=.105, y=.975, ha="left",
                 fontsize=21, fontweight="semibold", color="#203746")

    bar_specs = [(images[0], [.105, .54, map_width, .012], [0, 5, 10, 15, 20, 25],
                  "Height (m)", "neither"),
                 (images[1], [.565, .54, map_width, .012], [1500, 1600, 1700, 1800],
                  "Elevation (m)", "neither"),
                 (images[2], [.105, .063, .83, .012], [-3, -2, -1, 0, 1, 2, 3],
                  "Slope (°)", "min")]
    for image, position, ticks, label, extend in bar_specs:
        cax = fig.add_axes(position)
        cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks,
                          extend=extend, extendfrac=.025)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=3, width=.7, colors="#344653", labelsize=11, pad=5)
        cax.set_title(label, loc="left", fontsize=12, color="black", pad=10)
    fig.legend(handles=[Patch(facecolor=DRY_COLOR, edgecolor="#a6b1ba", label="Height = 0"),
                        Line2D([], [], color="#263c48", lw=.8, label="Aquifer footprint")],
               loc="upper left", bbox_to_anchor=(.10, .029), ncol=2, frameon=False,
               fontsize=11, handlelength=2, columnspacing=3)
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            font = font_manager.findfont(text.get_fontproperties(), fallback_to_default=False)
            assert font_manager.FontProperties(fname=font).get_name() == font_family
    output = OUT / "feature1_four_panel.png"
    fig.savefig(output, dpi=240)
    plt.close(fig)
    assert all(digest(path) == hashes[str(path.resolve())] for path in paths)
    metadata = dict(output=str(output.resolve()), layout="2 rows x 2 columns",
        panel_order=["Feature I initial aquifer height", "GIMP surface elevation", "x slope", "y slope"],
        view_bounds_utm_m=view, crs=str(crs), source_sha256=hashes,
        source_elevation_documentation="https://nsidc.org/data/nsidc-0645/versions/1",
        aquifer_height_grid_m=250, elevation_and_slope_grid_m=30,
        height_color_limits_m=[0, 25], elevation_color_limits_m=[1500, 1800],
        slope_color_limits_degrees=[-3, 3], zero_display_color=DRY_COLOR,
        zero_interpretation="Assumed dry initial condition outside reconstructed patches",
        elevation_reference="WGS84 ellipsoid", nominal_dem_epoch=2007,
        angle_formulas={"x": "degrees(arctan(dz/dx))", "y": "degrees(arctan(dz/dy))"},
        slope_method=slope_meta["method"],
        validation=dict(all_four_panels_same_view=True, all_inputs_unchanged=True,
                        elevation_matches_slope_source=True, all_display_fields_finite=True),
        resampling="none; native fields displayed with nearest-neighbour interpolation",
        font_family=font_family, heading_weight="semibold", all_visible_text_fonts_verified=True,
        font_reference="src/extract_montgomery_surface_transect.py and its SVG export (DejaVu Sans)",
        heading="Feature I", panel_titles=titles, subtitle_removed=True, explanatory_footer_removed=True)
    (OUT / "feature1_four_panel_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output.resolve()), "validation": metadata["validation"]}, indent=2))


if __name__ == "__main__":
    main()
