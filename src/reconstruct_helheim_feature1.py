"""Reconstruct a supported, composite 3D geometry for Chu's Feature I.

Run with a scientific Python containing numpy, scipy, matplotlib, rasterio,
pyproj and plotly. Source files are read without changes. No simulation runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helheim_geometry_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/helheim_geometry_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import label, map_coordinates
from scipy.spatial import cKDTree, Delaunay


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent / "CFA3D_Application" / "Realistic_data"
DEFAULT_DEM = (ROOT.parent / "meltwater-percolation" / "Infiltration" / "JPL"
               / "Greenland_surface_elevation" / "combined_greenland_tif.vrt")
OUTPUT = ROOT / "outputs" / "firn_aquifer" / "feature1_3d"
CHU_DOI = "https://doi.org/10.1029/2018GL079751"
MIEGE_DOI = "https://doi.org/10.1002/2016JF003869"
DATA_DOI = "https://doi.org/10.18739/A2TM72225"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    p.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--spacing-m", type=float, default=250.)
    p.add_argument("--max-distance-m", type=float, default=5000.)
    p.add_argument("--max-triangle-edge-m", type=float, default=10000.)
    p.add_argument("--min-thickness-m", type=float, default=3.9)
    p.add_argument("--vertical-layers", type=int, default=6)
    return p.parse_args()


def read_xlsx(path):
    """Read the first worksheet, keeping Excel row numbers for provenance."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", ns):
                strings.append("".join(t.text or "" for t in si.findall(".//s:t", ns)))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = wb.find("s:sheets/s:sheet", ns)
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in rels.findall("r:Relationship", rel_ns)
                      if r.attrib["Id"] == rid)
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        records = []
        for row in ET.fromstring(z.read(target)).findall("s:sheetData/s:row", ns):
            cols = {}
            for c in row.findall("s:c", ns):
                ref = "".join(ch for ch in c.attrib["r"] if ch.isalpha())
                v = c.find("s:v", ns)
                if v is None or v.text is None:
                    continue
                cols[ref] = strings[int(v.text)] if c.attrib.get("t") == "s" else v.text
            try:
                values = [float(cols[k]) for k in ["A", "B", "C"]]
            except (KeyError, ValueError):
                continue
            if np.isfinite(values).all():
                records.append([int(row.attrib["r"]), *values])
    a = np.asarray(records, dtype=float)
    if len(a) < 3 or np.any(a[:, 3] < 0):
        raise ValueError("Expected nonnegative, finite georeferenced thickness points.")
    return a, sheet.attrib["name"]


def read_depths(paths):
    rows = []
    invalid = 0
    for year, path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not {"LAT", "LONG", "ELEV", "DEPTH"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected columns in {path}")
            for line, row in enumerate(reader, 2):
                try:
                    v = [float(row[k]) for k in ["LAT", "LONG", "ELEV", "DEPTH"]]
                except ValueError:
                    invalid += 1
                    continue
                if np.isfinite(v).all() and v[3] > 0:
                    rows.append([year, line, *v])
                else:
                    invalid += 1
    return np.asarray(rows, dtype=float), invalid


def unique_observations(xy, values):
    """Median only at EXACT repeated projected positions; do not bin tracks."""
    unique, inverse = np.unique(xy, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    vals = np.empty(len(unique))
    for i in range(len(unique)):
        vals[i] = np.median(values[inverse == i])
    return unique, vals, counts


class LinearField:
    def __init__(self, xy, values):
        self.xy = xy
        self.values = values
        self.tri = Delaunay(xy)
        self.interpolate = LinearNDInterpolator(self.tri, values)
        self.tree = cKDTree(xy)
        vertices = xy[self.tri.simplices]
        self.max_edge = np.max(np.stack([
            np.linalg.norm(vertices[:, i] - vertices[:, (i+1) % 3], axis=1)
            for i in range(3)]), axis=0)

    def evaluate(self, query):
        simplex = self.tri.find_simplex(query)
        edge = np.full(len(query), np.nan)
        inside = simplex >= 0
        edge[inside] = self.max_edge[simplex[inside]]
        return np.asarray(self.interpolate(query)), self.tree.query(query)[0], edge


def sample_dem(dataset, x, y):
    """Pixel-center bilinear sampling, preserving nodata and out-of-bounds."""
    c, r = (~dataset.transform) * (np.asarray(x), np.asarray(y))
    c, r = c - .5, r - .5
    left = max(0, int(np.floor(c.min())) - 1)
    top = max(0, int(np.floor(r.min())) - 1)
    right = min(dataset.width, int(np.ceil(c.max())) + 3)
    bottom = min(dataset.height, int(np.ceil(r.max())) + 3)
    if right <= left or bottom <= top:
        return np.full(c.shape, np.nan)
    data = dataset.read(1, window=Window(left, top, right-left, bottom-top),
                        masked=True).astype(float).filled(np.nan)
    return map_coordinates(data, [r-top, c-left], order=1, mode="constant",
                           cval=np.nan, prefilter=False)


def spatial_cv(xy, values, max_distance, max_edge):
    """Five folds of 2 km spatial blocks, never random individual points."""
    blocks = np.floor((xy - xy.min(axis=0)) / 2000).astype(int)
    folds = (blocks[:, 0] * 3 + blocks[:, 1] * 7) % 5
    prediction = np.full(len(values), np.nan)
    for fold in range(5):
        train = folds != fold
        held = ~train
        if held.sum() == 0 or train.sum() < 4:
            continue
        model = LinearField(xy[train], values[train])
        pred, dist, edge = model.evaluate(xy[held])
        pred[(dist > max_distance) | (edge > max_edge)] = np.nan
        prediction[held] = pred
    good = np.isfinite(prediction)
    residual = prediction[good] - values[good]
    result = {
        "method": "5 folds, deterministic 2 km spatial blocks; same interpolation support limits",
        "total_points": len(values), "predicted_points": int(good.sum()),
        "coverage_fraction": float(good.mean()),
        "rmse_m": float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
        "mae_m": float(np.mean(np.abs(residual))) if len(residual) else None,
        "bias_m": float(np.mean(residual)) if len(residual) else None,
        "interpretation": "Interpolation diagnostic; not independent field validation or a confidence interval."
    }
    return result, prediction, folds


def cell_mask(nodes):
    return nodes[:-1, :-1] & nodes[:-1, 1:] & nodes[1:, 1:] & nodes[1:, :-1]


def build_mesh(active, x, y, top, base, layers):
    """Triangulate active quads and extrude wedges; preserve every active cell.

    Separate vertex fans at diagonal-only contacts to avoid nonmanifold joins.
    Side walls close computational cuts, not independently measured margins.
    """
    ny, nx = active.shape
    key_to_vertex = {}
    columns = []
    quads = []
    ij = []
    for j, i in zip(*np.nonzero(active)):
        quad = []
        for r, c in [(j, i), (j, i+1), (j+1, i+1), (j+1, i)]:
            adjacent = [(rr, cc) for rr, cc in [(r-1,c-1),(r-1,c),(r,c),(r,c-1)]
                        if 0 <= rr < ny and 0 <= cc < nx and active[rr, cc]]
            diagonal = (len(adjacent) == 2
                        and abs(adjacent[0][0]-adjacent[1][0]) == 1
                        and abs(adjacent[0][1]-adjacent[1][1]) == 1)
            key = (r, c, j, i) if diagonal else (r, c)
            if key not in key_to_vertex:
                key_to_vertex[key] = len(columns)
                columns.append((x[c], y[r], base[r, c], top[r, c]))
            quad.append(key_to_vertex[key])
        quads.append(quad)
        ij.append((j, i))
    columns = np.asarray(columns)
    quads = np.asarray(quads, dtype=int)
    triangles = np.concatenate([quads[:, [0,1,2]], quads[:, [0,2,3]]])
    n = len(columns)
    boundary_points = np.concatenate([
        columns[:, [0,1,2]], columns[:, [0,1,3]]])
    faces = [triangles[:, ::-1], triangles+n]
    walls = []
    for quad, (j, i) in zip(quads, ij):
        for k, (dj, di) in enumerate([(-1,0),(0,1),(1,0),(0,-1)]):
            jj, ii = j+dj, i+di
            if not (0 <= jj < ny and 0 <= ii < nx and active[jj, ii]):
                a, b = quad[k], quad[(k+1) % 4]
                walls.extend([[a,b,b+n], [a,b+n,a+n]])
    if walls:
        faces.append(np.asarray(walls))
    faces = np.concatenate(faces)
    sigma = np.linspace(0., 1., layers+1)
    vpoints = np.concatenate([np.column_stack([
        columns[:, 0], columns[:, 1], columns[:, 2] + s*(columns[:, 3]-columns[:, 2])])
        for s in sigma])
    wedges = np.concatenate([np.column_stack([triangles+k*n, triangles+(k+1)*n])
                             for k in range(layers)])
    return boundary_points, faces, vpoints, wedges, np.asarray(ij), columns, triangles


def mesh_checks(points, faces):
    edges = np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]])
    canonical = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    p = points - np.mean(points, axis=0)
    tri = p[faces]
    cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    volume = np.einsum("ij,ij->i", tri[:,0], np.cross(tri[:,1],tri[:,2])).sum()/6
    return {
        "boundary_vertices": len(points), "boundary_triangles": len(faces),
        "edges_not_shared_by_exactly_two_faces": int(np.count_nonzero(counts != 2)),
        "edges_with_inconsistent_face_orientation": int(np.count_nonzero(orientation)),
        "degenerate_faces": int(np.count_nonzero(np.linalg.norm(cross, axis=1) < 1.e-8)),
        "signed_volume_m3": float(volume),
    }


def write_obj(path, points, faces):
    with path.open("w") as f:
        f.write("# Helheim Feature I 2012-2014 composite, supported subset\n")
        f.write("# x/y: local east/north metres, z: WGS84 ellipsoid metres\n")
        f.write("# Origin, assumptions and sources: metadata.json and README.md\n")
        np.savetxt(f, points, fmt="v %.8f %.8f %.8f")
        np.savetxt(f, faces+1, fmt="f %d %d %d")


def write_vtu(path, points, wedges, ij, active, H, max_dist, components, layers):
    """Portable VTK XML volume mesh with provenance-related cell fields."""
    j, i = ij.T
    repeat = lambda a: np.tile(np.concatenate([a, a]), layers)
    thickness = (H[j,i]+H[j,i+1]+H[j+1,i+1]+H[j+1,i])/4
    distance = np.maximum.reduce([max_dist[j,i],max_dist[j,i+1],max_dist[j+1,i+1],max_dist[j+1,i]])
    with path.open("w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<UnstructuredGrid><Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(wedges)}">\n')
        f.write('<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, points, fmt="%.9f")
        f.write('</DataArray></Points><Cells>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, wedges, fmt="%d")
        f.write('</DataArray><DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(f, 6*np.arange(1,len(wedges)+1)[None,:], fmt="%d")
        f.write('</DataArray><DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(f, np.full((1,len(wedges)),13), fmt="%d")
        f.write('</DataArray></Cells><CellData>\n')
        fields = {"aquifer_thickness_m": repeat(thickness),
                  "maximum_observation_distance_m": repeat(distance),
                  "component_id": repeat(components[j,i])}
        for name, arr in fields.items():
            f.write(f'<DataArray type="Float64" Name="{name}" format="ascii">\n')
            np.savetxt(f, arr[None,:], fmt="%.8g")
            f.write('</DataArray>\n')
        f.write('</CellData></Piece></UnstructuredGrid></VTKFile>\n')


def save_csv(path, header, values, fmt="%.12g"):
    np.savetxt(path, values, delimiter=",", header=",".join(header), comments="", fmt=fmt)


def make_figure(output, x, y, H, D, active, dist, chu, depth, points, faces, summary, top, base, surface, components):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig = plt.figure(figsize=(14,10), facecolor="#f8fafb")
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1], hspace=.32, wspace=.30)
    ax3 = fig.add_subplot(gs[0,0], projection="3d", facecolor="#f8fafb")
    counts = np.bincount(components[active])
    largest = int(np.argmax(counts[1:])+1)
    detail_mask = components == largest
    detail_points,detail_faces,_,_,_,_,detail_triangles = build_mesh(detail_mask,x,y,top,base,1)
    transformed = detail_points.copy()
    transformed[:, :2] /= 1000
    # Show top surface colored by actual reconstructed thickness; base and walls below.
    ntop = len(detail_triangles)
    normalizer = Normalize(0, max(25, float(summary["mesh_thickness_range_m"][1])))
    colors = np.empty((len(detail_faces),4))
    colors[:] = [.10,.28,.38,.30]
    top_faces = detail_faces[ntop:2*ntop]
    node_thickness = detail_points[len(detail_points)//2:,2]-detail_points[:len(detail_points)//2,2]
    hface = node_thickness[(top_faces-len(detail_points)//2)].mean(axis=1)
    colors[ntop:2*ntop] = plt.cm.viridis(normalizer(hface))
    colors[ntop:2*ntop,3] = .98
    collection = Poly3DCollection(transformed[detail_faces], facecolors=colors,
                                 edgecolors="none", zsort="average")
    ax3.add_collection3d(collection)
    lo,hi = transformed.min(axis=0),transformed.max(axis=0)
    ax3.set(xlim=(lo[0]-.3,hi[0]+.3),ylim=(lo[1]-.3,hi[1]+.3),
            zlim=(lo[2]-5,hi[2]+5),
            xlabel="Local easting (km)",ylabel="Local northing (km)",
            zlabel="Elevation (m)")
    ax3.view_init(elev=28,azim=-125)
    ax3.tick_params(labelsize=8,pad=1)
    for axis in [ax3.xaxis,ax3.yaxis,ax3.zaxis]:
        axis.set_major_locator(MaxNLocator(3))
    # Axis display aspect is explicit; data and mesh retain unscaled coordinates.
    ax3.set_box_aspect((np.ptp(detail_points[:,0]),np.ptp(detail_points[:,1]),15*np.ptp(detail_points[:,2])))
    ax3.set_title("3D detail: largest supported patch · vertical scale 15×",pad=8,fontsize=11)
    im = plt.cm.ScalarMappable(norm=normalizer,cmap="viridis")
    fig.colorbar(im, ax=ax3, shrink=.58, pad=.08, label="Thickness (m)")
    ax = fig.add_subplot(gs[0,1],facecolor="#f8fafb")
    j = int(np.argmax(detail_mask.sum(axis=1)))
    centers = lambda field: (field[j,:-1]+field[j,1:]+field[j+1,1:]+field[j+1,:-1])/4
    profile_top,profile_base = [np.where(detail_mask[j],centers(field),np.nan) for field in [top,base]]
    xp = (x[:-1]+x[1:])/2000
    take = np.flatnonzero(detail_mask[j])
    show = slice(max(0,take.min()-2),min(len(xp),take.max()+3))
    ax.plot(xp[show],centers(surface)[show],color="#596b77",label="GIMP surface",lw=2)
    ax.plot(xp,profile_top,color="#218da5",label="Water table",lw=2)
    ax.plot(xp,profile_base,color="#173e56",label="Aquifer base",lw=2)
    ax.fill_between(xp,profile_base,profile_top,color="#45b6c4",alpha=.45)
    ax.set(xlim=(xp[show].min(),xp[show].max()),xlabel="Local easting (km)",
           ylabel="Elevation (m, WGS84 ellipsoid)",
           title=f"Section through largest patch · northing {(y[j]+y[j+1])/2000:.2f} km")
    ax.grid(alpha=.18); ax.legend(frameon=False,fontsize=9)
    extent = (x.min()/1000,x.max()/1000,y.min()/1000,y.max()/1000)
    crop = (points[:,0].min()/1000-1,points[:,0].max()/1000+1,
            points[:,1].min()/1000-1,points[:,1].max()/1000+1)
    centers_h = (H[:-1,:-1]+H[:-1,1:]+H[1:,1:]+H[1:,:-1])/4
    centers_dist = np.maximum.reduce([dist[:-1,:-1],dist[:-1,1:],dist[1:,1:],dist[1:,:-1]])
    for k, (field,cmap,title,unit) in enumerate([
        (centers_h,"viridis","Thickness and source observations","Thickness (m)"),
        (centers_dist/1000,"magma_r","Distance to observations","Maximum nearest-source distance (km)")]):
        ax = fig.add_subplot(gs[1,k],facecolor="#edf1f4")
        display = np.where(active,field,np.nan)
        im = ax.imshow(display,origin="lower",extent=extent,cmap=cmap,interpolation="nearest")
        if k == 0:
            ax.scatter(chu[:,0]/1000,chu[:,1]/1000,s=2,c="#162f40",alpha=.4,label="Chu thickness")
            ax.scatter(depth[:,0]/1000,depth[:,1]/1000,s=8,marker="+",c="#cb5c28",alpha=.65,label="Miège water table")
            ax.legend(loc="lower left",fontsize=8,framealpha=.9)
        ax.set(xlabel="Local easting (km)",ylabel="Local northing (km)",title=title)
        ax.set_xlim(crop[:2]); ax.set_ylim(crop[2:]); ax.set_aspect("equal")
        fig.colorbar(im,ax=ax,shrink=.88,pad=.025,label=unit)
    fig.suptitle("Helheim firn aquifer · Feature I",x=.065,y=.985,ha="left",fontsize=21,fontweight="bold",color="#17384b")
    fig.text(.065,.947,f"2012–2014 composite · radar-constrained subset · {summary['grid_spacing_m']:g} m grid; between-track surfaces are interpolated",
             fontsize=11,color="#526773")
    fig.text(.065,.035,
             f"Mesh footprint {summary['mesh_area_km2']:.1f} km²   |   Bulk aquifer volume {summary['mesh_volume_km3']:.2f} km³ (includes ice matrix)\n"
             "Interpolation between tracks is assumed. Missing areas are unspecified; mesh side walls are computational cuts.",
             fontsize=10,color="#526773",linespacing=1.6)
    fig.subplots_adjust(left=.065,right=.94,bottom=.11,top=.90)
    fig.savefig(output/"feature1_geometry_preview.png",dpi=180)
    plt.close(fig)


def make_interactive(output, points, faces, chu_xyz, depth_xyz, summary):
    import plotly.graph_objects as go
    p = points.copy()
    p[:,:2] /= 1000
    n = len(points)//2
    thickness = points[n:,2]-points[:n,2]
    traces = [go.Mesh3d(
        x=p[:,0],y=p[:,1],z=p[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
        intensity=np.tile(thickness,2),colorscale="Viridis",cmin=0,
        cmax=max(25,float(thickness.max())),colorbar=dict(title="Thickness (m)"),
        name="Aquifer solid",opacity=.94,flatshading=False,
        hovertemplate="East %{x:.2f} km<br>North %{y:.2f} km<br>Elevation %{z:.1f} m<extra>Aquifer</extra>")]
    for xyz, name, color in [(chu_xyz,"Chu thickness samples","#dd6d30"),
                             (depth_xyz,"Miège water-table samples","#263f50")]:
        traces.append(go.Scatter3d(x=xyz[:,0]/1000,y=xyz[:,1]/1000,z=xyz[:,2],
                                  mode="markers",marker=dict(size=2,color=color),name=name,
                                  visible="legendonly",hoverinfo="name+x+y+z"))
    span = np.ptp(p,axis=0)
    horizontal = max(span[:2])
    ratios = dict(x=span[0]/horizontal,y=span[1]/horizontal,z=15*span[2]/(horizontal*1000))
    true_ratios = dict(x=ratios["x"],y=ratios["y"],z=span[2]/(horizontal*1000))
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text="Helheim Feature I | 2012–2014 composite<br><sup>Interpolated subset; computational cut margins · choose vertical scale below</sup>",font=dict(size=22)),
        paper_bgcolor="#f8fafb",font=dict(family="Arial",color="#17384b"),
        scene=dict(xaxis_title="Local easting (km)",yaxis_title="Local northing (km)",
                   zaxis_title="Elevation (m, WGS84 ellipsoid)",aspectmode="manual",aspectratio=ratios,
                   camera=dict(eye=dict(x=-1.5,y=-1.7,z=1.0))),
        updatemenus=[dict(type="buttons",direction="right",x=.02,y=.02,buttons=[
            dict(label="15× vertical",method="relayout",args=[{"scene.aspectratio":ratios}]),
            dict(label="True scale",method="relayout",args=[{"scene.aspectratio":true_ratios}])])],
        legend=dict(x=.02,y=.88),margin=dict(l=0,r=0,t=95,b=55),
        annotations=[dict(text=f"250 m grid · {summary['mesh_area_km2']:.1f} km² subset · not a dated full-aquifer survey",xref="paper",yref="paper",x=.5,y=0,showarrow=False)])
    fig.write_html(output/"feature1_geometry_3d.html",include_plotlyjs=True,full_html=True)


def main():
    args = arguments()
    if (not np.isfinite([args.spacing_m,args.max_distance_m,args.max_triangle_edge_m,args.min_thickness_m]).all()
        or min(args.spacing_m,args.max_distance_m,args.max_triangle_edge_m) <= 0
        or args.min_thickness_m < 0 or args.vertical_layers < 1):
        raise ValueError("Require positive spacing/support limits/layers and nonnegative thickness cutoff.")
    args.output.mkdir(parents=True,exist_ok=True)
    chu_path = args.source_root/"Chu_et_al_2018"/"grl58203-sup-0002-2018gl079751_s01.xlsx"
    depth_paths = [(year,args.source_root/f"MiegeFirnAquiferDetections{year}.csv") for year in [2012,2014]]
    chu_raw, sheet = read_xlsx(chu_path)
    depth_raw, invalid = read_depths(depth_paths)
    forward = Transformer.from_crs(4326,32624,always_xy=True)
    inverse = Transformer.from_crs(32624,4326,always_xy=True)
    chu_utm = np.column_stack(forward.transform(chu_raw[:,2],chu_raw[:,1]))
    depth_utm = np.column_stack(forward.transform(depth_raw[:,3],depth_raw[:,2]))
    near = cKDTree(chu_utm).query(depth_utm)[0] <= 15000
    depth_raw, depth_utm = depth_raw[near],depth_utm[near]
    origin = np.floor(chu_utm.min(axis=0)/1000)*1000
    chu_xy, depth_xy = chu_utm-origin,depth_utm-origin
    cx,ch,cc = unique_observations(chu_xy,chu_raw[:,3])
    dx,dd,dc = unique_observations(depth_xy,depth_raw[:,5])
    hf,df = LinearField(cx,ch),LinearField(dx,dd)
    spacing = args.spacing_m
    x = np.arange(np.floor(chu_xy[:,0].min()/spacing)*spacing,
                  np.ceil(chu_xy[:,0].max()/spacing)*spacing+spacing/2,spacing)
    y = np.arange(np.floor(chu_xy[:,1].min()/spacing)*spacing,
                  np.ceil(chu_xy[:,1].max()/spacing)*spacing+spacing/2,spacing)
    if len(x)*len(y) > 1_000_000:
        raise ValueError("More than one million horizontal nodes; choose coarser spacing.")
    X,Y = np.meshgrid(x,y)
    query = np.column_stack([X.ravel(),Y.ravel()])
    H,dh,eh = hf.evaluate(query)
    D,ddist,ed = df.evaluate(query)
    lon,lat = inverse.transform(query[:,0]+origin[0],query[:,1]+origin[1])
    with rasterio.open(args.dem) as dem:
        to_dem = Transformer.from_crs(4326,dem.crs,always_xy=True)
        px,py = to_dem.transform(lon,lat)
        surface = sample_dem(dem,px,py)
        px,py = to_dem.transform(depth_raw[:,3],depth_raw[:,2])
        depth_dem = sample_dem(dem,px,py)
        px,py = to_dem.transform(chu_raw[:,2],chu_raw[:,1])
        chu_dem = sample_dem(dem,px,py)
        dem_meta = dict(crs=str(dem.crs),posting_m=list(dem.res),nodata=dem.nodata)
    supported = (np.isfinite(H)&np.isfinite(D)&np.isfinite(surface)&(D>0)&(H>=0)
                 &(dh<=args.max_distance_m)&(ddist<=args.max_distance_m)
                 &(eh<=args.max_triangle_edge_m)&(ed<=args.max_triangle_edge_m))
    shape = X.shape
    H,D,surface,dh,ddist,eh,ed,supported = [a.reshape(shape) for a in [H,D,surface,dh,ddist,eh,ed,supported]]
    H = np.where(supported,H,np.nan)
    D = np.where(supported,D,np.nan)
    top = surface-D
    base = top-H
    active = cell_mask(supported&(H>=args.min_thickness_m)&(H>0))
    if active.sum() == 0:
        raise ValueError("No supported aquifer cells under the chosen criteria.")
    components,ncomponents = label(active)
    component_sizes = np.bincount(components[active])
    points,faces,vpoints,wedges,ij,columns,triangles = build_mesh(active,x,y,top,base,args.vertical_layers)
    checks = mesh_checks(points,faces)
    coords = columns[triangles,:2]
    twice_area = np.abs(np.cross(coords[:,1]-coords[:,0],coords[:,2]-coords[:,0]))
    triangle_h = (columns[:,3]-columns[:,2])[triangles].mean(axis=1)
    integral_volume = float(np.sum(twice_area/2*triangle_h))
    checks["independent_thickness_integral_m3"] = integral_volume
    checks["relative_mesh_volume_error"] = abs(checks["signed_volume_m3"]-integral_volume)/integral_volume
    checks["maximum_surface_ordering_violation_m"] = float(np.maximum(base[supported]-top[supported],0).max())
    checks["minimum_water_table_depth_m"] = float(D[supported].min())
    checks["maximum_interpolant_recovery_error_thickness_m"] = float(np.nanmax(np.abs(hf.interpolate(cx)-ch)))
    checks["maximum_interpolant_recovery_error_depth_m"] = float(np.nanmax(np.abs(df.interpolate(dx)-dd)))
    lo,la = inverse.transform(chu_utm[:,0],chu_utm[:,1])
    checks["maximum_coordinate_roundtrip_error_deg"] = float(max(np.max(np.abs(lo-chu_raw[:,2])),np.max(np.abs(la-chu_raw[:,1]))))
    assert checks["edges_not_shared_by_exactly_two_faces"] == 0
    assert checks["edges_with_inconsistent_face_orientation"] == 0
    assert checks["degenerate_faces"] == 0
    assert checks["relative_mesh_volume_error"] < 1.e-8
    assert checks["maximum_surface_ordering_violation_m"] == 0
    assert checks["minimum_water_table_depth_m"] > 0
    hcv,hpred,hfold = spatial_cv(cx,ch,args.max_distance_m,args.max_triangle_edge_m)
    dcv,dpred,dfold = spatial_cv(dx,dd,args.max_distance_m,args.max_triangle_edge_m)
    checks["thickness_spatial_cross_validation"] = hcv
    checks["depth_spatial_cross_validation"] = dcv
    finite = np.isfinite(depth_dem)
    delta = depth_raw[finite,4]-depth_dem[finite]
    checks["source_ELEV_minus_GIMP_m"] = dict(count=int(finite.sum()),mean=float(delta.mean()),
        median=float(np.median(delta)),rmse=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.abs(delta).max()))
    band = 3.9
    summary = {
        "mesh_area_km2": float(active.sum()*spacing**2/1.e6),
        "mesh_volume_km3": integral_volume/1.e9,
        "bulk_volume_sensitivity_H_minus_3p9_km3": float(np.sum(twice_area/2*np.maximum(triangle_h-band,0))/1.e9),
        "bulk_volume_sensitivity_H_plus_3p9_km3": float(np.sum(twice_area/2*(triangle_h+band))/1.e9),
        "volume_interpretation": "Bulk aquifer material volume, includes firn ice; not liquid-water storage. Sensitivity is not a confidence interval.",
        "horizontal_cells": int(active.sum()),"volume_wedges":len(wedges),"volume_vertices":len(vpoints),
        "connected_components": int(ncomponents),
        "component_areas_km2_descending": sorted((component_sizes[1:]*spacing**2/1.e6).tolist(),reverse=True),
        "triangles_per_surface": len(triangles),
        "mesh_thickness_range_m": [float((columns[:,3]-columns[:,2]).min()),float((columns[:,3]-columns[:,2]).max())],
        "grid_shape_y_x":list(shape),"grid_spacing_m":spacing,
        "supported_node_count":int(supported.sum()),"total_node_count":int(supported.size),
        "spatial_resolution_note": "Grid sampling only; between-track observational resolution is much coarser."
    }
    sources = [chu_path,*(p for _,p in depth_paths)]
    manifest = [dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
    metadata = {
        "title":"Helheim Feature I, supported 3D reconstruction",
        "survey_period":"2012-2014 composite, not a single-date snapshot",
        "reconstruction_date":"2026-10-25",
        "source_root":str(args.source_root.resolve()),"source_manifest":manifest,
        "citations":{"thickness":CHU_DOI,"water_table":MIEGE_DOI,"water_table_dataset":DATA_DOI},
        "chu_worksheet":sheet,"chu_raw_points":len(chu_raw),"chu_unique_positions":len(cx),
        "miege_years":[2012,2014],"miege_selected_points":len(depth_raw),"miege_unique_positions":len(dx),
        "miege_selected_by_year":{str(yr):int((depth_raw[:,0]==yr).sum()) for yr in [2012,2014]},
        "miege_global_rejected_nonfinite_or_nonpositive_depth":invalid,
        "horizontal_crs":"EPSG:32624, WGS84 / UTM zone 24N",
        "local_origin_utm_easting_northing_m":origin.tolist(),
        "axes":"x east, y north, z up; metres. Add origin to x/y for absolute UTM.",
        "vertical_reference":"WGS84 ellipsoid, inherited from GIMP; no geoid correction or fitted elevation shift",
        "surface_DEM":{"path":str(args.dem.resolve()),"product":"NASA GIMP 30 m mosaic","nominal_epoch":2007,**dem_meta},
        "equations":["z_top = GIMP_surface - interpolated_water_table_depth","z_base = z_top - interpolated_Chu_thickness"],
        "interpolation":"Piecewise linear Delaunay in projected metre coordinates; no extrapolation. Exact duplicate positions use median.",
        "support":{"maximum_distance_to_each_source_m":args.max_distance_m,
                   "maximum_triangle_edge_in_each_source_m":args.max_triangle_edge_m,
                   "water_table_source_prefilter_distance_to_Chu_m":15000.,
                   "mesh_minimum_thickness_at_all_four_cell_corners_m":args.min_thickness_m,
                   "all_four_cell_corners_must_be_supported":True},
        "mesh":{"method":"Two planar triangles per grid quad, vertically extruded into six-node wedges",
                "sigma_layers":args.vertical_layers,"diagonal_only_contacts":"Vertex fans separated without deleting cells",
                "boundaries":"Side walls close the reconstruction cuts, not measured aquifer margins"},
        "limitations":[
            "Thickness workbook contains latitude, longitude and thickness only; individual acquisition years cannot be recovered from this file.",
            "2012 and 2014 water-table depths are combined; the result is a composite hypothetical state rather than an observed instantaneous volume.",
            "GIMP has nominal 2007 epoch; no epoch correction is applied to anchor the 2012-2014 relative depths.",
            "Between-track surfaces are interpolation assumptions. Nearest-source distance is a support diagnostic, not statistical uncertainty.",
            "3.9 m is a study-wide average thickness uncertainty used as a screening/sensitivity scale, not a local error model or confidence bound.",
            "Unrepresented areas remain unspecified. This is a supported subset, not a reconstruction of the published full Feature I footprint.",
            "The broad Miège 2010-2014 KML polygon is not a Feature I boundary and is not used to force the footprint.",
            "Montgomery 2015-2016 measurements are not imposed on this earlier composite geometry.",
            "No liquid volume, porosity, permeability, temperature, recharge, boundary conditions or transient simulation is assigned."
        ],"summary":summary
    }
    np.savez_compressed(args.output/"feature1_geometry.npz",
        x_m=x,y_m=y,longitude_deg=lon.reshape(shape),latitude_deg=lat.reshape(shape),
        utm_origin_m=origin,horizontal_crs_epsg=np.array(32624),
        surface_z_wgs84_ellipsoid_m=surface,water_table_z_m=top,base_z_m=base,
        aquifer_thickness_m=H,water_table_depth_m=D,supported_nodes=supported,active_cells=active,
        component_id=components,nearest_thickness_distance_m=dh,nearest_depth_distance_m=ddist,
        thickness_triangle_max_edge_m=eh,depth_triangle_max_edge_m=ed,
        boundary_vertices_xyz_m=points,boundary_triangles=faces,
        volume_vertices_xyz_m=vpoints,volume_wedges=wedges,
        sigma_levels=np.linspace(0,1,args.vertical_layers+1),
        metadata_json=np.array(json.dumps(metadata)))
    save_csv(args.output/"chu_source_points.csv",["excel_row","latitude_deg","longitude_deg","thickness_m","local_x_m","local_y_m"],np.column_stack([chu_raw,chu_xy]))
    save_csv(args.output/"miege_source_points.csv",["year","csv_line","latitude_deg","longitude_deg","ELEV_as_reported_m","water_table_depth_m","local_x_m","local_y_m","GIMP_surface_ellipsoid_m"],np.column_stack([depth_raw,depth_xy,depth_dem]))
    save_csv(args.output/"thickness_spatial_cv.csv",["x_m","y_m","observed_m","withheld_prediction_m","fold"],np.column_stack([cx,ch,hpred,hfold]))
    save_csv(args.output/"depth_spatial_cv.csv",["x_m","y_m","observed_m","withheld_prediction_m","fold"],np.column_stack([dx,dd,dpred,dfold]))
    write_obj(args.output/"feature1_aquifer_boundary.obj",points,faces)
    write_vtu(args.output/"feature1_aquifer_volume.vtu",vpoints,wedges,ij,active,H,
              np.maximum(dh,ddist),components,args.vertical_layers)
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    (args.output/"validation.json").write_text(json.dumps(checks,indent=2,allow_nan=False)+"\n")
    make_figure(args.output,x,y,H,D,active,np.maximum(dh,ddist),chu_xy,depth_xy,points,faces,summary,top,base,surface,components)
    chu_depth = df.interpolate(chu_xy)
    chu_xyz = np.column_stack([chu_xy,chu_dem-chu_depth])
    depth_xyz = np.column_stack([depth_xy,depth_dem-depth_raw[:,5]])
    make_interactive(args.output,points,faces,chu_xyz,depth_xyz,summary)
    readme = f"""# Helheim Feature I: reconstructed 3D geometry

This is a **2012–2014 composite, supported subset**, not a measured full aquifer
or a dated initial condition. It is built from existing CFA3D_Application data.

## Result

- Horizontal grid: {spacing:g} m sampling, {shape[1]} × {shape[0]} nodes.
- Active footprint: {summary['mesh_area_km2']:.3f} km² in {ncomponents} supported components.
- Bulk aquifer material volume: {summary['mesh_volume_km3']:.4f} km³, including ice matrix.
- Mesh thickness: {summary['mesh_thickness_range_m'][0]:.2f}–{summary['mesh_thickness_range_m'][1]:.2f} m.
- Volume mesh: {len(wedges):,} triangular-prism wedges in {args.vertical_layers} sigma layers.
- No porosity has been assigned; bulk volume must not be interpreted as liquid-water volume.

## Files and coordinates

- `feature1_geometry_3d.html`: self-contained interactive preview; 15× vertical
  exaggeration initially, with a true-scale button and optional source points.
- `feature1_geometry_preview.png`: static 3D view and source-support maps.
- `feature1_geometry.npz`: gridded surface/top/base, thickness/depth, masks,
  support distances, connected components, and boundary/volume mesh arrays.
- `feature1_aquifer_volume.vtu`: ParaView-compatible volume mesh, with thickness,
  observation-distance and component fields.
- `feature1_aquifer_boundary.obj`: closed boundary triangles at true coordinate scale.
- `metadata.json`, `validation.json`: methods, source hashes, assumptions and checks.
- `*_source_points.csv`, `*_spatial_cv.csv`: traceable inputs and interpolation diagnostics.

All mesh x/y coordinates are **local east/north metres**. Add the origin
({origin[0]:.3f}, {origin[1]:.3f}) m to recover EPSG:32624 UTM coordinates.
Mesh z coordinates are **metres above the WGS84 ellipsoid**, not local thickness
or orthometric heights. Grid arrays have shape (ny, nx); active_cells and
component_id have shape (ny-1, nx-1). Coordinates increase eastward and northward.
Indices in NPZ/VTU are zero-based; OBJ indices are one-based. NPZ loads without pickle.

## Construction and limits

1. Read {len(chu_raw):,} Feature I thickness points from Chu Dataset S1, worksheet
   `{sheet}`. Preserve original Excel row IDs. There is no per-point date column.
2. Read 2012/2014 Miège points, retaining {len(depth_raw):,} finite, positive-depth
   records within 15 km of Chu tracks. Exact repeated coordinates use median values.
3. Interpolate thickness and depth separately with piecewise-linear Delaunay
   interpolation in UTM coordinates. There is no extrapolation outside either hull.
4. Sample the existing GIMP DEM bilinearly. Preserve nodata. Calculate top = surface
   minus water-table depth; base = top minus thickness. GIMP is nominally 2007;
   the mixed epoch is explicit, without an invented correction.
5. Require distance to **each** observation set ≤ {args.max_distance_m:g} m and
   interpolation triangle maximum edge ≤ {args.max_triangle_edge_m:g} m in **each** set.
   A cell enters the mesh only when all four corners pass and thickness is
   ≥ {args.min_thickness_m:g} m. This screening does not identify a physical margin.
6. Triangulate and extrude each accepted cell. Retain every component. Separate
   diagonal-only vertex fans to prevent nonmanifold contact. Close reconstruction
   cuts with vertical side walls. Those walls are artificial modeling boundaries.

Missing areas are unspecified, not evidence of physical absence. Grid spacing
does not imply equivalent observation resolution. The full published Feature I
area is not forced. The broad Miège KML footprint is not used as a Feature I mask.
Montgomery's later 2015–2016 transect is not imposed on the 2012–2014 composite.

The study-wide 3.9 m thickness uncertainty is a screening/sensitivity scale,
not a local standard deviation. On the **fixed accepted footprint**, subtracting
or adding 3.9 m gives bulk volumes {summary['bulk_volume_sensitivity_H_minus_3p9_km3']:.4f}–
{summary['bulk_volume_sensitivity_H_plus_3p9_km3']:.4f} km³. This is not a confidence
interval and excludes depth, DEM, temporal, boundary and interpolation uncertainty.

## Validation

Surface order is checked; every mesh edge has two consistently oriented incident
faces; faces have nonzero area. Closed-mesh volume agrees with the independently
integrated piecewise-linear thickness to relative error {checks['relative_mesh_volume_error']:.3g}.
Five-fold 2 km spatial-block withholding gives thickness RMSE {hcv['rmse_m']:.2f} m
on {hcv['predicted_points']}/{hcv['total_points']} predictable points and water-table
depth RMSE {dcv['rmse_m']:.2f} m on {dcv['predicted_points']}/{dcv['total_points']} points.
These are interpolation diagnostics, not external field validation. Source ELEV
minus GIMP has RMSE {checks['source_ELEV_minus_GIMP_m']['rmse']:.2f} m; source ELEV is
retained for comparison but is not assigned a new vertical datum or used to shift DEM.

## Sources and reproduction

- Chu, Schroeder & Siegfried (2018), thickness Dataset S1: {CHU_DOI}
- Miège et al. (2016), water-table depth: {MIEGE_DOI}
- Miège (2018), data archive: {DATA_DOI}
- NASA GIMP DEM: https://nsidc.org/data/nsidc-0645/versions/1

From HydroFirn, with numpy/scipy/matplotlib/pyproj/rasterio/plotly installed:

```sh
python src/reconstruct_helheim_feature1.py
```

Use `--help` to change spacing, support limits, thickness cutoff, sigma layers,
input paths or output directory. A time-dependent model additionally needs
material properties, thermal state, recharge and explicit boundary conditions.
"""
    (args.output/"README.md").write_text(readme)
    print(json.dumps({"output":str(args.output),"summary":summary,"validation":checks},indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
