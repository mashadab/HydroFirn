#!/usr/bin/env python3
"""Plot the RetMIP/HydroFirn/upGPR percolation-depth comparison from CSV.

The input must contain the columns written by
``plot_dye2_16_percolation_depth.py``:

    datetime_utc, series, percolation_depth_m, criterion, threshold

Example
-------
python plot_percolation_depth_from_csv.py Dye-2_16_percolation_depth_data.csv
python plot_percolation_depth_from_csv.py data.csv --output comparison.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


SERIES_ORDER = [
    "CFM-Cr",
    "CFM-KM",
    "DTU",
    "DMIHH",
    "GEUS",
    "IMAUFDM",
    "MeyerHewitt",
    "UppsalaUniBucket",
    "UppsalaUniDeepPerc",
    "HydroFirn",
    "upGPR",
]

# ColorBrewer Set1 in the model order used by the RetMIP paper.
SERIES_COLORS = {
    "CFM-Cr": "#E41A1C",
    "CFM-KM": "#377EB8",
    "DTU": "#4DAF4A",
    "DMIHH": "#984EA3",
    "GEUS": "#FF7F00",
    "IMAUFDM": "#FFFF33",
    "MeyerHewitt": "#A65628",
    "UppsalaUniBucket": "#F781BF",
    "UppsalaUniDeepPerc": "#999999",
    "HydroFirn": (135 / 255, 206 / 255, 235 / 255),
    "upGPR": "black",
}

LINE_WIDTHS = {
    **{name: 3.0 for name in SERIES_ORDER[:9]},
    "HydroFirn": 4.0,
    "upGPR": 4.0,
}

X_START = datetime(2016, 5, 2, 0, 0)
X_END = datetime(2016, 10, 28, 9, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("Dye-2_16_percolation_depth_data.csv"),
        help="tidy percolation-depth CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path; defaults to the CSV name without '_data'",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def read_plot_data(path: Path) -> dict[str, tuple[list[datetime], list[float]]]:
    required = {"datetime_utc", "series", "percolation_depth_m"}
    grouped: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row["datetime_utc"])
                depth = float(row["percolation_depth_m"])
                series = row["series"].strip()
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid CSV value on line {line_number}: {error}") from error
            if not series:
                raise ValueError(f"empty series name on line {line_number}")
            if depth < 0:
                raise ValueError(f"negative depth on line {line_number}: {depth}")
            grouped[series].append((timestamp, depth))

    unknown = sorted(set(grouped) - set(SERIES_ORDER))
    if unknown:
        raise ValueError(f"CSV contains unknown series: {unknown}")
    missing_series = [name for name in SERIES_ORDER if name not in grouped]
    if missing_series:
        raise ValueError(f"CSV is missing plotted series: {missing_series}")

    results: dict[str, tuple[list[datetime], list[float]]] = {}
    for name in SERIES_ORDER:
        values = sorted(grouped[name], key=lambda item: item[0])
        results[name] = (
            [item[0] for item in values],
            [item[1] for item in values],
        )
    return results


def default_output_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.endswith("_data"):
        stem = stem[:-5]
    return csv_path.with_name(f"{stem}.png")


def plot_data(
    results: dict[str, tuple[list[datetime], list[float]]],
    output: Path,
    dpi: int,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    for name in SERIES_ORDER:
        times, depths = results[name]
        ax.plot(
            times,
            depths,
            label=name,
            color=SERIES_COLORS[name],
            linewidth=LINE_WIDTHS[name],
            zorder=9 if name == "upGPR" else 8 if name == "HydroFirn" else 2,
        )

    ax.legend(
        loc="center right",
        ncol=1,
        frameon=False,
        borderaxespad=0.6,
    )
    ax.set_ylabel("Percolation depth (m)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.set_xlim(X_START, X_END)
    ax.set_ylim(2.6, 0.0)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    output = output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_output, bbox_inches="tight")
    print(f"Saved {output}")
    print(f"Saved {pdf_output}")
    return output, pdf_output


def main() -> None:
    args = parse_args()
    csv_path = args.csv_file.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    output = args.output or default_output_path(csv_path)
    results = read_plot_data(csv_path)
    plot_data(results, output, args.dpi)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
