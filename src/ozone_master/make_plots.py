from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

WIDTH = 900
HEIGHT = 520
MARGIN_LEFT = 80
MARGIN_RIGHT = 30
MARGIN_TOP = 50
MARGIN_BOTTOM = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory ozone plots as SVG files.")
    parser.add_argument(
        "--county-year-file",
        default="data/processed/county_year_master.csv",
        help="County-year master dataset",
    )
    parser.add_argument(
        "--county-analysis-file",
        default="data/processed/county_analysis_master.csv",
        help="County analysis dataset",
    )
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory for plot files")
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_area() -> tuple[float, float]:
    return WIDTH - MARGIN_LEFT - MARGIN_RIGHT, HEIGHT - MARGIN_TOP - MARGIN_BOTTOM


def value_range(values: list[float], pad_ratio: float = 0.05) -> tuple[float, float]:
    clean = [float(value) for value in values if pd.notna(value)]
    if not clean:
        return 0.0, 1.0
    low, high = min(clean), max(clean)
    if low == high:
        delta = abs(low) * 0.05 if low else 1.0
        return low - delta, high + delta
    pad = (high - low) * pad_ratio
    return low - pad, high + pad


def scale_x(value: float, x_min: float, x_max: float) -> float:
    plot_width, _ = plot_area()
    if x_max == x_min:
        return MARGIN_LEFT + plot_width / 2
    return MARGIN_LEFT + (value - x_min) / (x_max - x_min) * plot_width


def scale_y(value: float, y_min: float, y_max: float) -> float:
    _, plot_height = plot_area()
    if y_max == y_min:
        return MARGIN_TOP + plot_height / 2
    return MARGIN_TOP + plot_height - (value - y_min) / (y_max - y_min) * plot_height


def base_svg(title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">'
        '<style>'
        'text { font-family: Arial, sans-serif; fill: #111; }'
        '.title { font-size: 22px; font-weight: 700; }'
        '.axis { stroke: #333; stroke-width: 1.2; }'
        '.grid { stroke: #d6d6d6; stroke-width: 1; }'
        '.tick { font-size: 12px; }'
        '.label { font-size: 15px; font-weight: 600; }'
        '</style>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white" />'
        f'<text x="{WIDTH / 2}" y="30" text-anchor="middle" class="title">{escape(title)}</text>'
        f"{body}</svg>"
    )


def axes_and_grid(
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    x_label: str,
    y_label: str,
) -> str:
    plot_width, plot_height = plot_area()
    x0, y0 = MARGIN_LEFT, MARGIN_TOP + plot_height
    x1, y1 = MARGIN_LEFT + plot_width, MARGIN_TOP

    parts = [
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis" />',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis" />',
    ]

    for x, label in x_ticks:
        parts.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" class="grid" />')
        parts.append(
            f'<text x="{x}" y="{y0 + 22}" text-anchor="middle" class="tick">{escape(label)}</text>'
        )

    for y, label in y_ticks:
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="grid" />')
        parts.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" class="tick">{escape(label)}</text>'
        )

    parts.append(
        f'<text x="{MARGIN_LEFT + plot_width / 2}" y="{HEIGHT - 20}" text-anchor="middle" class="label">{escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="22" y="{MARGIN_TOP + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {MARGIN_TOP + plot_height / 2})" class="label">{escape(y_label)}</text>'
    )
    return "".join(parts)


def numeric_ticks(v_min: float, v_max: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [v_min]
    step = (v_max - v_min) / (count - 1)
    return [v_min + index * step for index in range(count)]


def format_tick(value: float) -> str:
    return f"{value:.3f}" if abs(value) < 1 else f"{value:.1f}"


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def pick_treatment_column(df: pd.DataFrame) -> str | None:
    for column in ["nonattainment", "treatment", "attainment_status"]:
        if column in df.columns:
            return column
    return None


def pick_running_variable(df: pd.DataFrame) -> str | None:
    for column in ["distance_from_threshold", "design_value_2016", "running_variable", "design_value"]:
        if column in df.columns:
            return column
    return None


def pick_health_delta_column(df: pd.DataFrame) -> str | None:
    candidates = [column for column in df.columns if column.startswith("delta_") and "ozone_" not in column]
    return candidates[0] if candidates else None


def build_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
) -> str:
    clean = df[[x_col, y_col]].dropna().sort_values(x_col)
    x_values = clean[x_col].tolist()
    y_values = clean[y_col].tolist()
    x_min, x_max = value_range(x_values, 0.0)
    y_min, y_max = value_range(y_values)

    x_ticks = [
        (scale_x(value, x_min, x_max), str(int(value)) if float(value).is_integer() else format_tick(value))
        for value in x_values
    ]
    y_tick_values = numeric_ticks(y_min, y_max)
    y_ticks = [(scale_y(value, y_min, y_max), format_tick(value)) for value in y_tick_values]

    points = [
        f"{scale_x(x, x_min, x_max)},{scale_y(y, y_min, y_max)}" for x, y in zip(x_values, y_values)
    ]
    circles = "".join(
        f'<circle cx="{scale_x(x, x_min, x_max)}" cy="{scale_y(y, y_min, y_max)}" r="4" fill="{color}" />'
        for x, y in zip(x_values, y_values)
    )
    body = (
        axes_and_grid(x_ticks, y_ticks, x_label, y_label)
        + f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}" />'
        + circles
    )
    return base_svg(title, body)


def build_histogram(values: list[float], title: str, x_label: str, y_label: str, bins: int = 20) -> str:
    clean = [float(value) for value in values if pd.notna(value)]
    if not clean:
        clean = [0.0]
    x_min, x_max = value_range(clean, 0.0)
    if x_min == x_max:
        x_max = x_min + 1.0
    bin_width = (x_max - x_min) / bins
    counts = [0 for _ in range(bins)]
    for value in clean:
        index = min(int((value - x_min) / bin_width), bins - 1)
        counts[index] += 1
    y_min, y_max = 0.0, max(counts) if max(counts) else 1.0

    x_tick_values = numeric_ticks(x_min, x_max)
    x_ticks = [(scale_x(value, x_min, x_max), format_tick(value)) for value in x_tick_values]
    y_tick_values = numeric_ticks(y_min, y_max)
    y_ticks = [(scale_y(value, y_min, y_max), format_tick(value)) for value in y_tick_values]

    plot_width, _ = plot_area()
    bar_width = plot_width / bins * 0.85
    bars = []
    for index, count in enumerate(counts):
        left_value = x_min + index * bin_width
        left = scale_x(left_value, x_min, x_max)
        top = scale_y(count, y_min, y_max)
        bottom = scale_y(0, y_min, y_max)
        bars.append(
            f'<rect x="{left}" y="{top}" width="{bar_width}" height="{bottom - top}" fill="#2f6db2" opacity="0.85" />'
        )

    zero_x = scale_x(0, x_min, x_max)
    body = axes_and_grid(x_ticks, y_ticks, x_label, y_label)
    body += f'<line x1="{zero_x}" y1="{MARGIN_TOP}" x2="{zero_x}" y2="{HEIGHT - MARGIN_BOTTOM}" stroke="#111" stroke-dasharray="5,5" />'
    body += "".join(bars)
    return base_svg(title, body)


def build_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    group_col: str | None = None,
) -> str:
    clean = df[[column for column in [x_col, y_col, group_col] if column]].dropna(subset=[x_col, y_col])
    x_values = clean[x_col].tolist()
    y_values = clean[y_col].tolist()
    x_min, x_max = value_range(x_values)
    y_min, y_max = value_range(y_values)

    x_tick_values = numeric_ticks(x_min, x_max)
    y_tick_values = numeric_ticks(y_min, y_max)
    x_ticks = [(scale_x(value, x_min, x_max), format_tick(value)) for value in x_tick_values]
    y_ticks = [(scale_y(value, y_min, y_max), format_tick(value)) for value in y_tick_values]

    colors = ["#2f6db2", "#d1495b", "#2b9348", "#9c6644", "#7b2cbf"]
    body = axes_and_grid(x_ticks, y_ticks, x_label, y_label)

    if group_col:
        legend_parts = []
        for index, (label, subset) in enumerate(clean.groupby(group_col)):
            color = colors[index % len(colors)]
            for _, row in subset.iterrows():
                body += (
                    f'<circle cx="{scale_x(row[x_col], x_min, x_max)}" '
                    f'cy="{scale_y(row[y_col], y_min, y_max)}" r="4" fill="{color}" opacity="0.75" />'
                )
            legend_y = 55 + index * 20
            legend_parts.append(f'<circle cx="{WIDTH - 180}" cy="{legend_y}" r="5" fill="{color}" />')
            legend_parts.append(
                f'<text x="{WIDTH - 168}" y="{legend_y + 4}" class="tick">{escape(str(label))}</text>'
            )
        body += "".join(legend_parts)
    else:
        for _, row in clean.iterrows():
            body += (
                f'<circle cx="{scale_x(row[x_col], x_min, x_max)}" '
                f'cy="{scale_y(row[y_col], y_min, y_max)}" r="4" fill="#2f6db2" opacity="0.75" />'
            )

    return base_svg(title, body)


def build_pre_post_scatter(analysis: pd.DataFrame) -> str:
    return build_scatter(
        analysis,
        x_col="pre_ozone_mean_ppm",
        y_col="post_ozone_mean_ppm",
        title="Pre vs Post County Ozone",
        x_label="Pre-period mean ozone (ppm)",
        y_label="Post-period mean ozone (ppm)",
        group_col=pick_treatment_column(analysis),
    )


def build_running_scatter(analysis: pd.DataFrame, running_col: str) -> str:
    return build_scatter(
        analysis,
        x_col=running_col,
        y_col="delta_ozone_mean_ppm",
        title="Change in Ozone vs Running Variable",
        x_label=running_col,
        y_label="Post minus pre ozone mean (ppm)",
    )


def build_health_scatter(analysis: pd.DataFrame, health_col: str) -> str:
    return build_scatter(
        analysis,
        x_col="delta_ozone_mean_ppm",
        y_col=health_col,
        title="Health Change vs Ozone Change",
        x_label="Post minus pre ozone mean (ppm)",
        y_label=health_col,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    county_year = pd.read_csv(args.county_year_file)
    analysis = pd.read_csv(args.county_analysis_file)

    yearly = (
        county_year.groupby("year", as_index=False)
        .agg(
            mean_ozone_ppm=("ozone_mean_ppm", "mean"),
            mean_exceed_days=("ozone_exceed_days", "mean"),
        )
        .sort_values("year")
    )

    write_svg(
        output_dir / "01_yearly_mean_ozone.svg",
        build_line_chart(
            yearly,
            x_col="year",
            y_col="mean_ozone_ppm",
            title="Average County Ozone by Year",
            x_label="Year",
            y_label="Ozone (ppm)",
            color="#2f6db2",
        ),
    )
    write_svg(
        output_dir / "02_yearly_exceed_days.svg",
        build_line_chart(
            yearly,
            x_col="year",
            y_col="mean_exceed_days",
            title="Average County Exceedance Days by Year",
            x_label="Year",
            y_label="Days Above Threshold",
            color="#d1495b",
        ),
    )
    write_svg(output_dir / "03_pre_post_ozone_scatter.svg", build_pre_post_scatter(analysis))
    write_svg(
        output_dir / "04_delta_ozone_histogram.svg",
        build_histogram(
            analysis["delta_ozone_mean_ppm"].tolist(),
            title="Distribution of Change in County Ozone",
            x_label="Post minus pre ozone mean (ppm)",
            y_label="County count",
        ),
    )

    running_col = pick_running_variable(analysis)
    if running_col:
        write_svg(
            output_dir / "05_running_variable_vs_delta_ozone.svg",
            build_running_scatter(analysis, running_col),
        )

    health_col = pick_health_delta_column(analysis)
    if health_col:
        write_svg(
            output_dir / "06_health_vs_delta_ozone.svg",
            build_health_scatter(analysis, health_col),
        )

    print(f"Wrote plot files to {output_dir}")


if __name__ == "__main__":
    main()
