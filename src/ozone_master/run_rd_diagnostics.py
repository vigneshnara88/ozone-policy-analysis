from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

WIDTH = 920
HEIGHT = 540
MARGIN_LEFT = 82
MARGIN_RIGHT = 34
MARGIN_TOP = 54
MARGIN_BOTTOM = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RD plots and diagnostics around the 70 ppb ozone design-value cutoff."
    )
    parser.add_argument(
        "--analysis-file",
        default="data/processed/county_analysis_master.csv",
        help="County analysis master file",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rd",
        help="Directory for RD plots and summary files",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.070,
        help="Design-value cutoff in ppm",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.005,
        help="Bandwidth around the cutoff for RD fits",
    )
    parser.add_argument(
        "--plot-bandwidth",
        type=float,
        default=0.010,
        help="Bandwidth around the cutoff for plotted binned means",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.001,
        help="Bin width for RD and McCrary plots",
    )
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


def base_svg(title: str, subtitle: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">'
        "<style>"
        "text { font-family: Arial, sans-serif; fill: #111; }"
        ".title { font-size: 22px; font-weight: 700; }"
        ".subtitle { font-size: 13px; fill: #555; }"
        ".axis { stroke: #333; stroke-width: 1.2; }"
        ".grid { stroke: #d6d6d6; stroke-width: 1; }"
        ".tick { font-size: 12px; }"
        ".label { font-size: 15px; font-weight: 600; }"
        "</style>"
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white" />'
        f'<text x="{WIDTH / 2}" y="28" text-anchor="middle" class="title">{escape(title)}</text>'
        f'<text x="{WIDTH / 2}" y="46" text-anchor="middle" class="subtitle">{escape(subtitle)}</text>'
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
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def load_rd_sample(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if "county_design_value_ppm" not in df.columns:
        raise ValueError("Analysis file is missing county_design_value_ppm.")

    df = df[df["county_design_value_ppm"].notna()].copy()
    df["county_fips"] = df["county_fips"].astype("string").str.replace(r"\.0$", "", regex=True)
    df["running"] = df["county_design_value_ppm"] - 0.070
    df["treated_above_70"] = (df["county_design_value_ppm"] > 0.070).astype(int)

    state_summary = (
        df.groupby("state_name", as_index=False)
        .agg(
            total_counties=("county_fips", "size"),
            switched_counties=("newly_treated_70_rule", lambda s: int(pd.Series(s).fillna(0).sum())),
            below70_counties=("county_design_value_ppm", lambda s: int((s <= 0.070).sum())),
            above70_counties=("county_design_value_ppm", lambda s: int((s > 0.070).sum())),
        )
    )
    state_scope = state_summary[
        (state_summary["switched_counties"] > 0) & (state_summary["below70_counties"] > 0)
    ].copy()
    scoped = df[df["state_name"].isin(state_scope["state_name"])].copy()
    scoped["rd_scope_state"] = 1
    return scoped.reset_index(drop=True), state_scope.reset_index(drop=True)


def local_linear_rd(df: pd.DataFrame, outcome: str, bandwidth: float) -> dict[str, float]:
    sample = df[df["running"].between(-bandwidth, bandwidth) & df[outcome].notna()].copy()
    if sample.empty:
        return {"outcome": outcome, "n": 0, "left_n": 0, "right_n": 0, "tau": np.nan}

    x = sample["running"].to_numpy(dtype=float)
    y = sample[outcome].to_numpy(dtype=float)
    d = (x > 0).astype(float)
    weights = np.maximum(1 - np.abs(x) / bandwidth, 0)
    X = np.column_stack([np.ones(len(sample)), x, d, x * d])
    sw = np.sqrt(weights)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    tau = beta[2]
    return {
        "outcome": outcome,
        "n": int(len(sample)),
        "left_n": int((x <= 0).sum()),
        "right_n": int((x > 0).sum()),
        "tau": float(tau),
    }


def binned_means(
    df: pd.DataFrame,
    outcome: str,
    plot_bandwidth: float,
    bin_width: float,
) -> pd.DataFrame:
    sample = df[df["running"].between(-plot_bandwidth, plot_bandwidth) & df[outcome].notna()].copy()
    if sample.empty:
        return pd.DataFrame(columns=["bin_center", "y_mean", "n", "side", "outcome"])
    sample["bin_id"] = np.floor(sample["running"] / bin_width).astype(int)
    sample["bin_center"] = (sample["bin_id"] + 0.5) * bin_width
    out = (
        sample.groupby("bin_center", as_index=False)
        .agg(y_mean=(outcome, "mean"), n=(outcome, "size"))
        .sort_values("bin_center")
        .reset_index(drop=True)
    )
    out["side"] = np.where(out["bin_center"] <= 0, "left", "right")
    out["outcome"] = outcome
    return out


def rd_fit_line(df: pd.DataFrame, outcome: str, bandwidth: float, side: str) -> tuple[np.ndarray, np.ndarray]:
    sample = df[df["running"].between(-bandwidth, bandwidth) & df[outcome].notna()].copy()
    if side == "left":
        sample = sample[sample["running"] <= 0]
    else:
        sample = sample[sample["running"] > 0]
    if sample.empty:
        return np.array([]), np.array([])
    x = sample["running"].to_numpy(dtype=float)
    y = sample[outcome].to_numpy(dtype=float)
    weights = np.maximum(1 - np.abs(x) / bandwidth, 0)
    X = np.column_stack([np.ones(len(sample)), x])
    sw = np.sqrt(weights)
    beta, _, _, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    grid = np.linspace(x.min(), x.max(), 50)
    preds = beta[0] + beta[1] * grid
    return grid, preds


def build_rd_plot(
    binned: pd.DataFrame,
    fit_left: tuple[np.ndarray, np.ndarray],
    fit_right: tuple[np.ndarray, np.ndarray],
    title: str,
    subtitle: str,
    y_label: str,
) -> str:
    x_values = binned["bin_center"].tolist()
    y_values = binned["y_mean"].tolist()
    if fit_left[0].size:
        x_values += fit_left[0].tolist()
        y_values += fit_left[1].tolist()
    if fit_right[0].size:
        x_values += fit_right[0].tolist()
        y_values += fit_right[1].tolist()
    x_min, x_max = value_range(x_values, pad_ratio=0.0)
    y_min, y_max = value_range(y_values, pad_ratio=0.08)
    x_ticks = [(scale_x(v, x_min, x_max), format_tick(v)) for v in numeric_ticks(x_min, x_max)]
    y_ticks = [(scale_y(v, y_min, y_max), format_tick(v)) for v in numeric_ticks(y_min, y_max)]
    body = axes_and_grid(x_ticks, y_ticks, "Running Variable: county design value - 0.070", y_label)
    cutoff_x = scale_x(0, x_min, x_max)
    body += f'<line x1="{cutoff_x}" y1="{MARGIN_TOP}" x2="{cutoff_x}" y2="{HEIGHT - MARGIN_BOTTOM}" stroke="#111" stroke-dasharray="6,4" />'

    for side, color in [("left", "#2f6db2"), ("right", "#d1495b")]:
        side_data = binned[binned["side"] == side]
        for _, row in side_data.iterrows():
            radius = 3 + min(float(row["n"]) / 8.0, 6)
            body += (
                f'<circle cx="{scale_x(row["bin_center"], x_min, x_max)}" '
                f'cy="{scale_y(row["y_mean"], y_min, y_max)}" r="{radius:.2f}" fill="{color}" opacity="0.7" />'
            )

    for grid, preds, color in [
        (fit_left[0], fit_left[1], "#2f6db2"),
        (fit_right[0], fit_right[1], "#d1495b"),
    ]:
        if len(grid):
            points = " ".join(
                f"{scale_x(x, x_min, x_max)},{scale_y(y, y_min, y_max)}"
                for x, y in zip(grid, preds)
            )
            body += f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />'
    return base_svg(title, subtitle, body)


def build_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, subtitle: str, y_label: str) -> str:
    clean = df[[x_col, y_col]].dropna().copy()
    clean = clean.sort_values(y_col, ascending=False).head(20).iloc[::-1]
    x_values = clean[x_col].astype(str).tolist()
    y_values = clean[y_col].astype(float).tolist()
    y_min, y_max = 0.0, max(y_values) if y_values else 1.0
    y_max *= 1.15
    y_ticks = [(scale_y(v, y_min, y_max), format_tick(v)) for v in numeric_ticks(y_min, y_max)]
    x_ticks = []
    plot_width, _ = plot_area()
    bar_span = plot_width / max(len(clean), 1)
    body = axes_and_grid(x_ticks, y_ticks, "", y_label)
    for idx, (label, value) in enumerate(zip(x_values, y_values)):
        x = MARGIN_LEFT + idx * bar_span + bar_span * 0.1
        width = bar_span * 0.8
        y = scale_y(value, y_min, y_max)
        bottom = scale_y(0, y_min, y_max)
        body += f'<rect x="{x}" y="{y}" width="{width}" height="{bottom - y}" fill="#2f6db2" />'
        body += f'<text x="{x + width / 2}" y="{bottom + 14}" text-anchor="end" transform="rotate(-45 {x + width / 2} {bottom + 14})" class="tick">{escape(label)}</text>'
    return base_svg(title, subtitle, body)


def build_histogram(values: pd.Series, title: str, subtitle: str, x_label: str, y_label: str, bin_width: float, cutoff: float) -> str:
    clean = values.dropna().astype(float).tolist()
    x_min, x_max = value_range(clean, pad_ratio=0.0)
    start = np.floor(x_min / bin_width) * bin_width
    end = np.ceil(x_max / bin_width) * bin_width
    bins = np.arange(start, end + bin_width, bin_width)
    counts, edges = np.histogram(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    y_min, y_max = 0.0, max(counts) if len(counts) else 1.0
    y_max *= 1.15
    x_ticks = [(scale_x(v, start, end), format_tick(v)) for v in numeric_ticks(start, end)]
    y_ticks = [(scale_y(v, y_min, y_max), format_tick(v)) for v in numeric_ticks(y_min, y_max)]
    body = axes_and_grid(x_ticks, y_ticks, x_label, y_label)
    plot_width, _ = plot_area()
    bar_width = plot_width / max(len(counts), 1) * 0.9
    for center, count in zip(centers, counts):
        left = scale_x(center, start, end) - bar_width / 2
        top = scale_y(count, y_min, y_max)
        bottom = scale_y(0, y_min, y_max)
        body += f'<rect x="{left}" y="{top}" width="{bar_width}" height="{bottom - top}" fill="#7aa6c2" opacity="0.9" />'
    cutoff_x = scale_x(cutoff, start, end)
    body += f'<line x1="{cutoff_x}" y1="{MARGIN_TOP}" x2="{cutoff_x}" y2="{HEIGHT - MARGIN_BOTTOM}" stroke="#111" stroke-dasharray="6,4" />'
    return base_svg(title, subtitle, body)


def compute_mccrary(df: pd.DataFrame, plot_bandwidth: float, bin_width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = df[df["running"].between(-plot_bandwidth, plot_bandwidth)].copy()
    sample["bin_id"] = np.floor(sample["running"] / bin_width).astype(int)
    sample["bin_center"] = (sample["bin_id"] + 0.5) * bin_width
    counts = sample.groupby("bin_center", as_index=False).size().rename(columns={"size": "count"})
    counts = counts.sort_values("bin_center").reset_index(drop=True)
    counts["side"] = np.where(counts["bin_center"] <= 0, "left", "right")

    near = counts[(counts["bin_center"].abs() <= plot_bandwidth) & (counts["count"] > 0)].copy()
    left = near[near["bin_center"] < 0]["count"]
    right = near[near["bin_center"] > 0]["count"]
    left_mean = left.mean() if not left.empty else np.nan
    right_mean = right.mean() if not right.empty else np.nan
    log_diff = np.log(right_mean) - np.log(left_mean) if left_mean and right_mean else np.nan
    summary = pd.DataFrame(
        [{
            "left_mean_bin_count": left_mean,
            "right_mean_bin_count": right_mean,
            "log_density_jump": log_diff,
            "bandwidth": plot_bandwidth,
            "bin_width": bin_width,
            "note": "Approximate McCrary-style diagnostic using mean histogram counts by side of cutoff.",
        }]
    )
    return counts, summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    rd_df, state_scope = load_rd_sample(Path(args.analysis_file))
    rd_df.to_csv(output_dir / "rd_sample.csv", index=False)
    state_scope.to_csv(output_dir / "rd_state_scope.csv", index=False)

    # Binned RD plots and estimates.
    outcomes = [
        ("pre_ozone_mean_ppm", "Pre-period Ozone RD", "Pre-period mean ozone (ppm)"),
        ("post_ozone_mean_ppm", "Post-period Ozone RD", "Post-period mean ozone (ppm)"),
        ("delta_ozone_mean_ppm", "Change in Ozone RD", "Post minus pre ozone (ppm)"),
    ]
    binned_frames: list[pd.DataFrame] = []
    estimate_rows: list[dict[str, float]] = []
    for outcome, title, y_label in outcomes:
        binned = binned_means(rd_df, outcome, args.plot_bandwidth, args.bin_width)
        binned_frames.append(binned)
        estimate_rows.append(local_linear_rd(rd_df, outcome, args.bandwidth))
        fit_left = rd_fit_line(rd_df, outcome, args.bandwidth, "left")
        fit_right = rd_fit_line(rd_df, outcome, args.bandwidth, "right")
        write_svg(
            output_dir / f"{outcome}_rd.svg",
            build_rd_plot(
                binned,
                fit_left,
                fit_right,
                title,
                "Sample scoped to states with at least one switcher and at least one below-70 county.",
                y_label,
            ),
        )

    pd.concat(binned_frames, ignore_index=True).to_csv(output_dir / "rd_binned_means.csv", index=False)
    pd.DataFrame(estimate_rows).to_csv(output_dir / "rd_estimates_summary.csv", index=False)

    # EDA.
    switched_counts = (
        rd_df.groupby("state_name", as_index=False)
        .agg(switched_counties=("newly_treated_70_rule", lambda s: int(pd.Series(s).fillna(0).sum())))
        .sort_values("switched_counties", ascending=False)
        .reset_index(drop=True)
    )
    switched_counts.to_csv(output_dir / "switched_counties_by_state.csv", index=False)
    write_svg(
        output_dir / "switched_counties_by_state.svg",
        build_bar_chart(
            switched_counts[switched_counts["switched_counties"] > 0],
            "state_name",
            "switched_counties",
            "Count of Counties Newly Captured by 70 ppb Rule",
            "States in RD scope only.",
            "Switched counties",
        ),
    )
    write_svg(
        output_dir / "design_value_histogram.svg",
        build_histogram(
            rd_df["county_design_value_ppm"],
            "County Design Value Distribution",
            "Distribution for states in RD scope.",
            "County design value (ppm)",
            "County count",
            args.bin_width,
            args.cutoff,
        ),
    )

    # Covariate balance.
    balance_covariates = [
        ("population_2010", "Population 2010 RD Balance", "Population (2010)"),
        ("pre_avg_monitor_count", "Pre-period Monitor Count RD Balance", "Average monitor count"),
        ("pre_ozone_days_observed", "Pre-period Ozone Days Observed RD Balance", "Observed days"),
    ]
    balance_rows: list[dict[str, float]] = []
    for covariate, title, y_label in balance_covariates:
        if covariate not in rd_df.columns:
            continue
        balance_rows.append(local_linear_rd(rd_df, covariate, args.bandwidth))
        binned = binned_means(rd_df, covariate, args.plot_bandwidth, args.bin_width)
        fit_left = rd_fit_line(rd_df, covariate, args.bandwidth, "left")
        fit_right = rd_fit_line(rd_df, covariate, args.bandwidth, "right")
        write_svg(
            output_dir / f"{covariate}_balance.svg",
            build_rd_plot(
                binned,
                fit_left,
                fit_right,
                title,
                "Continuity check around the 70 ppb cutoff.",
                y_label,
            ),
        )
    pd.DataFrame(balance_rows).to_csv(output_dir / "covariate_balance_summary.csv", index=False)

    # McCrary-style density diagnostic.
    mccrary_bins, mccrary_summary = compute_mccrary(rd_df, args.plot_bandwidth, args.bin_width)
    mccrary_bins.to_csv(output_dir / "mccrary_bins.csv", index=False)
    mccrary_summary.to_csv(output_dir / "mccrary_summary.csv", index=False)
    write_svg(
        output_dir / "mccrary_density.svg",
        build_histogram(
            rd_df["running"],
            "McCrary-style Density Diagnostic",
            "Approximate histogram-based density check around running-variable cutoff 0.",
            "Running variable: county design value - 0.070",
            "County count",
            args.bin_width,
            0.0,
        ),
    )

    print(f"Wrote RD diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
