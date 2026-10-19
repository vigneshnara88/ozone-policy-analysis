from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OZONE_PARAMETER_CODES = {"44201"}
OZONE_PARAMETER_NAMES = {"ozone"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build county-year ozone master datasets with optional health merges."
    )
    parser.add_argument("--aqs-dir", default="data/raw/aqs", help="Directory of daily AQS CSV files")
    parser.add_argument(
        "--health-file",
        default="data/raw/health/health_county_year.csv",
        help="Optional county-year or county-date health file",
    )
    parser.add_argument(
        "--designations-file",
        default="data/raw/designations/county_designations.csv",
        help="Optional county-level treatment/designation file",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional first calendar year to keep in the ozone dataset",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional last calendar year to keep in the ozone dataset",
    )
    parser.add_argument(
        "--ozone-threshold-ppm",
        type=float,
        default=0.070,
        help="Daily ozone exceedance threshold in ppm",
    )
    parser.add_argument(
        "--pre-years", nargs="+", type=int, default=[2014, 2015, 2016], help="Pre-period ozone years"
    )
    parser.add_argument(
        "--post-years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Post-period ozone years",
    )
    parser.add_argument(
        "--health-pre-years",
        nargs="+",
        type=int,
        default=[2015, 2016, 2017],
        help="Pre-period health years",
    )
    parser.add_argument(
        "--health-post-years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Post-period health years",
    )
    parser.add_argument(
        "--health-agg",
        choices=["auto", "sum", "mean"],
        default="auto",
        help="Aggregation rule if the health file is daily rather than yearly",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_folder(folder: Path) -> pd.DataFrame:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = [pd.read_csv(file, low_memory=False) for file in files]
    return pd.concat(frames, ignore_index=True)


def standardize_fips(series: pd.Series, width: int) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(width)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    normalized_columns = [
        column.strip().lower().replace(" ", "_").replace("-", "_")
        for column in renamed.columns
    ]
    renamed.columns = normalized_columns

    if renamed.columns.is_unique:
        return renamed

    collapsed_columns: dict[str, pd.Series] = {}
    ordered_names: list[str] = []
    for column in renamed.columns.unique():
        selected = renamed.loc[:, renamed.columns == column]
        if isinstance(selected, pd.Series):
            collapsed_columns[column] = selected
        else:
            collapsed_columns[column] = selected.bfill(axis=1).iloc[:, 0]
        ordered_names.append(column)

    return pd.DataFrame(collapsed_columns)[ordered_names]


def available_column_mapping(path: Path) -> dict[str, str]:
    header = pd.read_csv(path, nrows=0)
    mapping = {}
    for column in header.columns:
        normalized = column.strip().lower().replace(" ", "_").replace("-", "_")
        mapping[normalized] = column
    return mapping


def combine_grouped_frames(frames: list[pd.DataFrame], group_cols: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=group_cols)
    combined = pd.concat(frames, ignore_index=True)
    aggregations = {
        column: "sum" for column in combined.columns if column not in set(group_cols)
    }
    return combined.groupby(group_cols, as_index=False).agg(aggregations)


def load_daily_ozone(
    aqs_dir: Path,
    threshold_ppm: float,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    if not aqs_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {aqs_dir}")

    files = sorted(aqs_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {aqs_dir}")

    monitor_frames: list[pd.DataFrame] = []
    saw_ozone = False

    for file in files:
        column_map = available_column_mapping(file)
        required = ["state_code", "county_code", "site_num", "date_local", "arithmetic_mean"]
        missing = [column for column in required if column not in column_map]
        if missing:
            continue

        optional = [column for column in ["parameter_code", "parameter_name"] if column in column_map]
        usecols = [column_map[column] for column in [*required, *optional]]

        for chunk in pd.read_csv(file, usecols=usecols, chunksize=250_000, low_memory=False):
            chunk = standardize_columns(chunk)

            parameter_code = chunk.get(
                "parameter_code", pd.Series(index=chunk.index, dtype="string")
            ).astype("string")
            parameter_name = chunk.get(
                "parameter_name", pd.Series(index=chunk.index, dtype="string")
            ).astype("string")

            if "parameter_code" in chunk.columns or "parameter_name" in chunk.columns:
                ozone_mask = parameter_code.isin(OZONE_PARAMETER_CODES) | parameter_name.str.lower().isin(
                    OZONE_PARAMETER_NAMES
                )
                chunk = chunk.loc[ozone_mask].copy()

            if chunk.empty:
                continue

            saw_ozone = True
            chunk["county_fips"] = standardize_fips(chunk["state_code"], 2) + standardize_fips(
                chunk["county_code"], 3
            )
            chunk["monitor_id"] = (
                standardize_fips(chunk["state_code"], 2)
                + standardize_fips(chunk["county_code"], 3)
                + standardize_fips(chunk["site_num"], 4)
            )
            chunk["date"] = pd.to_datetime(chunk["date_local"], errors="coerce").dt.normalize()
            chunk["ozone_ppm"] = pd.to_numeric(chunk["arithmetic_mean"], errors="coerce")
            chunk = chunk.dropna(subset=["county_fips", "monitor_id", "date", "ozone_ppm"])
            if chunk.empty:
                continue

            grouped = (
                chunk.groupby(["county_fips", "monitor_id", "date"], as_index=False)
                .agg(ozone_sum=("ozone_ppm", "sum"), ozone_count=("ozone_ppm", "size"))
            )
            monitor_frames.append(grouped)

    if not saw_ozone or not monitor_frames:
        raise ValueError("No ozone observations found in the supplied AQS files.")

    monitor_day = combine_grouped_frames(monitor_frames, ["county_fips", "monitor_id", "date"])
    monitor_day["ozone_ppm"] = monitor_day["ozone_sum"] / monitor_day["ozone_count"]

    county_day = (
        monitor_day.groupby(["county_fips", "date"], as_index=False)
        .agg(
            ozone_ppm=("ozone_ppm", "mean"),
            monitor_count=("monitor_id", "nunique"),
        )
        .sort_values(["county_fips", "date"])
        .reset_index(drop=True)
    )
    county_day["year"] = county_day["date"].dt.year
    if start_year is not None:
        county_day = county_day.loc[county_day["year"] >= start_year].copy()
    if end_year is not None:
        county_day = county_day.loc[county_day["year"] <= end_year].copy()
    if county_day.empty:
        raise ValueError(
            "No ozone observations remain after applying the requested year filter."
        )
    county_day["ozone_exceed_day"] = (county_day["ozone_ppm"] > threshold_ppm).astype(int)
    return county_day


def build_county_year_ozone(county_day: pd.DataFrame, threshold_ppm: float) -> pd.DataFrame:
    county_year = (
        county_day.groupby(["county_fips", "year"], as_index=False)
        .agg(
            ozone_mean_ppm=("ozone_ppm", "mean"),
            ozone_median_ppm=("ozone_ppm", "median"),
            ozone_p90_ppm=("ozone_ppm", lambda x: x.quantile(0.9)),
            ozone_max_ppm=("ozone_ppm", "max"),
            ozone_days_observed=("ozone_ppm", "size"),
            ozone_exceed_days=("ozone_exceed_day", "sum"),
            avg_monitor_count=("monitor_count", "mean"),
        )
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )
    county_year["ozone_exceed_share"] = (
        county_year["ozone_exceed_days"] / county_year["ozone_days_observed"]
    )
    county_year["ozone_threshold_ppm"] = threshold_ppm
    return county_year


def infer_health_aggregation(numeric_columns: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    if any("rate" in column or "pct" in column or "share" in column for column in numeric_columns):
        return "mean"
    return "sum"


def load_health(path: Path, requested_agg: str) -> pd.DataFrame | None:
    if not path.exists():
        return None

    health = standardize_columns(pd.read_csv(path))
    if "county_fips" not in health.columns:
        raise ValueError("Health file must contain `county_fips`.")

    health["county_fips"] = standardize_fips(health["county_fips"], 5)

    if "year" not in health.columns:
        if "date" not in health.columns:
            raise ValueError("Health file must contain either `year` or `date`.")
        health["year"] = pd.to_datetime(health["date"], errors="coerce").dt.year

    numeric_columns = [
        column
        for column in health.columns
        if column not in {"county_fips", "year", "date"} and pd.api.types.is_numeric_dtype(health[column])
    ]
    if not numeric_columns:
        raise ValueError("Health file does not contain any numeric outcome columns.")

    yearly = health[["county_fips", "year", *numeric_columns]].copy()
    yearly = yearly.dropna(subset=["county_fips", "year"])

    aggregation = infer_health_aggregation(numeric_columns, requested_agg)
    agg_func = "mean" if aggregation == "mean" else "sum"
    return (
        yearly.groupby(["county_fips", "year"], as_index=False)[numeric_columns]
        .agg(agg_func)
        .sort_values(["county_fips", "year"])
        .reset_index(drop=True)
    )


def load_designations(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    designations = standardize_columns(pd.read_csv(path))
    if "county_fips" not in designations.columns:
        raise ValueError("Designations file must contain `county_fips`.")
    designations["county_fips"] = standardize_fips(designations["county_fips"], 5)
    return designations.drop_duplicates(subset=["county_fips"]).reset_index(drop=True)


def build_period_change(
    df: pd.DataFrame,
    years: list[int],
    prefix: str,
    value_columns: list[str],
) -> pd.DataFrame:
    subset = df.loc[df["year"].isin(years), ["county_fips", "year", *value_columns]].copy()
    grouped = subset.groupby("county_fips", as_index=False)[value_columns].mean()
    return grouped.rename(columns={column: f"{prefix}_{column}" for column in value_columns})


def build_analysis_master(
    county_year: pd.DataFrame,
    designations: pd.DataFrame | None,
    pre_years: list[int],
    post_years: list[int],
    health_pre_years: list[int],
    health_post_years: list[int],
) -> pd.DataFrame:
    ozone_columns = [
        "ozone_mean_ppm",
        "ozone_median_ppm",
        "ozone_p90_ppm",
        "ozone_max_ppm",
        "ozone_days_observed",
        "ozone_exceed_days",
        "ozone_exceed_share",
        "avg_monitor_count",
    ]
    health_columns = [
        column
        for column in county_year.columns
        if column not in {"county_fips", "year", *ozone_columns, "ozone_threshold_ppm"}
        and pd.api.types.is_numeric_dtype(county_year[column])
    ]

    ozone_pre = build_period_change(county_year, pre_years, "pre", ozone_columns)
    ozone_post = build_period_change(county_year, post_years, "post", ozone_columns)
    analysis = ozone_pre.merge(ozone_post, on="county_fips", how="outer")

    for column in ozone_columns:
        analysis[f"delta_{column}"] = (
            analysis[f"post_{column}"] - analysis[f"pre_{column}"]
        )

    if health_columns:
        health_pre = build_period_change(county_year, health_pre_years, "health_pre", health_columns)
        health_post = build_period_change(
            county_year, health_post_years, "health_post", health_columns
        )
        analysis = analysis.merge(health_pre, on="county_fips", how="left")
        analysis = analysis.merge(health_post, on="county_fips", how="left")
        for column in health_columns:
            analysis[f"delta_{column}"] = (
                analysis[f"health_post_{column}"] - analysis[f"health_pre_{column}"]
            )

    analysis["ozone_pre_period"] = ",".join(str(year) for year in pre_years)
    analysis["ozone_post_period"] = ",".join(str(year) for year in post_years)
    analysis["health_pre_period"] = ",".join(str(year) for year in health_pre_years)
    analysis["health_post_period"] = ",".join(str(year) for year in health_post_years)

    if designations is not None:
        analysis = analysis.merge(designations, on="county_fips", how="left")

    return analysis.sort_values("county_fips").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    county_day = load_daily_ozone(
        Path(args.aqs_dir),
        args.ozone_threshold_ppm,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    county_year = build_county_year_ozone(county_day, args.ozone_threshold_ppm)

    health = load_health(Path(args.health_file), args.health_agg)
    if health is not None:
        county_year = county_year.merge(health, on=["county_fips", "year"], how="left")

    designations = load_designations(Path(args.designations_file))
    analysis = build_analysis_master(
        county_year=county_year,
        designations=designations,
        pre_years=args.pre_years,
        post_years=args.post_years,
        health_pre_years=args.health_pre_years,
        health_post_years=args.health_post_years,
    )

    county_day_path = output_dir / "county_day_ozone.csv"
    county_year_path = output_dir / "county_year_master.csv"
    analysis_path = output_dir / "county_analysis_master.csv"

    county_day.to_csv(county_day_path, index=False)
    county_year.to_csv(county_year_path, index=False)
    analysis.to_csv(analysis_path, index=False)

    print(f"Wrote {len(county_day):,} rows to {county_day_path}")
    print(f"Wrote {len(county_year):,} rows to {county_year_path}")
    print(f"Wrote {len(analysis):,} rows to {analysis_path}")


if __name__ == "__main__":
    main()
