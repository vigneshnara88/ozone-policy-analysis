from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .igra import build_igra_daily_files
from .io import ensure_directory, read_csv_folder, read_required_csv

POLLUTANT_CODE_MAP = {
    "88101": "pm25",
    "81102": "pm10",
    "44201": "ozone",
    "42602": "no2",
    "42401": "so2",
}

POLLUTANT_NAME_MAP = {
    "pm2.5 local conditions": "pm25",
    "pm10 total 0-10um stp": "pm10",
    "ozone": "ozone",
    "nitrogen dioxide (no2)": "no2",
    "sulfur dioxide": "so2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monitor-day and county-day master datasets."
    )
    parser.add_argument("--data-dir", default="data/raw", help="Root input data directory")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Output directory for merged files"
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=150.0,
        help="Maximum distance allowed between AQS monitor and IGRA station",
    )
    parser.add_argument(
        "--igra-lower-hpa",
        type=int,
        default=925,
        help="Lower pressure level used for inversion detection when parsing raw IGRA soundings",
    )
    parser.add_argument(
        "--igra-upper-hpa",
        type=int,
        default=850,
        help="Upper pressure level used for inversion detection when parsing raw IGRA soundings",
    )
    parser.add_argument(
        "--igra-pressure-tolerance-hpa",
        type=int,
        default=15,
        help="Tolerance used when matching IGRA pressure levels during raw parsing",
    )
    return parser.parse_args()


def standardize_fips(series: pd.Series, width: int) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(width)


def standardize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def build_monitor_id(df: pd.DataFrame) -> pd.Series:
    state = standardize_fips(df["state_code"], 2)
    county = standardize_fips(df["county_code"], 3)
    site = standardize_fips(df["site_num"], 4)
    return state + county + site


def infer_pollutant_label(df: pd.DataFrame) -> pd.Series:
    code_series = df.get("parameter_code")
    if code_series is not None:
        code_series = code_series.astype("string")
        from_code = code_series.map(POLLUTANT_CODE_MAP)
    else:
        from_code = pd.Series(index=df.index, dtype="string")

    name_series = df.get("parameter_name")
    if name_series is not None:
        from_name = name_series.astype("string").str.lower().map(POLLUTANT_NAME_MAP)
    else:
        from_name = pd.Series(index=df.index, dtype="string")

    return from_code.fillna(from_name)


def load_aqs(aqs_dir: Path) -> pd.DataFrame:
    raw = read_csv_folder(aqs_dir)
    raw.columns = [column.strip().lower() for column in raw.columns]

    required = [
        "state_code",
        "county_code",
        "site_num",
        "latitude",
        "longitude",
        "date_local",
        "arithmetic_mean",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"AQS files are missing required columns: {missing}")

    df = raw.copy()
    df["monitor_id"] = build_monitor_id(df)
    df["county_fips"] = (
        standardize_fips(df["state_code"], 2) + standardize_fips(df["county_code"], 3)
    )
    df["date"] = standardize_date(df["date_local"])
    df["pollutant"] = infer_pollutant_label(df)
    df["value"] = pd.to_numeric(df["arithmetic_mean"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(
        subset=["monitor_id", "county_fips", "date", "latitude", "longitude", "pollutant", "value"]
    )

    pivoted = (
        df.pivot_table(
            index=["monitor_id", "county_fips", "date", "latitude", "longitude"],
            columns="pollutant",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    return pivoted


def load_igra_stations(path: Path) -> pd.DataFrame:
    stations = read_required_csv(path)
    stations.columns = [column.strip().lower() for column in stations.columns]
    required = ["station_id", "latitude", "longitude"]
    missing = [column for column in required if column not in stations.columns]
    if missing:
        raise ValueError(f"IGRA station file is missing required columns: {missing}")

    stations = stations[required].copy()
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations = stations.dropna(subset=["station_id", "latitude", "longitude"]).drop_duplicates()
    return stations


def load_inversions(path: Path) -> pd.DataFrame:
    inversions = read_required_csv(path)
    inversions.columns = [column.strip().lower() for column in inversions.columns]
    required = ["station_id", "date", "is_inversion"]
    missing = [column for column in required if column not in inversions.columns]
    if missing:
        raise ValueError(f"IGRA inversion file is missing required columns: {missing}")

    inversions["date"] = standardize_date(inversions["date"])
    inversions["is_inversion"] = pd.to_numeric(inversions["is_inversion"], errors="coerce")
    return inversions.dropna(subset=["station_id", "date", "is_inversion"]).copy()


def load_county_day_file(path: Path, label: str) -> pd.DataFrame:
    df = read_required_csv(path)
    df.columns = [column.strip().lower() for column in df.columns]
    required = ["county_fips", "date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} file is missing required columns: {missing}")

    df = df.copy()
    df["county_fips"] = standardize_fips(df["county_fips"], 5)
    df["date"] = standardize_date(df["date"])
    return df.dropna(subset=["county_fips", "date"])


def match_monitors_to_stations(
    monitors: pd.DataFrame, stations: pd.DataFrame, max_distance_km: float
) -> pd.DataFrame:
    unique_monitors = (
        monitors[["monitor_id", "county_fips", "latitude", "longitude"]].drop_duplicates().copy()
    )
    expanded = unique_monitors.merge(stations, how="cross", suffixes=("_monitor", "_igra"))
    expanded["distance_km"] = haversine_km(
        expanded["latitude_monitor"],
        expanded["longitude_monitor"],
        expanded["latitude_igra"],
        expanded["longitude_igra"],
    )
    nearest = (
        expanded.sort_values(["monitor_id", "distance_km"])
        .groupby("monitor_id", as_index=False)
        .first()
        .rename(
            columns={
                "station_id": "igra_station_id",
                "distance_km": "igra_distance_km",
                "latitude_igra": "igra_latitude",
                "longitude_igra": "igra_longitude",
            }
        )
    )
    return nearest.loc[nearest["igra_distance_km"] <= max_distance_km].copy()


def build_monitor_day_master(
    aqs: pd.DataFrame,
    matched_stations: pd.DataFrame,
    inversions: pd.DataFrame,
) -> pd.DataFrame:
    monitor_day = aqs.merge(
        matched_stations[
            [
                "monitor_id",
                "igra_station_id",
                "igra_distance_km",
                "igra_latitude",
                "igra_longitude",
            ]
        ],
        on="monitor_id",
        how="inner",
    )
    monitor_day = monitor_day.merge(
        inversions,
        left_on=["igra_station_id", "date"],
        right_on=["station_id", "date"],
        how="left",
    )
    if "station_id" in monitor_day.columns:
        monitor_day = monitor_day.drop(columns=["station_id"])
    return monitor_day.sort_values(["date", "county_fips", "monitor_id"]).reset_index(drop=True)


def build_county_day_master(
    monitor_day: pd.DataFrame, weather: pd.DataFrame, health: pd.DataFrame
) -> pd.DataFrame:
    protected_columns = {
        "monitor_id",
        "county_fips",
        "date",
        "latitude",
        "longitude",
        "igra_station_id",
        "igra_latitude",
        "igra_longitude",
    }
    numeric_columns = [
        column
        for column in monitor_day.columns
        if column not in protected_columns and pd.api.types.is_numeric_dtype(monitor_day[column])
    ]

    county_day = (
        monitor_day.groupby(["county_fips", "date"], as_index=False)
        .agg({**{column: "mean" for column in numeric_columns}, "monitor_id": "nunique"})
        .rename(columns={"monitor_id": "monitor_count"})
    )

    county_day = county_day.merge(weather, on=["county_fips", "date"], how="left")
    county_day = county_day.merge(
        health, on=["county_fips", "date"], how="left", suffixes=("", "_health")
    )
    return county_day.sort_values(["date", "county_fips"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)

    igra_dir = data_dir / "igra"
    stations_path = igra_dir / "stations.csv"
    inversions_path = igra_dir / "inversion_daily.csv"
    soundings_dir = igra_dir / "soundings"

    if not stations_path.exists() or not inversions_path.exists():
        if soundings_dir.exists():
            build_igra_daily_files(
                soundings_dir=soundings_dir,
                output_dir=igra_dir,
                lower_hpa=args.igra_lower_hpa,
                upper_hpa=args.igra_upper_hpa,
                pressure_tolerance_hpa=args.igra_pressure_tolerance_hpa,
            )
        else:
            missing = [path.name for path in [stations_path, inversions_path] if not path.exists()]
            raise FileNotFoundError(
                "Missing IGRA standardized files "
                f"{missing} and no raw sounding directory found at {soundings_dir}"
            )

    aqs = load_aqs(data_dir / "aqs")
    igra_stations = load_igra_stations(stations_path)
    inversions = load_inversions(inversions_path)
    weather = load_county_day_file(data_dir / "weather" / "weather_daily.csv", "Weather")
    health = load_county_day_file(data_dir / "health" / "health_daily.csv", "Health")

    matched = match_monitors_to_stations(aqs, igra_stations, args.max_distance_km)
    if matched.empty:
        raise ValueError(
            "No AQS monitors were within the requested distance threshold of an IGRA station."
        )

    monitor_day = build_monitor_day_master(aqs, matched, inversions)
    county_day = build_county_day_master(monitor_day, weather, health)

    monitor_day.to_csv(output_dir / "monitor_day_master.csv", index=False)
    county_day.to_csv(output_dir / "county_day_master.csv", index=False)

    print(f"Wrote {len(monitor_day):,} rows to {output_dir / 'monitor_day_master.csv'}")
    print(f"Wrote {len(county_day):,} rows to {output_dir / 'county_day_master.csv'}")


if __name__ == "__main__":
    main()
