from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import zipfile

import numpy as np
import pandas as pd


HEADER_MIN_LENGTH = 71
DATA_MIN_LENGTH = 51
MISSING_VALUES = {-9999, -8888}


@dataclass
class SoundingHeader:
    station_id: str
    date: pd.Timestamp
    numlev: int
    latitude: float
    longitude: float


def _open_text(path: Path):
    suffixes = path.suffixes
    if suffixes[-2:] == [".txt", ".zip"] or path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one file inside {path}")
            with archive.open(members[0], "r") as handle:
                for raw in handle:
                    yield raw.decode("utf-8", errors="replace").rstrip("\r\n")
        return
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line.rstrip("\r\n")
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def _parse_header(line: str) -> SoundingHeader:
    if len(line) < HEADER_MIN_LENGTH or not line.startswith("#"):
        raise ValueError("Invalid IGRA header record")

    year = int(line[13:17])
    month = int(line[18:20])
    day = int(line[21:23])
    hour = int(line[24:26])
    timestamp = pd.Timestamp(year=year, month=month, day=day)
    if hour != 99:
        timestamp = timestamp + pd.Timedelta(hours=hour)

    return SoundingHeader(
        station_id=line[1:12].strip(),
        date=timestamp,
        numlev=int(line[32:36]),
        latitude=int(line[55:62]) / 10000.0,
        longitude=int(line[63:71]) / 10000.0,
    )


def _parse_data_record(line: str) -> dict[str, float | int | None]:
    if len(line) < DATA_MIN_LENGTH:
        raise ValueError("Invalid IGRA data record")

    pressure_raw = int(line[9:15])
    gph_raw = int(line[16:21])
    temp_raw = int(line[22:27])

    return {
        "lvltyp1": int(line[0:1]),
        "lvltyp2": int(line[1:2]),
        "pressure_pa": None if pressure_raw in MISSING_VALUES else pressure_raw,
        "height_m": None if gph_raw in MISSING_VALUES else gph_raw,
        "temp_c": None if temp_raw in MISSING_VALUES else temp_raw / 10.0,
    }


def _extract_target_levels(
    levels: pd.DataFrame, lower_hpa: int, upper_hpa: int, pressure_tolerance_hpa: int
) -> tuple[pd.Series | None, pd.Series | None]:
    pressure_hpa = levels["pressure_pa"] / 100.0
    candidates = levels.assign(pressure_hpa=pressure_hpa).dropna(
        subset=["pressure_hpa", "temp_c", "height_m"]
    )
    if candidates.empty:
        return None, None

    lower_candidates = candidates.loc[
        (candidates["pressure_hpa"] - lower_hpa).abs() <= pressure_tolerance_hpa
    ].copy()
    upper_candidates = candidates.loc[
        (candidates["pressure_hpa"] - upper_hpa).abs() <= pressure_tolerance_hpa
    ].copy()
    if lower_candidates.empty or upper_candidates.empty:
        return None, None

    lower_row = lower_candidates.iloc[(lower_candidates["pressure_hpa"] - lower_hpa).abs().argmin()]
    upper_row = upper_candidates.iloc[(upper_candidates["pressure_hpa"] - upper_hpa).abs().argmin()]
    return lower_row, upper_row


def compute_inversion_from_levels(
    levels: pd.DataFrame, lower_hpa: int = 925, upper_hpa: int = 850, pressure_tolerance_hpa: int = 15
) -> dict[str, float | int | pd.Timestamp | str | None] | None:
    lower_row, upper_row = _extract_target_levels(levels, lower_hpa, upper_hpa, pressure_tolerance_hpa)
    if lower_row is None or upper_row is None:
        return None

    height_diff = upper_row["height_m"] - lower_row["height_m"]
    if pd.isna(height_diff) or height_diff <= 0:
        return None

    temp_diff = upper_row["temp_c"] - lower_row["temp_c"]
    lapse_rate = temp_diff / height_diff

    header_row = levels.iloc[0]
    return {
        "station_id": header_row["station_id"],
        "datetime_utc": header_row["datetime_utc"],
        "date": header_row["datetime_utc"].normalize(),
        "is_inversion": int(lapse_rate > 0),
        "inversion_strength": round(float(temp_diff), 4),
        "lapse_rate": round(float(lapse_rate), 6),
        "temp_lower_c": float(lower_row["temp_c"]),
        "temp_upper_c": float(upper_row["temp_c"]),
        "height_lower_m": float(lower_row["height_m"]),
        "height_upper_m": float(upper_row["height_m"]),
        "pressure_lower_hpa": float(lower_row["pressure_hpa"]),
        "pressure_upper_hpa": float(upper_row["pressure_hpa"]),
        "sounding_latitude": float(header_row["latitude"]),
        "sounding_longitude": float(header_row["longitude"]),
    }


def parse_igra_station_file(
    path: Path, lower_hpa: int = 925, upper_hpa: int = 850, pressure_tolerance_hpa: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows: list[dict[str, object]] = []
    inversion_rows: list[dict[str, object]] = []

    lines = _open_text(path)
    for line in lines:
        if not line:
            continue
        if not line.startswith("#"):
            raise ValueError(f"Unexpected IGRA record outside sounding header in {path}")

        header = _parse_header(line)
        station_rows.append(
            {
                "station_id": header.station_id,
                "latitude": header.latitude,
                "longitude": header.longitude,
            }
        )

        level_rows = []
        for _ in range(header.numlev):
            try:
                level_line = next(lines)
            except StopIteration as exc:
                raise ValueError(f"Unexpected end of file in {path}") from exc
            row = _parse_data_record(level_line)
            row["station_id"] = header.station_id
            row["datetime_utc"] = header.date
            row["latitude"] = header.latitude
            row["longitude"] = header.longitude
            level_rows.append(row)

        levels = pd.DataFrame(level_rows)
        inversion = compute_inversion_from_levels(
            levels,
            lower_hpa=lower_hpa,
            upper_hpa=upper_hpa,
            pressure_tolerance_hpa=pressure_tolerance_hpa,
        )
        if inversion is not None:
            inversion_rows.append(inversion)

    stations = pd.DataFrame(station_rows).drop_duplicates().reset_index(drop=True)
    inversions = pd.DataFrame(inversion_rows)
    return stations, inversions


def parse_igra_soundings_directory(
    soundings_dir: Path, lower_hpa: int = 925, upper_hpa: int = 850, pressure_tolerance_hpa: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patterns = ["*.txt", "*.zip", "*.gz", "*.txt.zip"]
    files = []
    for pattern in patterns:
        files.extend(soundings_dir.glob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No IGRA sounding files found in {soundings_dir}")

    station_frames = []
    inversion_frames = []
    for path in files:
        stations, inversions = parse_igra_station_file(
            path,
            lower_hpa=lower_hpa,
            upper_hpa=upper_hpa,
            pressure_tolerance_hpa=pressure_tolerance_hpa,
        )
        station_frames.append(stations)
        if not inversions.empty:
            inversion_frames.append(inversions)

    station_df = pd.concat(station_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if inversion_frames:
        inversion_df = pd.concat(inversion_frames, ignore_index=True)
    else:
        inversion_df = pd.DataFrame(
            columns=[
                "station_id",
                "datetime_utc",
                "date",
                "is_inversion",
                "inversion_strength",
                "lapse_rate",
                "temp_lower_c",
                "temp_upper_c",
                "height_lower_m",
                "height_upper_m",
                "pressure_lower_hpa",
                "pressure_upper_hpa",
                "sounding_latitude",
                "sounding_longitude",
            ]
        )
    return station_df, inversion_df


def aggregate_daily_inversions(inversions: pd.DataFrame) -> pd.DataFrame:
    if inversions.empty:
        return inversions.copy()

    aggregations = {
        "is_inversion": "max",
        "inversion_strength": "mean",
        "lapse_rate": "mean",
        "temp_lower_c": "mean",
        "temp_upper_c": "mean",
        "height_lower_m": "mean",
        "height_upper_m": "mean",
        "pressure_lower_hpa": "mean",
        "pressure_upper_hpa": "mean",
        "sounding_latitude": "mean",
        "sounding_longitude": "mean",
        "datetime_utc": "count",
    }
    daily = (
        inversions.groupby(["station_id", "date"], as_index=False)
        .agg(aggregations)
        .rename(columns={"datetime_utc": "soundings_used"})
    )
    return daily


def build_igra_daily_files(
    soundings_dir: Path,
    output_dir: Path,
    lower_hpa: int = 925,
    upper_hpa: int = 850,
    pressure_tolerance_hpa: int = 15,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stations, inversions = parse_igra_soundings_directory(
        soundings_dir,
        lower_hpa=lower_hpa,
        upper_hpa=upper_hpa,
        pressure_tolerance_hpa=pressure_tolerance_hpa,
    )
    daily = aggregate_daily_inversions(inversions)

    stations_path = output_dir / "stations.csv"
    inversions_path = output_dir / "inversion_daily.csv"
    stations.to_csv(stations_path, index=False)
    daily.to_csv(inversions_path, index=False)
    return stations_path, inversions_path
