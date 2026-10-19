from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


STATION_LIST_URL = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
BASE_URLS = {
    "por": "https://www.ncei.noaa.gov/pub/data/igra/data/data-por",
    "y2d": "https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA IGRA station files.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/igra/soundings",
        help="Directory to write downloaded station files",
    )
    parser.add_argument(
        "--station-id",
        action="append",
        dest="station_ids",
        default=[],
        help="IGRA station ID to download, may be supplied multiple times",
    )
    parser.add_argument(
        "--station-file",
        help="Optional text file containing one IGRA station ID per line",
    )
    parser.add_argument(
        "--period",
        choices=sorted(BASE_URLS),
        default="y2d",
        help="Choose current-year-to-date or full-period-of-record files",
    )
    parser.add_argument(
        "--download-station-list",
        action="store_true",
        help="Also download the official IGRA station list text file",
    )
    return parser.parse_args()


def load_station_ids(args: argparse.Namespace) -> list[str]:
    station_ids = [station.strip() for station in args.station_ids if station.strip()]
    if args.station_file:
        file_ids = [
            line.strip()
            for line in Path(args.station_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        station_ids.extend(file_ids)
    station_ids = sorted(set(station_ids))
    if not station_ids and not args.download_station_list:
        raise ValueError("Provide at least one --station-id or use --download-station-list")
    return station_ids


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    station_ids = load_station_ids(args)

    if args.download_station_list:
        station_list_path = output_dir.parent / "igra2-station-list.txt"
        download(STATION_LIST_URL, station_list_path)
        print(f"Downloaded station list to {station_list_path}")

    base_url = BASE_URLS[args.period]
    for station_id in station_ids:
        filename = f"{station_id}-data.txt.zip"
        destination = output_dir / filename
        download(f"{base_url}/{filename}", destination)
        print(f"Downloaded {station_id} to {destination}")


if __name__ == "__main__":
    main()
