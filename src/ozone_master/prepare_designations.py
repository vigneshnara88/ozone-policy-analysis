from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EPA Green Book ozone designation tables into a county-level CSV."
    )
    parser.add_argument(
        "--county-area-html",
        default="data/raw/designations/greenbook_2015_ozone_county_area.html",
        help="Downloaded EPA county-area Green Book HTML file",
    )
    parser.add_argument(
        "--design-values-html",
        default="data/raw/designations/greenbook_2015_ozone_design_values.html",
        help="Downloaded EPA design-value Green Book HTML file",
    )
    parser.add_argument(
        "--output-file",
        default="data/raw/designations/county_designations.csv",
        help="Output county-level designation CSV",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(part).strip() for part in col if str(part) != "nan").strip("_")
            for col in out.columns
        ]
    out.columns = [normalize_name(column) for column in out.columns]
    return out


def standardize_fips_piece(value: object, width: int) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace(".0", "")
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(width)


def parse_county_fips(series: pd.Series) -> pd.Series:
    state_part = series.astype("string").str.extract(r"^\s*(\d{1,2})", expand=False)
    county_part = series.astype("string").str.extract(r"(\d{1,3})\s*$", expand=False)
    return state_part.str.zfill(2) + county_part.str.zfill(3)


def load_county_area(path: Path) -> pd.DataFrame:
    table = pd.read_html(path)[0]
    table = flatten_columns(table)
    table = table.rename(
        columns={
            "state_state": "state",
            "county_county": "county",
            "area_name_area_name": "area_name",
            "current_status_current_status": "current_status",
            "county_na_whole_part_click_link_for_partial_county_description": "county_na_whole_part",
            "2010_population_whole_or_partial_county_population": "population_2010",
            "state_county_fips_state_county_fips": "state_county_fips",
        }
    )
    keep = [
        "state",
        "county",
        "area_name",
        "current_status",
        "county_na_whole_part",
        "population_2010",
        "state_county_fips",
    ]
    county = table[keep].copy()
    county = county[county["state"].str.upper().ne("STATE")].copy()
    county["county_fips"] = parse_county_fips(county["state_county_fips"])
    county["population_2010"] = pd.to_numeric(county["population_2010"], errors="coerce")
    county["listed_in_greenbook_2015_ozone"] = 1
    county["ever_nonattainment_or_maintenance_2015_ozone"] = 1
    return county.dropna(subset=["county_fips"]).reset_index(drop=True)


def load_design_values(path: Path) -> pd.DataFrame:
    tables = pd.read_html(path)
    table = flatten_columns(tables[1])
    table = table.rename(
        columns={
            "click_underlined_column_heading_to_change_report_order_area_name_click_area_name_for_state_county_list": "area_name",
            "design_values_at_the_time_of_designation_design_values_design_values": "designation_design_value_ppm",
            "design_values_at_the_time_of_designation_monitoring_years_monitoring_years": "designation_monitoring_years",
            "current_design_values_as_of_05_28_2025_2022_2024_design_values_design_values": "current_design_value_ppm_2022_2024",
            "current_design_values_as_of_05_28_2025_2022_2024_meets_naaqs_determin_1_meets_naaqs_determin_1": "meets_naaqs_2022_2024",
            "unnamed_5_level_0_current_status_current_status": "current_status",
            "unnamed_6_level_0_current_classification_or_at_redesignation_click_for_classif_history": "current_classification_or_redesignation",
            "unnamed_7_level_0_2010_population_2010_population": "area_population_2010",
        }
    )
    keep = [
        "area_name",
        "designation_design_value_ppm",
        "designation_monitoring_years",
        "current_design_value_ppm_2022_2024",
        "meets_naaqs_2022_2024",
        "current_status",
        "current_classification_or_redesignation",
        "area_population_2010",
    ]
    values = table[keep].copy()
    values = values[values["area_name"].str.lower().ne("area_name")].copy()
    values["designation_design_value_ppm"] = pd.to_numeric(
        values["designation_design_value_ppm"], errors="coerce"
    )
    values["current_design_value_ppm_2022_2024"] = pd.to_numeric(
        values["current_design_value_ppm_2022_2024"], errors="coerce"
    )
    values["area_population_2010"] = pd.to_numeric(values["area_population_2010"], errors="coerce")
    values = (
        values.groupby(["area_name", "current_status"], as_index=False)
        .agg(
            designation_design_value_ppm=("designation_design_value_ppm", "first"),
            designation_monitoring_years=("designation_monitoring_years", "first"),
            current_design_value_ppm_2022_2024=("current_design_value_ppm_2022_2024", "first"),
            meets_naaqs_2022_2024=("meets_naaqs_2022_2024", "first"),
            current_classification_or_redesignation=(
                "current_classification_or_redesignation",
                lambda s: " | ".join(sorted({str(v) for v in s.dropna()})),
            ),
            area_population_2010=("area_population_2010", "first"),
        )
    )
    return values.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    county = load_county_area(Path(args.county_area_html))
    values = load_design_values(Path(args.design_values_html))
    merged = county.merge(
        values,
        on=["area_name", "current_status"],
        how="left",
        validate="many_to_one",
    )

    merged["above_70"] = (merged["designation_design_value_ppm"] > 0.070).astype("Int64")
    merged["above_75"] = (merged["designation_design_value_ppm"] > 0.075).astype("Int64")
    merged["between_70_and_75"] = (
        (merged["designation_design_value_ppm"] > 0.070)
        & (merged["designation_design_value_ppm"] <= 0.075)
    ).astype("Int64")
    merged["distance_from_70"] = merged["designation_design_value_ppm"] - 0.070
    merged["distance_from_75"] = merged["designation_design_value_ppm"] - 0.075

    # "Switched" means the county falls above the new 70 ppb standard but not above 75 ppb,
    # so it is newly captured by the tighter 2015 NAAQS rather than the older 75 ppb rule.
    merged["newly_treated_70_rule"] = merged["between_70_and_75"]

    # This narrower flag keeps only counties that also appear in the EPA county-area designation list.
    merged["switched_and_listed_by_epa"] = (
        (merged["newly_treated_70_rule"] == 1)
        & (merged["listed_in_greenbook_2015_ozone"] == 1)
    ).astype("Int64")

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Wrote {len(merged):,} rows to {output}")


if __name__ == "__main__":
    main()
