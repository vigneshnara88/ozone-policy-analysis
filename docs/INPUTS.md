# Input contracts

All datasets are supplied locally. Column names and units below describe the existing code, not a promise that any downloaded file will already match.

## Ozone county summaries

`ozone_master.build_master_dataset` reads CSV files from `--aqs-dir`. After normalizing spaces and case, the required columns are:

```text
state_code, county_code, site_num, date_local, arithmetic_mean
```

When present, `parameter_code` or `parameter_name` is used to keep ozone records (`44201` or `Ozone`). Otherwise the input is assumed to contain ozone only. `arithmetic_mean` is interpreted as ppm. The loader combines records for a monitor and day and then aggregates across monitors.

The optional health table needs `county_fips` and either `year` or a date column. Numeric health fields are carried into the county-year summaries. The optional designation table needs `county_fips`; useful additional fields include `designation_design_value_ppm`, `distance_from_threshold`, and treatment indicators.

Outputs are `county_day_ozone.csv`, `county_year_master.csv`, and `county_analysis_master.csv`. The last contains selected pre/post averages and changes.

The designation HTML parser expects the specific saved county-area and design-value tables referenced in its CLI defaults. Table positions and header labels are hard-coded, including labels referring to 2022–2024 values in a 2025 table. Changed tables require an updated mapping.

## Inversion linkage

Under the `--data-dir` passed to `metrics_data_merger.build_master_dataset`, supply:

| Path | Required fields or contents |
| --- | --- |
| `aqs/*.csv` | Standard AQS identifiers, `date_local`, `latitude`, `longitude`, `arithmetic_mean`, and a recognized pollutant code or name. |
| `igra/stations.csv` | `station_id`, `latitude`, `longitude`. |
| `igra/inversion_daily.csv` | `station_id`, `date`, `is_inversion`; optional strength, lapse-rate, temperature, and height fields. |
| `weather/weather_daily.csv` | `county_fips`, `date`, and the chosen weather controls. |
| `health/health_daily.csv` | `county_fips`, `date`, and the chosen outcome fields. |

If either standardized IGRA file is absent, a populated `igra/soundings/` directory can be parsed to create both. The default target pressure levels are 925 and 850 hPa, with a 15 hPa matching tolerance. Raw formats include `.txt`, `.gz`, `.zip`, and `.txt.zip`.

The weather and health tables are required files even though individual merged records may be missing. The script writes `monitor_day_master.csv` and `county_day_master.csv`. Monitor matching is geographical nearest-neighbor matching within a specified radius, not a demonstration of atmospheric representativeness.

## Econometric county panels

Place the following two files in `analysis/`, alongside `run_first_stage_analysis.py`.

`county_year_panel.csv` uses this source schema:

```text
Year
county_fips
county_4thmax_ppb
county_mean_ppb
n_monitors
state_name
county_name
design_value_3yr
running_var_70
running_var_75
nonattain_70
nonattain_75
newly_nonattain
```

The analysis interprets design values, fourth-maximum outcomes, and running-variable distances in ppb. The sample uses the 2016 row and historical screens involving 2010 and years before 2015. The supplied panel must contain the outcome years requested by the script; its dynamic windows currently extend through 2025. Complete pre/post year counts are required within each selected window.

`county_daily_outcomes.csv` is an annual county table summarizing daily observations. Its source schema is:

```text
year
county_fips
n_obs
mean_daily_max
max_daily_max
p95_daily_max
p99_daily_max
exceedance_days_70
exceedance_days_75
exceedance_days_65
mean_aqi
max_aqi
season_mean
season_p95
season_exceed70
season_n_days
```

The current alternative-outcome models use `mean_daily_max`, `exceedance_days_70`, `p95_daily_max`, and `season_mean`. Concentration outcomes must be consistently expressed in ppb for the plotted and reported units. Preserve county identifiers consistently across both files and provide one observation per county-year at the intended aggregation level.

There is no automatic adapter from the preparation packages' outputs to these two schemas. In particular, an arithmetic-mean ozone outcome is not interchangeable with an annual fourth-maximum measure, and county threshold proxies are not automatically administrative designation records.
