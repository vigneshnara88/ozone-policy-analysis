# Ozone policy analysis

An exploratory environmental-economics workflow connecting pollution measurement, county-level policy comparisons, and the assumptions behind an empirical design. The code supports two related strands: ozone outcomes around a historical designation threshold, and linking temperature inversions with pollution, weather, and health data.

The repository brings together data preparation, graphical diagnostics, and RD/DiD specifications. It contains code and input documentation; source datasets, unpublished reports, and saved effect estimates are not bundled.

## Motivation and research questions

Environmental policy raises two connected empirical questions: does a regulatory intervention change pollution, and what can variation in pollution tell us about health? Answering either requires more than comparing high- and low-pollution places. Those places can differ in industry, weather, population and many other characteristics that also affect outcomes.

The first strand investigates whether counties on different sides of a historical ozone threshold show different subsequent pollution changes. Counties close to a cutoff may offer a more comparable local contrast than counties far apart in their initial pollution levels. The second strand assembles temperature-inversion measures with pollution, weather and health records so that the relationship between atmospheric conditions and measured exposure can be studied. These strands have distinct input structures and do not form one automatic causal pipeline.

## How the methods fit the questions

For the threshold strand, the preparation code converts monitor-day observations into county summaries, joins assignment-related fields, and creates pre/post outcome windows. Local regression-discontinuity comparisons focus on observations near the chosen cutoff. RD-in-changes examines whether the change in an outcome differs across it, while panel difference-in-differences and event-study specifications examine the time pattern of the comparison. Together, these tools help investigate whether a threshold-based design has useful empirical support.

The key distinction is between a pollution measurement and a policy assignment variable. An annual arithmetic mean is useful for describing pollution, but it does not automatically reproduce the regulatory design value or the administrative assignment process. The code keeps these inputs explicit because a threshold comparison is only as meaningful as the variable and institutional rule it represents.

For the inversion strand, the mechanism of interest is whether atmospheric temperature structure is associated with pollution accumulation. The code derives inversion indicators from soundings, matches monitors to nearby sounding stations, and joins observations to weather and health inputs. That produces data for examining a potential source of exposure variation. Establishing an instrumental-variable argument would require additional evidence that inversions predict the relevant pollution measure and affect health only through the allowed channels; the merge alone does not provide it.

This project is therefore an exploration of measurement and identification as well as model fitting. Monitor coverage, geographic matching, unit conversions, historical exposure and the choice of outcome window all change what comparison an estimate represents. The repository documents those choices so readers can follow the route from raw records to a proposed policy comparison.

## Workflow

| Component | Purpose |
| --- | --- |
| `src/ozone_master/build_master_dataset.py` | Prepare monitor-day and county-year ozone summaries, threshold exceedance measures, and pre/post comparisons; optionally merge health and designation fields. |
| `src/ozone_master/prepare_designations.py` | Parse locally saved Green Book HTML tables into a county-level merge file. |
| `src/ozone_master/make_plots.py` | Produce exploratory SVG figures from the prepared county tables. |
| `src/ozone_master/run_rd_diagnostics.py` | Examine local linear comparisons, binned outcomes, and running-variable density around a selected cutoff. |
| `src/metrics_data_merger/` | Parse sounding records, derive temperature-inversion indicators, match pollution monitors to nearby sounding stations, and merge county-day weather and health variables. |
| `analysis/run_first_stage_analysis.py` | Estimate local RD-in-changes models, annual DiD, event studies, local RD/DiD comparisons, density diagnostics, and specification sensitivity using separately prepared county panels. |

The research code uses a 70 ppb threshold, a 2014–2016 baseline, and several subsequent outcome windows. These are the workflow's historical design choices. They are not a statement about current legal standards or a complete implementation of an administrative designation process.

## Setup

Python 3.11 or newer is required. Install the preparation packages from the repository root:

```powershell
python -m pip install -e .
```

For the separate econometric analysis script, include its optional dependencies:

```powershell
python -m pip install -e ".[analysis]"
```

The package metadata lists dependencies used by the source. It is not a lockfile for a recorded execution environment.

## Ozone preparation

Place compatible AQS CSV files in `data/raw/aqs/`. See [the input contracts](docs/INPUTS.md) for column requirements and units.

```powershell
python -m ozone_master.build_master_dataset --aqs-dir data/raw/aqs --output-dir data/processed/ozone --pre-years 2014 2015 2016 --post-years 2021 2022 2023 --ozone-threshold-ppm 0.070
```

The optional health and designation inputs default to `data/raw/health/health_county_year.csv` and `data/raw/designations/county_designations.csv`.

With compatible saved designation HTML files, the preparation utility can create the designation merge file:

```powershell
python -m ozone_master.prepare_designations --county-area-html data/raw/designations/greenbook_2015_ozone_county_area.html --design-values-html data/raw/designations/greenbook_2015_ozone_design_values.html --output-file data/raw/designations/county_designations.csv
```

That parser expects a particular table structure, including headers from a saved 2025 table layout. It is not a general scraper for the current website.

For graphical summaries and exploratory local diagnostics:

```powershell
python -m ozone_master.make_plots --county-year-file data/processed/ozone/county_year_master.csv --county-analysis-file data/processed/ozone/county_analysis_master.csv --output-dir outputs/ozone-plots
python -m ozone_master.run_rd_diagnostics --analysis-file data/processed/ozone/county_analysis_master.csv --output-dir outputs/ozone-rd --cutoff 0.070 --bandwidth 0.005
```

The diagnostics require designation/running-variable information in the analysis table. An ozone-only preparation run does not supply those fields by itself.

## Inversion and pollution linkage

The second package uses its own raw-input tree containing AQS, IGRA, weather, and health files:

```powershell
python -m metrics_data_merger.build_master_dataset --data-dir data/raw/inversions --output-dir data/processed/inversions --max-distance-km 150 --igra-lower-hpa 925 --igra-upper-hpa 850
```

Supply standardized station/inversion tables or compatible raw IGRA sounding files. The parser supports text, gzip, and ZIP inputs. The implemented inversion indicator compares temperatures near the configured pressure levels; the default is an upper-level temperature greater than the lower-level temperature. The daily indicator records whether an included sounding shows an inversion, while continuous measures are averaged.

Each monitor is matched to its nearest station within the distance limit. Monitor-day records are then aggregated and merged with county-day weather and health records. Both weather and health input files are required by this package.

`metrics_data_merger.download_igra` is an optional downloader for NOAA station files. Running that module makes network requests; the preparation and analysis commands above use local files.

## First-stage econometric analysis

The econometric script expects **separately prepared** files beside it:

- `analysis/county_year_panel.csv`
- `analysis/county_daily_outcomes.csv`

Their schemas are documented in [docs/INPUTS.md](docs/INPUTS.md). Despite its filename, `county_daily_outcomes.csv` contains annual summaries of daily measures. The package outputs are not drop-in replacements for these files: variable names, units, and aggregation rules differ.

Once those inputs exist:

```powershell
python analysis/run_first_stage_analysis.py
```

The script writes under `analysis/analysis_outputs/`. Its baseline screen uses the 2010 design value, with a stricter historical-exposure screen included for comparison. The primary local specification uses `rdrobust` with a triangular kernel, local linear fit, MSE-selected bandwidth, and nearest-neighbor variance estimation. Additional models examine alternative outcome windows, county/year effects, event studies, density, and local panel comparisons.

## Interpretation and limitations

Pre/post changes and threshold comparisons do not by themselves establish a causal effect. An interpretation requires attention to the assignment process, historical exposure, monitor coverage, spatial spillovers, concurrent policies, and the validity of the comparison group. The inversion merge prepares exposure variables; it does not establish an instrument's relevance or exclusion restriction, or estimate a pollution-to-health causal effect by itself.

The ozone preparation module uses `Arithmetic Mean` values in ppm. Those summaries and exceedance counts should not be confused with a regulatory design value based on a different measurement definition. Design values and assignment fields must be supplied through an appropriate external preparation process. The first-stage analysis uses ppb; a ppm-to-ppb conversion multiplies by 1,000.

The specification grid is exploratory. Its generated summary includes the strongest negative estimate meeting a significance filter; that selection is not an independent confirmatory result, and the full grid matters for interpretation. Threshold conventions at exactly zero and historical screens should be checked against the intended design before substantive use.

The first-stage script contains a compatibility patch for older `rddensity` internals that changes pandas behavior within the process. Importing it also creates its output directory. Treat it as an analysis entry point rather than a general-purpose library.

Outputs, data, and local credentials are excluded by `.gitignore`. No numerical policy-effect claim is made here.
