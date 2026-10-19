from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdrobust import rdrobust
from rddensity import rddensity
from scipy.stats import chi2
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DESIGNATION_YEAR = 2016
BASELINE_2008_STANDARD_YEAR = 2010


def patch_pandas_for_rddensity() -> None:
    """Patch pandas methods that older rddensity internals still expect."""
    if not hasattr(pd.Series, "_append"):
        def _append(self, to_append, ignore_index=False, verify_integrity=False):
            return pd.concat([self, to_append], ignore_index=ignore_index)
        pd.Series._append = _append  # type: ignore[attr-defined]

    old_getitem = pd.Series.__getitem__

    if getattr(pd.Series.__getitem__, "_codex_patched", False):
        return

    def _getitem(self, key):
        try:
            return old_getitem(self, key)
        except KeyError:
            if isinstance(key, int):
                return self.iloc[key]
            raise

    _getitem._codex_patched = True  # type: ignore[attr-defined]
    pd.Series.__getitem__ = _getitem  # type: ignore[assignment]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(ROOT / "county_year_panel.csv")
    daily = pd.read_csv(ROOT / "county_daily_outcomes.csv")
    return panel, daily


def build_historical_treatment_flags(panel: pd.DataFrame) -> pd.DataFrame:
    pre2015 = panel.loc[panel["Year"] <= 2014, ["county_fips", "Year", "design_value_3yr"]].copy()
    flags = (
        pre2015.groupby("county_fips")
        .agg(
            ever_dv_pre2015=("design_value_3yr", lambda s: int(s.notna().any())),
            ever_above_75_pre2015=("design_value_3yr", lambda s: int((s > 75).any())),
            ever_above_84_pre2015=("design_value_3yr", lambda s: int((s > 84).any())),
            max_design_value_pre2015=("design_value_3yr", "max"),
        )
        .reset_index()
    )

    dv_2010 = panel.loc[
        panel["Year"] == 2010,
        ["county_fips", "design_value_3yr"],
    ].rename(columns={"design_value_3yr": "design_value_2010"})

    flags = flags.merge(dv_2010, on="county_fips", how="left")
    flags["nonattain_2008_2010dv"] = (flags["design_value_2010"] > 75).fillna(False).astype(int)
    flags["nonattain_1997_proxy"] = flags["ever_above_84_pre2015"].fillna(0).astype(int)
    flags["treated_pre_2015"] = flags["ever_above_75_pre2015"].fillna(0).astype(int)
    flags["below75_in_2010"] = (flags["design_value_2010"] <= 75).fillna(False).astype(int)
    return flags


def build_designation_sample(panel: pd.DataFrame, screen_mode: str = "baseline_2010_below75") -> pd.DataFrame:
    sample = panel.loc[
        panel["Year"] == DESIGNATION_YEAR,
        [
            "county_fips",
            "state_name",
            "county_name",
            "design_value_3yr",
            "running_var_70",
            "nonattain_70",
            "nonattain_75",
            "newly_nonattain",
        ],
    ].copy()
    sample = sample[sample["design_value_3yr"].notna()].copy()
    flags = build_historical_treatment_flags(panel)
    sample = sample.merge(flags, on="county_fips", how="left")
    sample["treated_pre_2015"] = sample["treated_pre_2015"].fillna(0).astype(int)
    sample["ever_above_75_pre2015"] = sample["ever_above_75_pre2015"].fillna(0).astype(int)
    sample["ever_above_84_pre2015"] = sample["ever_above_84_pre2015"].fillna(0).astype(int)
    sample["nonattain_2008_2010dv"] = sample["nonattain_2008_2010dv"].fillna(0).astype(int)
    sample["nonattain_1997_proxy"] = sample["nonattain_1997_proxy"].fillna(0).astype(int)
    sample["below75_in_2010"] = sample["below75_in_2010"].fillna(0).astype(int)

    if screen_mode == "baseline_2010_below75":
        sample = sample[sample["below75_in_2010"] == 1].copy()
    elif screen_mode == "never_above75_pre2015":
        sample = sample[sample["treated_pre_2015"] == 0].copy()
    else:
        raise ValueError(f"Unknown screen_mode: {screen_mode}")

    sample["screen_mode"] = screen_mode
    return sample


def window_mean(
    df: pd.DataFrame,
    year_col: str,
    value_col: str,
    years: list[int],
    prefix: str,
) -> pd.DataFrame:
    out = (
        df.loc[df[year_col].isin(years), ["county_fips", year_col, value_col]]
        .groupby("county_fips")[value_col]
        .agg(count="count", mean="mean")
        .reset_index()
        .rename(columns={"count": f"{prefix}_count", "mean": f"{prefix}_mean"})
    )
    return out


def build_delta_sample(
    base: pd.DataFrame,
    df: pd.DataFrame,
    year_col: str,
    value_col: str,
    pre_years: list[int],
    post_years: list[int],
) -> pd.DataFrame:
    pre = window_mean(df, year_col, value_col, pre_years, "pre")
    post = window_mean(df, year_col, value_col, post_years, "post")
    out = base.merge(pre, on="county_fips", how="inner").merge(post, on="county_fips", how="inner")
    out = out[(out["pre_count"] == len(pre_years)) & (out["post_count"] == len(post_years))].copy()
    out["delta_outcome"] = out["post_mean"] - out["pre_mean"]
    return out


def run_rd(sample: pd.DataFrame, outcome_col: str = "delta_outcome"):
    return rdrobust(
        sample[outcome_col].to_numpy(),
        sample["running_var_70"].to_numpy(),
        c=0,
        p=1,
        kernel="tri",
        bwselect="mserd",
        vce="nn",
    )


def run_rd_local_ols(sample: pd.DataFrame, bandwidth: float, donut: float = 0.0) -> dict[str, float]:
    work = sample.loc[sample["running_var_70"].abs() <= bandwidth].copy()
    if donut > 0:
        work = work.loc[work["running_var_70"].abs() >= donut].copy()
    work["treat"] = (work["running_var_70"] > 0).astype(int)
    work["interact"] = work["treat"] * work["running_var_70"]
    if len(work) < 25 or work["treat"].nunique() < 2:
        return {
            "n_total": len(work),
            "coef": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
        }
    model = smf.ols("delta_outcome ~ treat + running_var_70 + interact", data=work).fit(
        cov_type="HC1"
    )
    coef = float(model.params["treat"])
    se = float(model.bse["treat"])
    return {
        "n_total": len(work),
        "coef": coef,
        "se": se,
        "ci_lower": coef - 1.96 * se,
        "ci_upper": coef + 1.96 * se,
        "pvalue": float(model.pvalues["treat"]),
    }


def extract_rd_row(
    result,
    label: str,
    sample: pd.DataFrame,
    pre_label: str,
    post_label: str,
    outcome_name: str,
) -> dict[str, object]:
    robust_ci = result.ci.loc["Robust"]
    robust_p = result.pv.loc["Robust", "P>|t|"]
    bc_coef = result.coef.loc["Bias-Corrected", "Coeff"]
    conv_coef = result.coef.loc["Conventional", "Coeff"]

    return {
        "spec": label,
        "outcome": outcome_name,
        "pre_period": pre_label,
        "post_period": post_label,
        "n_total": len(sample),
        "n_left": int((sample["running_var_70"] < 0).sum()),
        "n_right": int((sample["running_var_70"] >= 0).sum()),
        "n_h_left": int(result.N_h[0]),
        "n_h_right": int(result.N_h[1]),
        "bandwidth_left": float(result.bws.loc["h", "left"]),
        "bandwidth_right": float(result.bws.loc["h", "right"]),
        "coef_conventional": float(conv_coef),
        "coef_bc": float(bc_coef),
        "ci_robust_lower": float(robust_ci["CI Lower"]),
        "ci_robust_upper": float(robust_ci["CI Upper"]),
        "pvalue_robust": float(robust_p),
    }


def run_rd_spec_exploration(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre_years = [2014, 2015, 2016]
    post_windows = [
        [2017, 2018, 2019],
        [2018, 2019, 2020],
        [2019, 2020, 2021],
        [2020, 2021, 2022],
        [2021, 2022, 2023],
        [2022, 2023, 2024, 2025],
    ]
    for screen_mode in ["baseline_2010_below75", "never_above75_pre2015"]:
        base = build_designation_sample(panel, screen_mode=screen_mode)
        for outcome in ["county_4thmax_ppb", "county_mean_ppb"]:
            for post_years in post_windows:
                sample = build_delta_sample(base, panel, "Year", outcome, pre_years, post_years)
                if len(sample) == 0:
                    continue
                post_label = f"{post_years[0]}-{post_years[-1]}"
                rd = run_rd(sample)
                row = extract_rd_row(
                    result=rd,
                    label=f"explore_{screen_mode}_{outcome}_{post_label}",
                    sample=sample,
                    pre_label="2014-2016",
                    post_label=post_label,
                    outcome_name=outcome,
                )
                row["screen_mode"] = screen_mode
                row["bandwidth_type"] = "rdrobust_mserd"
                row["donut"] = 0.0
                rows.append(row)
                for bandwidth in [1.5, 2.5, 4.0]:
                    for donut in [0.0, 0.334]:
                        local = run_rd_local_ols(sample, bandwidth=bandwidth, donut=donut)
                        rows.append(
                            {
                                "spec": f"localols_{screen_mode}_{outcome}_{post_label}_bw{bandwidth}_donut{donut}",
                                "outcome": outcome,
                                "pre_period": "2014-2016",
                                "post_period": post_label,
                                "n_total": int(local["n_total"]),
                                "n_left": int(
                                    (
                                        sample.loc[
                                            (sample["running_var_70"].abs() <= bandwidth)
                                            & (sample["running_var_70"].abs() >= donut),
                                            "running_var_70",
                                        ]
                                        < 0
                                    ).sum()
                                ),
                                "n_right": int(
                                    (
                                        sample.loc[
                                            (sample["running_var_70"].abs() <= bandwidth)
                                            & (sample["running_var_70"].abs() >= donut),
                                            "running_var_70",
                                        ]
                                        > 0
                                    ).sum()
                                ),
                                "n_h_left": np.nan,
                                "n_h_right": np.nan,
                                "bandwidth_left": bandwidth,
                                "bandwidth_right": bandwidth,
                                "coef_conventional": local["coef"],
                                "coef_bc": local["coef"],
                                "ci_robust_lower": local["ci_lower"],
                                "ci_robust_upper": local["ci_upper"],
                                "pvalue_robust": local["pvalue"],
                                "screen_mode": screen_mode,
                                "bandwidth_type": f"local_ols_bw_{bandwidth}",
                                "donut": donut,
                            }
                        )
    return pd.DataFrame(rows)


def make_rd_plot(sample: pd.DataFrame, result, path: Path, title: str) -> None:
    h_left = float(result.bws.loc["h", "left"])
    h_right = float(result.bws.loc["h", "right"])
    plot_sample = sample[
        (sample["running_var_70"] >= -h_left) & (sample["running_var_70"] <= h_right)
    ].copy()

    left = plot_sample[plot_sample["running_var_70"] < 0].copy()
    right = plot_sample[plot_sample["running_var_70"] >= 0].copy()

    n_bins_per_side = min(30, max(8, int(np.sqrt(len(plot_sample)) // 2)))
    left_bins = np.linspace(-h_left, 0, n_bins_per_side + 1)
    right_bins = np.linspace(0, h_right, n_bins_per_side + 1)

    left["bin"] = pd.cut(left["running_var_70"], bins=left_bins, include_lowest=True)
    right["bin"] = pd.cut(right["running_var_70"], bins=right_bins, include_lowest=True)

    left_pts = left.groupby("bin", observed=True).agg(
        x=("running_var_70", "mean"),
        y=("delta_outcome", "mean"),
    ).dropna()
    right_pts = right.groupby("bin", observed=True).agg(
        x=("running_var_70", "mean"),
        y=("delta_outcome", "mean"),
    ).dropna()

    beta_l = np.asarray(result.beta_p_l).flatten()
    beta_r = np.asarray(result.beta_p_r).flatten()

    x_l = np.linspace(-h_left, 0, 200)
    x_r = np.linspace(0, h_right, 200)
    y_l = beta_l[0] + beta_l[1] * x_l
    y_r = beta_r[0] + beta_r[1] * x_r

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(left_pts["x"], left_pts["y"], color="#1f77b4", s=30, label="Binned average")
    ax.scatter(right_pts["x"], right_pts["y"], color="#1f77b4", s=30)
    ax.plot(x_l, y_l, color="#d62728", linewidth=2.2, label="Local linear fit")
    ax.plot(x_r, y_r, color="#d62728", linewidth=2.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Running variable: design value - 70 ppb")
    ax.set_ylabel("Change in county 4th-highest ozone (ppb)")
    ax.text(
        0.02,
        0.03,
        f"Binned points shown within RD bandwidth only; {len(left_pts) + len(right_pts)} bins from {len(plot_sample)} counties",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_rd_plot_from_column(
    sample: pd.DataFrame,
    result,
    outcome_col: str,
    y_label: str,
    path: Path,
    title: str,
) -> None:
    h_left = float(result.bws.loc["h", "left"])
    h_right = float(result.bws.loc["h", "right"])
    plot_sample = sample[
        (sample["running_var_70"] >= -h_left) & (sample["running_var_70"] <= h_right)
    ].copy()

    left = plot_sample[plot_sample["running_var_70"] < 0].copy()
    right = plot_sample[plot_sample["running_var_70"] >= 0].copy()

    n_bins_per_side = min(30, max(8, int(np.sqrt(len(plot_sample)) // 2)))
    left_bins = np.linspace(-h_left, 0, n_bins_per_side + 1)
    right_bins = np.linspace(0, h_right, n_bins_per_side + 1)

    left["bin"] = pd.cut(left["running_var_70"], bins=left_bins, include_lowest=True)
    right["bin"] = pd.cut(right["running_var_70"], bins=right_bins, include_lowest=True)

    left_pts = left.groupby("bin", observed=True).agg(
        x=("running_var_70", "mean"),
        y=(outcome_col, "mean"),
    ).dropna()
    right_pts = right.groupby("bin", observed=True).agg(
        x=("running_var_70", "mean"),
        y=(outcome_col, "mean"),
    ).dropna()

    beta_l = np.asarray(result.beta_p_l).flatten()
    beta_r = np.asarray(result.beta_p_r).flatten()
    x_l = np.linspace(-h_left, 0, 200)
    x_r = np.linspace(0, h_right, 200)
    y_l = beta_l[0] + beta_l[1] * x_l
    y_r = beta_r[0] + beta_r[1] * x_r

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(left_pts["x"], left_pts["y"], color="#1f77b4", s=30, label="Binned average")
    ax.scatter(right_pts["x"], right_pts["y"], color="#1f77b4", s=30)
    ax.plot(x_l, y_l, color="#d62728", linewidth=2.2, label="Local linear fit")
    ax.plot(x_r, y_r, color="#d62728", linewidth=2.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Running variable: design value - 70 ppb")
    ax.set_ylabel(y_label)
    ax.text(
        0.02,
        0.03,
        f"Binned points shown within RD bandwidth only; {len(left_pts) + len(right_pts)} bins from {len(plot_sample)} counties",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_parallel_trends_plot(panel_df: pd.DataFrame, outcome: str, path: Path) -> None:
    means = (
        panel_df.groupby(["Year", "treated"])[outcome]
        .mean()
        .reset_index()
        .pivot(index="Year", columns="treated", values=outcome)
        .rename(columns={0: "control", 1: "treated"})
    )
    for col in ["control", "treated"]:
        if 2017 in means.index:
            means[f"{col}_norm"] = means[col] - means.loc[2017, col]
        else:
            means[f"{col}_norm"] = means[col] - means[col].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(means.index, means["control_norm"], marker="o", color="#1f77b4", label="Control counties")
    ax.plot(means.index, means["treated_norm"], marker="o", color="#d62728", label="Treated counties")
    ax.axvline(2018, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="#555555", linewidth=1, alpha=0.5)
    ax.set_title("Parallel-trends check: mean annual ozone relative to 2017")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{outcome} relative to 2017")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_parallel_trends_raw_plot(panel_df: pd.DataFrame, outcome: str, path: Path) -> None:
    means = (
        panel_df.groupby(["Year", "treated"])[outcome]
        .mean()
        .reset_index()
        .pivot(index="Year", columns="treated", values=outcome)
        .rename(columns={0: "control", 1: "treated"})
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(means.index, means["control"], marker="o", color="#1f77b4", label="Control counties")
    ax.plot(means.index, means["treated"], marker="o", color="#d62728", label="Treated counties")
    ax.axvline(2018, color="black", linestyle="--", linewidth=1)
    ax.set_title("Raw annual mean ozone by treatment status")
    ax.set_xlabel("Year")
    ax.set_ylabel(outcome)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_dynamic_plot(results_df: pd.DataFrame, path: Path) -> None:
    ordered = results_df.copy()
    ordered["x"] = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        ordered["x"],
        ordered["coef_bc"],
        yerr=[
            ordered["coef_bc"] - ordered["ci_robust_lower"],
            ordered["ci_robust_upper"] - ordered["coef_bc"],
        ],
        fmt="o-",
        color="#d62728",
        ecolor="#444444",
        elinewidth=1.2,
        capsize=4,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(ordered["x"])
    ax.set_xticklabels(ordered["post_period"], rotation=20)
    ax.set_ylabel("RD treatment effect on ozone change (ppb)")
    ax.set_xlabel("Post period")
    ax.set_title("Dynamic first stage around the 70 ppb cutoff")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_did_panel(panel: pd.DataFrame, base: pd.DataFrame, outcome: str) -> pd.DataFrame:
    treat_info = base[["county_fips", "nonattain_70"]].copy().rename(columns={"nonattain_70": "treated"})
    did = panel.loc[
        panel["Year"].between(2008, 2023),
        ["county_fips", "Year", outcome, "state_name", "county_name"],
    ].copy()
    did = did.merge(treat_info, on="county_fips", how="inner")
    did = did[did[outcome].notna()].copy()
    did["post_2018"] = (did["Year"] >= 2018).astype(int)
    did["treated"] = did["treated"].astype(int)
    did["year_str"] = did["Year"].astype(str)
    return did


def run_did_fe(panel_df: pd.DataFrame, outcome: str) -> pd.Series:
    model = smf.ols(
        f"{outcome} ~ treated:post_2018 + C(county_fips) + C(Year)",
        data=panel_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": panel_df["county_fips"]})
    coef = float(model.params["treated:post_2018"])
    se = float(model.bse["treated:post_2018"])
    return pd.Series(
        {
            "post_definition": "2018plus",
            "outcome": outcome,
            "coef": coef,
            "se": se,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "pvalue": float(model.pvalues["treated:post_2018"]),
            "n_obs": int(model.nobs),
            "n_counties": int(panel_df["county_fips"].nunique()),
        }
    )


def run_did_fe_custom_post(panel_df: pd.DataFrame, outcome: str, post_start_year: int) -> pd.Series:
    work = panel_df.copy()
    work["post_custom"] = (work["Year"] >= post_start_year).astype(int)
    model = smf.ols(
        f"{outcome} ~ treated:post_custom + C(county_fips) + C(Year)",
        data=work,
    ).fit(cov_type="cluster", cov_kwds={"groups": work["county_fips"]})
    coef = float(model.params["treated:post_custom"])
    se = float(model.bse["treated:post_custom"])
    return pd.Series(
        {
            "post_definition": f"{post_start_year}plus",
            "outcome": outcome,
            "coef": coef,
            "se": se,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "pvalue": float(model.pvalues["treated:post_custom"]),
            "n_obs": int(model.nobs),
            "n_counties": int(work["county_fips"].nunique()),
        }
    )


def run_event_study(panel_df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.Series]:
    work = panel_df.copy()
    event_terms = []
    for year in sorted(work["Year"].unique()):
        if year == 2017:
            continue
        col = f"event_{year}"
        work[col] = ((work["Year"] == year) & (work["treated"] == 1)).astype(int)
        event_terms.append(col)

    formula = f"{outcome} ~ C(county_fips) + C(Year) + " + " + ".join(event_terms)
    model = smf.ols(formula, data=work).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["county_fips"]},
    )

    rows = []
    pre_terms = []
    for year in sorted(work["Year"].unique()):
        if year == 2017:
            continue
        term = f"event_{year}"
        if term not in model.params.index:
            continue
        coef = float(model.params[term])
        se = float(model.bse[term])
        rows.append(
            {
                "year": int(year),
                "event_time": int(year - 2018),
                "coef": coef,
                "se": se,
                "ci_lower": coef - 1.96 * se,
                "ci_upper": coef + 1.96 * se,
                "is_pre": int(year < 2018),
            }
        )
        if year < 2017:
            pre_terms.append(term)

    event_df = pd.DataFrame(rows).sort_values("year")

    if pre_terms:
        beta = model.params[pre_terms].to_numpy()
        cov = model.cov_params().loc[pre_terms, pre_terms].to_numpy()
        stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
        pvalue = float(1 - chi2.cdf(stat, df=len(pre_terms)))
    else:
        stat = np.nan
        pvalue = np.nan

    summary = pd.Series(
        {
            "outcome": outcome,
            "pretrend_chi2": stat,
            "pretrend_df": len(pre_terms),
            "pretrend_pvalue": pvalue,
            "n_obs": int(model.nobs),
            "n_counties": int(panel_df["county_fips"].nunique()),
        }
    )
    return event_df, summary


def make_event_study_plot(event_df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        event_df["year"],
        event_df["coef"],
        yerr=[event_df["coef"] - event_df["ci_lower"], event_df["ci_upper"] - event_df["coef"]],
        fmt="o",
        color="#d62728",
        ecolor="#444444",
        capsize=4,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(2018, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Treated-control gap relative to 2017")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_hybrid_panel(panel: pd.DataFrame, base: pd.DataFrame, outcome: str, bandwidth: float) -> pd.DataFrame:
    keep = base.loc[base["running_var_70"].abs() <= bandwidth, ["county_fips", "running_var_70", "nonattain_70"]].copy()
    keep = keep.rename(columns={"nonattain_70": "treated"})
    hybrid = panel.loc[
        panel["Year"].between(2008, 2025),
        ["county_fips", "Year", outcome],
    ].copy()
    hybrid = hybrid.merge(keep, on="county_fips", how="inner")
    hybrid = hybrid[hybrid[outcome].notna()].copy()
    hybrid["post_2018"] = (hybrid["Year"] >= 2018).astype(int)
    hybrid["post_2018_2020"] = hybrid["Year"].between(2018, 2020).astype(int)
    hybrid["post_2022"] = (hybrid["Year"] >= 2022).astype(int)
    hybrid["treated"] = hybrid["treated"].astype(int)
    hybrid["rv_post_2018"] = hybrid["running_var_70"] * hybrid["post_2018"]
    hybrid["rv_post_2018_2020"] = hybrid["running_var_70"] * hybrid["post_2018_2020"]
    hybrid["rv_post_2022"] = hybrid["running_var_70"] * hybrid["post_2022"]
    hybrid["treat_rv_post_2018"] = hybrid["treated"] * hybrid["running_var_70"] * hybrid["post_2018"]
    hybrid["treat_rv_post_2018_2020"] = hybrid["treated"] * hybrid["running_var_70"] * hybrid["post_2018_2020"]
    hybrid["treat_rv_post_2022"] = hybrid["treated"] * hybrid["running_var_70"] * hybrid["post_2022"]
    return hybrid


def run_hybrid_fe(panel_df: pd.DataFrame, outcome: str, post_col: str) -> pd.Series:
    model = smf.ols(
        f"{outcome} ~ treated:{post_col} + rv_{post_col} + treat_rv_{post_col} + C(county_fips) + C(Year)",
        data=panel_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": panel_df["county_fips"]})
    coef_name = f"treated:{post_col}"
    coef = float(model.params[coef_name])
    se = float(model.bse[coef_name])
    return pd.Series(
        {
            "county_fe": 1,
            "post_definition": post_col,
            "coef": coef,
            "se": se,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "pvalue": float(model.pvalues[coef_name]),
            "n_obs": int(model.nobs),
            "n_counties": int(panel_df["county_fips"].nunique()),
        }
    )


def run_hybrid_no_county_fe(panel_df: pd.DataFrame, outcome: str, post_col: str) -> pd.Series:
    model = smf.ols(
        f"{outcome} ~ treated:{post_col} + rv_{post_col} + treat_rv_{post_col} + C(Year)",
        data=panel_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": panel_df["county_fips"]})
    coef_name = f"treated:{post_col}"
    coef = float(model.params[coef_name])
    se = float(model.bse[coef_name])
    return pd.Series(
        {
            "county_fe": 0,
            "post_definition": post_col,
            "coef": coef,
            "se": se,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "pvalue": float(model.pvalues[coef_name]),
            "n_obs": int(model.nobs),
            "n_counties": int(panel_df["county_fips"].nunique()),
        }
    )


def run_hybrid_event_study(panel_df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.Series]:
    work = panel_df.copy()
    event_terms = []
    slope_terms = []
    for year in sorted(work["Year"].unique()):
        if year == 2017:
            continue
        event_col = f"hybrid_event_{year}"
        slope_col = f"hybrid_slope_{year}"
        work[event_col] = ((work["Year"] == year) & (work["treated"] == 1)).astype(int)
        work[slope_col] = ((work["Year"] == year) * work["running_var_70"]).astype(float)
        event_terms.append(event_col)
        slope_terms.append(slope_col)

    formula = f"{outcome} ~ C(county_fips) + C(Year) + " + " + ".join(event_terms + slope_terms)
    model = smf.ols(formula, data=work).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["county_fips"]},
    )

    rows = []
    pre_terms = []
    for year in sorted(work["Year"].unique()):
        if year == 2017:
            continue
        term = f"hybrid_event_{year}"
        if term not in model.params.index:
            continue
        coef = float(model.params[term])
        se = float(model.bse[term])
        rows.append(
            {
                "year": int(year),
                "event_time": int(year - 2018),
                "coef": coef,
                "se": se,
                "ci_lower": coef - 1.96 * se,
                "ci_upper": coef + 1.96 * se,
                "is_pre": int(year < 2018),
            }
        )
        if year < 2017:
            pre_terms.append(term)

    event_df = pd.DataFrame(rows).sort_values("year")
    if pre_terms:
        beta = model.params[pre_terms].to_numpy()
        cov = model.cov_params().loc[pre_terms, pre_terms].to_numpy()
        stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
        pvalue = float(1 - chi2.cdf(stat, df=len(pre_terms)))
    else:
        stat = np.nan
        pvalue = np.nan

    summary = pd.Series(
        {
            "county_fe": 1,
            "pretrend_chi2": stat,
            "pretrend_df": len(pre_terms),
            "pretrend_pvalue": pvalue,
            "n_obs": int(model.nobs),
            "n_counties": int(panel_df["county_fips"].nunique()),
        }
    )
    return event_df, summary


def run_hybrid_event_study_no_county_fe(panel_df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, pd.Series]:
    work = panel_df.copy()
    event_terms = []
    slope_terms = []
    for year in sorted(work["Year"].unique()):
        if year == 2017:
            continue
        event_col = f"hybrid_nofe_event_{year}"
        slope_col = f"hybrid_nofe_slope_{year}"
        work[event_col] = ((work["Year"] == year) & (work["treated"] == 1)).astype(int)
        work[slope_col] = ((work["Year"] == year) * work["running_var_70"]).astype(float)
        event_terms.append(event_col)
        slope_terms.append(slope_col)

    formula = f"{outcome} ~ C(Year) + " + " + ".join(event_terms + slope_terms)
    model = smf.ols(formula, data=work).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["county_fips"]},
    )

    rows = []
    pre_terms = []
    for year in sorted(work["Year"].unique()):
        if year == 2017:
            continue
        term = f"hybrid_nofe_event_{year}"
        if term not in model.params.index:
            continue
        coef = float(model.params[term])
        se = float(model.bse[term])
        rows.append(
            {
                "year": int(year),
                "event_time": int(year - 2018),
                "coef": coef,
                "se": se,
                "ci_lower": coef - 1.96 * se,
                "ci_upper": coef + 1.96 * se,
                "is_pre": int(year < 2018),
            }
        )
        if year < 2017:
            pre_terms.append(term)

    event_df = pd.DataFrame(rows).sort_values("year")
    if pre_terms:
        beta = model.params[pre_terms].to_numpy()
        cov = model.cov_params().loc[pre_terms, pre_terms].to_numpy()
        stat = float(beta.T @ np.linalg.pinv(cov) @ beta)
        pvalue = float(1 - chi2.cdf(stat, df=len(pre_terms)))
    else:
        stat = np.nan
        pvalue = np.nan

    summary = pd.Series(
        {
            "county_fe": 0,
            "pretrend_chi2": stat,
            "pretrend_df": len(pre_terms),
            "pretrend_pvalue": pvalue,
            "n_obs": int(model.nobs),
            "n_counties": int(panel_df["county_fips"].nunique()),
        }
    )
    return event_df, summary


def make_hybrid_parallel_trends_plot(panel_df: pd.DataFrame, outcome: str, path: Path) -> None:
    means = (
        panel_df.groupby(["Year", "treated"])[outcome]
        .mean()
        .reset_index()
        .pivot(index="Year", columns="treated", values=outcome)
        .rename(columns={0: "control", 1: "treated"})
    )
    for col in ["control", "treated"]:
        means[f"{col}_norm"] = means[col] - means.loc[2017, col]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(means.index, means["control_norm"], marker="o", color="#1f77b4", label="Control, within RD bw")
    ax.plot(means.index, means["treated_norm"], marker="o", color="#d62728", label="Treated, within RD bw")
    ax.axvline(2018, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="#555555", linewidth=1, alpha=0.5)
    ax.set_title("Hybrid RD-DID local pretrend check within main RD bandwidth")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{outcome} relative to 2017")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_hybrid_parallel_trends_comparison_plot(panel_df: pd.DataFrame, outcome: str, path: Path) -> None:
    means = (
        panel_df.groupby(["Year", "treated"])[outcome]
        .mean()
        .reset_index()
        .pivot(index="Year", columns="treated", values=outcome)
        .rename(columns={0: "control", 1: "treated"})
    )
    means["gap"] = means["treated"] - means["control"]
    means["gap_norm"] = means["gap"] - means.loc[2017, "gap"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(means.index, means["gap_norm"], marker="o", color="#7f3c8d")
    ax.axvline(2018, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="#555555", linewidth=1, alpha=0.5)
    ax.set_title("Hybrid RD-DID local treated-control gap within main RD bandwidth")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Treated-control gap in {outcome}, relative to 2017")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_hybrid_rd_style_plot(panel_df: pd.DataFrame, outcome: str, post_col: str, county_fe: bool, path: Path, title: str) -> None:
    work = panel_df.copy()
    if county_fe:
        formula = f"{outcome} ~ C(county_fips) + C(Year)"
    else:
        formula = f"{outcome} ~ C(Year)"
    base_model = smf.ols(formula, data=work).fit()
    work["resid_outcome"] = base_model.resid

    pre = (
        work.loc[work[post_col] == 0]
        .groupby(["county_fips", "running_var_70"])["resid_outcome"]
        .mean()
        .reset_index(name="pre_mean")
    )
    post = (
        work.loc[work[post_col] == 1]
        .groupby(["county_fips", "running_var_70"])["resid_outcome"]
        .mean()
        .reset_index(name="post_mean")
    )
    plot_df = pre.merge(post, on=["county_fips", "running_var_70"], how="inner")
    plot_df["delta_resid"] = plot_df["post_mean"] - plot_df["pre_mean"]

    plot_df["treat"] = (plot_df["running_var_70"] > 0).astype(int)
    plot_df["interact"] = plot_df["treat"] * plot_df["running_var_70"]
    fit = smf.ols("delta_resid ~ treat + running_var_70 + interact", data=plot_df).fit(cov_type="HC1")

    left = plot_df[plot_df["running_var_70"] < 0].copy()
    right = plot_df[plot_df["running_var_70"] >= 0].copy()
    n_bins_per_side = min(20, max(6, int(np.sqrt(len(plot_df)) // 2)))
    left_bins = np.linspace(plot_df["running_var_70"].min(), 0, n_bins_per_side + 1)
    right_bins = np.linspace(0, plot_df["running_var_70"].max(), n_bins_per_side + 1)
    left["bin"] = pd.cut(left["running_var_70"], bins=left_bins, include_lowest=True)
    right["bin"] = pd.cut(right["running_var_70"], bins=right_bins, include_lowest=True)
    left_pts = left.groupby("bin", observed=True).agg(x=("running_var_70", "mean"), y=("delta_resid", "mean")).dropna()
    right_pts = right.groupby("bin", observed=True).agg(x=("running_var_70", "mean"), y=("delta_resid", "mean")).dropna()

    x_l = np.linspace(plot_df["running_var_70"].min(), 0, 200)
    x_r = np.linspace(0, plot_df["running_var_70"].max(), 200)
    y_l = fit.params["Intercept"] + fit.params["running_var_70"] * x_l
    y_r = (
        fit.params["Intercept"]
        + fit.params["treat"]
        + (fit.params["running_var_70"] + fit.params["interact"]) * x_r
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(left_pts["x"], left_pts["y"], color="#1f77b4", s=30, label="Binned residualized change")
    ax.scatter(right_pts["x"], right_pts["y"], color="#1f77b4", s=30)
    ax.plot(x_l, y_l, color="#d62728", linewidth=2.2, label="Local linear fit")
    ax.plot(x_r, y_r, color="#d62728", linewidth=2.2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Running variable: design value - 70 ppb")
    ax.set_ylabel("Residualized post-pre ozone change")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_density_test(base: pd.DataFrame) -> pd.Series:
    patch_pandas_for_rddensity()
    try:
        result = rddensity(base["running_var_70"].to_numpy(), c=0)
        return pd.Series(
            {
                "n": len(base),
                "bandwidth_left": float(result.h["left"]),
                "bandwidth_right": float(result.h["right"]),
                "density_left": float(result.hat["left"]),
                "density_right": float(result.hat["right"]),
                "density_diff": float(result.hat["diff"]),
                "test_stat_jk": float(result.test["t_jk"]),
                "pvalue_jk": float(result.test["p_jk"]),
                "density_test_error": "",
            }
        )
    except Exception as exc:
        return pd.Series(
            {
                "n": len(base),
                "bandwidth_left": np.nan,
                "bandwidth_right": np.nan,
                "density_left": np.nan,
                "density_right": np.nan,
                "density_diff": np.nan,
                "test_stat_jk": np.nan,
                "pvalue_jk": np.nan,
                "density_test_error": str(exc),
            }
        )


def write_summary(
    base: pd.DataFrame,
    main_row: pd.Series,
    long_run_row: pd.Series,
    density_row: pd.Series,
    did_row: pd.Series,
    did_2022_row: pd.Series,
    pretrend_row: pd.Series,
    hybrid_row: pd.Series,
    hybrid_2022_row: pd.Series,
    hybrid_pretrend_row: pd.Series,
    strongest_row: pd.Series,
    panel: pd.DataFrame,
    data_end_year: int,
    output_path: Path,
) -> None:
    full_2016 = panel.loc[panel["Year"] == DESIGNATION_YEAR].copy()
    full_2016 = full_2016[full_2016["design_value_3yr"].notna()].copy()
    pre2015_flags = build_historical_treatment_flags(panel)
    full_2016 = full_2016.merge(pre2015_flags, on="county_fips", how="left")
    treated_pre2015_n = int(full_2016["treated_pre_2015"].fillna(0).sum())
    treated_2016_nonattain75_n = int((full_2016["nonattain_75"] == 1).sum())
    below75_2010_n = int(full_2016["below75_in_2010"].fillna(0).sum())
    right_of_cutoff_n = int((base["running_var_70"] >= 0).sum())
    left_of_cutoff_n = int((base["running_var_70"] < 0).sum())
    screen_mode = str(base["screen_mode"].iloc[0]) if len(base) else "unknown"

    lines = [
        "# First-stage RD analysis summary",
        "",
        f"- Designation sample year: {DESIGNATION_YEAR} (design value based on 2014-2016 ozone)",
        f"- Active screening rule: `{screen_mode}`",
        f"- Restricted sample: {int(base.shape[0])} counties in the active screen with non-missing {DESIGNATION_YEAR} design value",
        f"- Baseline 2008-standard screen: `below75_in_2010 = 1` if the 2010 design value, based on 2008-2010 ozone, is at or below 75 ppb",
        f"- Stricter comparison flag: `treated_pre_2015 = 1` if a county ever had `design_value_3yr > 75` in any year from 2010 through 2014",
        f"- Additional proxy flag saved in the sample: `nonattain_1997_proxy = 1` if a county ever had `design_value_3yr > 84` before 2015",
        f"- Counties with `below75_in_2010 == 1`: {below75_2010_n}",
        f"- Counties excluded by `treated_pre_2015 == 1`: {treated_pre2015_n}",
        f"- Counties with `nonattain_75 == 1` in the {DESIGNATION_YEAR} row only: {treated_2016_nonattain75_n}",
        f"- Counties remaining below 70 in the analysis sample: {left_of_cutoff_n}",
        f"- Counties remaining at or above 70 in the analysis sample: {right_of_cutoff_n}",
        f"- Available county panel years: 2008-{data_end_year}",
        "- Headline RD plot and main cross-sectional result now use the pre-COVID post-period `2018-2020`.",
        "- Later post windows are still reported in the dynamic and spec-exploration outputs, including a longer-run `2022-2025` window.",
        "",
        "## Main first-stage result",
        "",
        f"- Bias-corrected RD estimate: {main_row['coef_bc']:.3f} ppb",
        f"- Robust 95% CI: [{main_row['ci_robust_lower']:.3f}, {main_row['ci_robust_upper']:.3f}]",
        f"- Robust p-value: {main_row['pvalue_robust']:.3f}",
        f"- MSE-optimal bandwidths: left={main_row['bandwidth_left']:.3f}, right={main_row['bandwidth_right']:.3f}",
        "",
        "## Long-run RD",
        "",
        f"- `2022-2025` bias-corrected RD estimate: {long_run_row['coef_bc']:.3f} ppb",
        f"- `2022-2025` robust 95% CI: [{long_run_row['ci_robust_lower']:.3f}, {long_run_row['ci_robust_upper']:.3f}]",
        f"- `2022-2025` robust p-value: {long_run_row['pvalue_robust']:.3f}",
        "",
        "## DID extension",
        "",
        f"- Two-way fixed-effects DID estimate on annual county ozone with post starting in 2018: {did_row['coef']:.3f} ppb",
        f"- DID 2018+ 95% CI: [{did_row['ci_lower']:.3f}, {did_row['ci_upper']:.3f}]",
        f"- DID 2018+ p-value: {did_row['pvalue']:.3f}",
        f"- Two-way fixed-effects DID estimate with post starting in 2022: {did_2022_row['coef']:.3f} ppb",
        f"- DID 2022+ 95% CI: [{did_2022_row['ci_lower']:.3f}, {did_2022_row['ci_upper']:.3f}]",
        f"- DID 2022+ p-value: {did_2022_row['pvalue']:.3f}",
        f"- Event-study pretrend test p-value: {pretrend_row['pretrend_pvalue']:.3f}",
        "",
        "## Hybrid RD-DID",
        "",
        f"- Local hybrid FE estimate within the main RD bandwidth, post 2018: {hybrid_row['coef']:.3f} ppb",
        f"- Hybrid 2018+ 95% CI: [{hybrid_row['ci_lower']:.3f}, {hybrid_row['ci_upper']:.3f}]",
        f"- Hybrid 2018+ p-value: {hybrid_row['pvalue']:.3f}",
        f"- Local hybrid FE estimate within the main RD bandwidth, post 2022: {hybrid_2022_row['coef']:.3f} ppb",
        f"- Hybrid 2022+ 95% CI: [{hybrid_2022_row['ci_lower']:.3f}, {hybrid_2022_row['ci_upper']:.3f}]",
        f"- Hybrid 2022+ p-value: {hybrid_2022_row['pvalue']:.3f}",
        f"- Hybrid local event-study pretrend test p-value: {hybrid_pretrend_row['pretrend_pvalue']:.3f}",
        "",
        "## Exploratory spec search",
        "",
        "- These are exploratory robustness checks, not a license to cherry-pick.",
        f"- Strongest negative estimate with p < 0.10 in the saved spec grid: `{strongest_row['spec']}`",
        f"- That spec's estimate is {strongest_row['coef_bc']:.3f} with p-value {strongest_row['pvalue_robust']:.3f}",
        "",
        "## Density test",
        "",
    ]

    if pd.notna(density_row["test_stat_jk"]):
        lines.extend(
            [
                f"- Jackknife test statistic: {density_row['test_stat_jk']:.3f}",
                f"- Jackknife p-value: {density_row['pvalue_jk']:.3f}",
            ]
        )
    else:
        lines.extend(
            [
                "- Density test did not run successfully in this environment because the Python `rddensity` package hit a pandas compatibility bug.",
                f"- Package error: {density_row['density_test_error']}",
            ]
        )

    lines.extend(
        [
        "",
        "## Limits in the supplied data",
        "",
        "- The county covariates you listed for balance checks are not present in the provided CSVs, so those tests are not included here.",
        "- The RD plot uses binned averages within the selected bandwidth, so the number of plotted dots is much smaller than the number of county observations by design.",
        "- County and year fixed effects are not part of this cross-sectional RD-in-changes specification; they would belong to a panel DID/RD hybrid, not this one-equation local linear RD.",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    panel, daily = load_data()
    base = build_designation_sample(panel, screen_mode="baseline_2010_below75")
    comparison_base = build_designation_sample(panel, screen_mode="never_above75_pre2015")
    base.to_csv(OUTPUT_DIR / "designation_sample_with_flags.csv", index=False)
    comparison_base.to_csv(OUTPUT_DIR / "designation_sample_never_above75_pre2015.csv", index=False)

    pre_years = [2014, 2015, 2016]
    dynamic_posts = [
        [2017, 2018, 2019],
        [2018, 2019, 2020],
        [2019, 2020, 2021],
        [2020, 2021, 2022],
        [2021, 2022, 2023],
        [2022, 2023, 2024, 2025],
    ]

    dynamic_rows = []
    dynamic_samples: dict[str, pd.DataFrame] = {}

    for post_years in dynamic_posts:
        sample = build_delta_sample(base, panel, "Year", "county_4thmax_ppb", pre_years, post_years)
        result = run_rd(sample)
        post_label = f"{post_years[0]}-{post_years[-1]}"
        dynamic_samples[post_label] = sample
        dynamic_rows.append(
            extract_rd_row(
                result=result,
                label=f"dynamic_{post_label}",
                sample=sample,
                pre_label="2014-2016",
                post_label=post_label,
                outcome_name="county_4thmax_ppb",
            )
        )

    dynamic_df = pd.DataFrame(dynamic_rows)
    dynamic_df.to_csv(OUTPUT_DIR / "rd_dynamic_results.csv", index=False)

    main_post_label = "2018-2020"
    main_sample = dynamic_samples[main_post_label]
    main_result = run_rd(main_sample)
    main_df = dynamic_df[dynamic_df["post_period"] == main_post_label].copy()
    main_df.to_csv(OUTPUT_DIR / "rd_main_result.csv", index=False)
    long_run_df = dynamic_df[dynamic_df["post_period"] == "2022-2025"].copy()
    long_run_df.to_csv(OUTPUT_DIR / "rd_long_run_result.csv", index=False)

    alt_rows = []
    alt_plot_samples: dict[str, tuple[pd.DataFrame, object, str]] = {}
    for outcome in ["mean_daily_max", "exceedance_days_70", "p95_daily_max", "season_mean"]:
        sample = build_delta_sample(base, daily, "year", outcome, pre_years, [2018, 2019, 2020])
        result = run_rd(sample)
        alt_plot_samples[outcome] = (sample, result, "2018-2020")
        alt_rows.append(
            extract_rd_row(
                result=result,
                label=f"alt_{outcome}",
                sample=sample,
                pre_label="2014-2016",
                post_label="2018-2020",
                outcome_name=outcome,
            )
        )
    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(OUTPUT_DIR / "rd_alternative_outcomes.csv", index=False)

    spec_df = run_rd_spec_exploration(panel)
    spec_df.to_csv(OUTPUT_DIR / "rd_spec_exploration.csv", index=False)
    sig_negative = spec_df[(spec_df["coef_bc"] < 0) & (spec_df["pvalue_robust"] < 0.10)].copy()
    if len(sig_negative):
        strongest_row = sig_negative.sort_values("coef_bc").iloc[0]
    else:
        strongest_row = pd.Series({"spec": "none", "coef_bc": np.nan, "pvalue_robust": np.nan})

    density_row = run_density_test(base)
    density_row.to_frame().T.to_csv(OUTPUT_DIR / "rddensity_test.csv", index=False)

    did_panel = build_did_panel(panel, base, "county_4thmax_ppb")
    did_row = run_did_fe(did_panel, "county_4thmax_ppb")
    did_2022_row = run_did_fe_custom_post(did_panel, "county_4thmax_ppb", 2022)
    pd.DataFrame([did_row.to_dict(), did_2022_row.to_dict()]).to_csv(OUTPUT_DIR / "did_results.csv", index=False)
    event_df, pretrend_row = run_event_study(did_panel, "county_4thmax_ppb")
    event_df.to_csv(OUTPUT_DIR / "did_event_study.csv", index=False)
    pretrend_row.to_frame().T.to_csv(OUTPUT_DIR / "did_pretrend_test.csv", index=False)

    hybrid_panel = build_hybrid_panel(panel, base, "county_4thmax_ppb", float(main_df.iloc[0]["bandwidth_left"]))
    hybrid_2018_2020_row_nofe = run_hybrid_no_county_fe(hybrid_panel, "county_4thmax_ppb", "post_2018_2020")
    hybrid_row_nofe = run_hybrid_no_county_fe(hybrid_panel, "county_4thmax_ppb", "post_2018")
    hybrid_2022_row_nofe = run_hybrid_no_county_fe(hybrid_panel, "county_4thmax_ppb", "post_2022")
    hybrid_2018_2020_row = run_hybrid_fe(hybrid_panel, "county_4thmax_ppb", "post_2018_2020")
    hybrid_row = run_hybrid_fe(hybrid_panel, "county_4thmax_ppb", "post_2018")
    hybrid_2022_row = run_hybrid_fe(hybrid_panel, "county_4thmax_ppb", "post_2022")
    pd.DataFrame(
        [
            hybrid_2018_2020_row_nofe.to_dict(),
            hybrid_row_nofe.to_dict(),
            hybrid_2022_row_nofe.to_dict(),
            hybrid_2018_2020_row.to_dict(),
            hybrid_row.to_dict(),
            hybrid_2022_row.to_dict(),
        ]
    ).to_csv(
        OUTPUT_DIR / "hybrid_rd_did_results.csv", index=False
    )
    hybrid_event_df_nofe, hybrid_pretrend_row_nofe = run_hybrid_event_study_no_county_fe(hybrid_panel, "county_4thmax_ppb")
    hybrid_event_df, hybrid_pretrend_row = run_hybrid_event_study(hybrid_panel, "county_4thmax_ppb")
    pd.concat(
        [
            hybrid_event_df_nofe.assign(county_fe=0),
            hybrid_event_df.assign(county_fe=1),
        ],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "hybrid_rd_did_event_study.csv", index=False)
    pd.DataFrame([hybrid_pretrend_row_nofe.to_dict(), hybrid_pretrend_row.to_dict()]).to_csv(
        OUTPUT_DIR / "hybrid_rd_did_pretrend_test.csv", index=False
    )

    make_rd_plot(
        main_sample,
        main_result,
        OUTPUT_DIR / "rd_main_plot.png",
        "RD plot: change in county 4th-highest ozone, 2014-2016 to 2018-2020",
    )
    mean_sample, mean_result, mean_post = alt_plot_samples["mean_daily_max"]
    make_rd_plot_from_column(
        mean_sample,
        mean_result,
        "delta_outcome",
        "Change in annual mean daily max ozone (ppb)",
        OUTPUT_DIR / "rd_mean_daily_max_plot.png",
        f"RD plot: change in annual mean daily max ozone, 2014-2016 to {mean_post}",
    )
    season_sample, season_result, season_post = alt_plot_samples["season_mean"]
    make_rd_plot_from_column(
        season_sample,
        season_result,
        "delta_outcome",
        "Change in ozone-season mean ozone (ppb)",
        OUTPUT_DIR / "rd_season_mean_plot.png",
        f"RD plot: change in ozone-season mean ozone, 2014-2016 to {season_post}",
    )
    p95_sample, p95_result, p95_post = alt_plot_samples["p95_daily_max"]
    make_rd_plot_from_column(
        p95_sample,
        p95_result,
        "delta_outcome",
        "Change in 95th percentile daily max ozone (ppb)",
        OUTPUT_DIR / "rd_p95_plot.png",
        f"RD plot: change in 95th percentile daily max ozone, 2014-2016 to {p95_post}",
    )
    make_dynamic_plot(dynamic_df, OUTPUT_DIR / "rd_dynamic_plot.png")
    make_parallel_trends_plot(did_panel, "county_4thmax_ppb", OUTPUT_DIR / "did_parallel_trends_plot.png")
    make_parallel_trends_raw_plot(
        did_panel,
        "county_4thmax_ppb",
        OUTPUT_DIR / "did_parallel_trends_raw_plot.png",
    )
    make_event_study_plot(
        event_df,
        OUTPUT_DIR / "did_event_study_plot.png",
        "Event study with county and year fixed effects",
    )
    make_hybrid_parallel_trends_plot(
        hybrid_panel,
        "county_4thmax_ppb",
        OUTPUT_DIR / "hybrid_rd_did_parallel_trends_plot.png",
    )
    make_hybrid_parallel_trends_comparison_plot(
        hybrid_panel,
        "county_4thmax_ppb",
        OUTPUT_DIR / "hybrid_rd_did_gap_plot.png",
    )
    make_hybrid_rd_style_plot(
        hybrid_panel,
        "county_4thmax_ppb",
        "post_2018_2020",
        False,
        OUTPUT_DIR / "hybrid_rd_style_plot_nocountyfe.png",
        "Hybrid RD-DID style plot, 2018-2020 post, no county FE",
    )
    make_hybrid_rd_style_plot(
        hybrid_panel,
        "county_4thmax_ppb",
        "post_2018_2020",
        True,
        OUTPUT_DIR / "hybrid_rd_style_plot_countyfe.png",
        "Hybrid RD-DID style plot, 2018-2020 post, with county FE",
    )
    make_event_study_plot(
        hybrid_event_df_nofe,
        OUTPUT_DIR / "hybrid_rd_did_event_study_plot_nocountyfe.png",
        "Hybrid RD-DID event study within main RD bandwidth, no county FE",
    )
    make_event_study_plot(
        hybrid_event_df,
        OUTPUT_DIR / "hybrid_rd_did_event_study_plot.png",
        "Hybrid RD-DID event study within main RD bandwidth",
    )

    write_summary(
        base=base,
        main_row=main_df.iloc[0],
        long_run_row=long_run_df.iloc[0],
        density_row=density_row,
        did_row=did_row,
        did_2022_row=did_2022_row,
        pretrend_row=pretrend_row,
        hybrid_row=hybrid_row,
        hybrid_2022_row=hybrid_2022_row,
        hybrid_pretrend_row=hybrid_pretrend_row,
        strongest_row=strongest_row,
        panel=panel,
        data_end_year=int(panel["Year"].max()),
        output_path=OUTPUT_DIR / "analysis_summary.md",
    )


if __name__ == "__main__":
    main()
