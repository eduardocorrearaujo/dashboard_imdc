"""Basic smoke and data-contract tests; run with pytest -q after installing requirements."""

import pandas as pd
import pytest

from app import (
    REGIONS,
    accumulated_figure,
    build_cumulative_table,
    observed_series,
    preceding_season,
    read_data,
    weekly_figure,
)


@pytest.mark.parametrize("disease", ["dengue", "chik"])
def test_inputs_dates_and_national_data(disease):
    weekly, totals, observed = read_data(disease)
    assert weekly.loc[weekly.uf.eq("BR"), "date"].nunique() == 52
    assert "BR" in totals.uf.values
    assert "ES" not in weekly.uf.values
    assert "ES" in observed.uf.values
    assert weekly.date.min() == pd.Timestamp("2026-10-11")
    assert weekly.date.max() == pd.Timestamp("2027-10-03")


@pytest.mark.parametrize("disease", ["dengue", "chik"])
def test_national_observed_is_sum_of_all_states(disease):
    _, _, observed = read_data(disease)
    br = observed_series(observed, "BR")
    date = pd.Timestamp("2026-10-11")
    assert br.loc[br.date.eq(date), "casos"].iloc[0] == pytest.approx(
        observed.loc[observed.date.eq(date), "casos"].sum()
    )


@pytest.mark.parametrize("disease", ["dengue", "chik"])
def test_cumulative_uses_original_ensemble_quantiles(disease):
    weekly, totals, observed = read_data(disease)
    table = build_cumulative_table(totals, observed, weekly.date.min(), 52)
    br = table.set_index("uf").loc["BR"]
    expected = totals.set_index("uf").loc["BR"]
    for col in ("pred", "lower_80", "upper_80", "lower_90", "upper_90"):
        assert br[col] == pytest.approx(expected[col])
    first, last = preceding_season(weekly.date.min(), 52)
    assert first == pd.Timestamp("2025-10-12")
    assert last == pd.Timestamp("2026-10-04")
    assert br["observed_prev"] == pytest.approx(
        observed.loc[observed.date.between(first, last), "casos"].sum()
    )
    assert pd.isna(table.set_index("uf").loc["ES", "pred"])


@pytest.mark.parametrize("disease", ["dengue", "chik"])
def test_weekly_and_cumulative_plots_render(disease):
    weekly, totals, observed = read_data(disease)
    table = build_cumulative_table(totals, observed, weekly.date.min(), 52)
    fig = weekly_figure(
        REGIONS["Sudeste"], weekly, observed, weekly.date.min(), 52,
        True, [80, 90], "#1978a3", columns=3,
    )
    assert fig.data
    assert fig.to_json()
    fig_bar = accumulated_figure(table, REGIONS["Sul"], "#1978a3")
    assert len(fig_bar.data) == 4
    assert fig_bar.to_json()
