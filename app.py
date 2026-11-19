"""Dashboard de previsões epidemiológicas — dengue e chikungunya.

Execute com: streamlit run app.py
Os sete CSVs .gz devem estar dentro de ./data/ (ou DASH_DATA_DIR).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DASH_DATA_DIR", BASE_DIR / "data"))

REGIONS = {
    "Sul": ["PR", "SC", "RS"],
    "Sudeste": ["SP", "MG", "RJ", "ES"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
}

DISEASES = {
    "Dengue": {"key": "dengue", "color": "#1978a3"},
    "Chikungunya": {"key": "chik", "color": "#a6539d"},
}

MEDIUM_TERM_COLOR = "#d97732"  # Laranja, contrastando com o ensemble de dengue.
OBS_COLOR = "#111111"
OBS_BAR_COLOR = "#9aabba"  # Cinza: observado na temporada anterior.
OBS_CURRENT_BAR_COLOR = "#14866d"  # Verde: observado na temporada atual.
GRID_COLOR = "rgba(101, 119, 142, 0.15)"
TEXT_COLOR = "#183147"

WEEKLY_COLUMNS = {"date", "uf", "lower_90", "lower_80", "pred", "upper_80", "upper_90"}
TOTAL_COLUMNS = WEEKLY_COLUMNS - {"date"}
OBS_COLUMNS = {"date", "uf", "casos"}
QUANTILES = ["lower_90", "lower_80", "pred", "upper_80", "upper_90"]
MEDIUM_TERM_COLUMNS = WEEKLY_COLUMNS | {"lower_50", "upper_50", "lower_95", "upper_95"}
MEDIUM_TERM_QUANTILES = [
    "lower_95", "lower_90", "lower_80", "lower_50", "pred",
    "upper_50", "upper_80", "upper_90", "upper_95",
]


def fmt_num(value: float | int | None) -> str:
    """Thousands separator used in Brazilian Portuguese, without decimals."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.0f}".replace(",", ".")


def hex_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def check_columns(df: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: faltam colunas: {', '.join(sorted(missing))}.")


def csv_fingerprint(path: Path) -> str:
    """Hash the CSV bytes so a changed file always gets a new cache key.

    A content hash works even when the filename, file size, and timestamps
    remain unchanged. CSVs in this project are small compressed files.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_paths(disease_key: str) -> dict[str, Path]:
    return {
        "weekly": DATA_DIR / f"ensemble_weekly_2027_{disease_key}.csv.gz",
        "totals": DATA_DIR / f"ensemble_total_2027_{disease_key}.csv.gz",
        "observed": DATA_DIR / f"{disease_key}_casprov.csv.gz",
    }


def read_data(disease_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reload cached data whenever any of this disease's CSV files changes."""
    files = dataset_paths(disease_key)
    file_versions = tuple(csv_fingerprint(path) for path in files.values())
    return _read_data_cached(disease_key, file_versions)


@st.cache_data(show_spinner=False, max_entries=4)
def _read_data_cached(
    disease_key: str, file_versions: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cache by disease and CSV hashes; file_versions is part of the cache key."""
    files = dataset_paths(disease_key)
    weekly = pd.read_csv(files["weekly"], parse_dates=["date"])
    totals = pd.read_csv(files["totals"])
    observed = pd.read_csv(files["observed"], parse_dates=["date"])
    check_columns(weekly, WEEKLY_COLUMNS, files["weekly"].name)
    check_columns(totals, TOTAL_COLUMNS, files["totals"].name)
    check_columns(observed, OBS_COLUMNS, files["observed"].name)

    for frame in (weekly, totals, observed):
        frame["uf"] = frame["uf"].astype(str).str.strip().str.upper()

    for frame in (weekly, totals):
        frame[QUANTILES] = frame[QUANTILES].apply(pd.to_numeric, errors="raise")
        if frame[QUANTILES].isna().any().any():
            raise ValueError("Foram encontrados quantis vazios nas previsões.")
        if not (frame[QUANTILES].diff(axis=1).iloc[:, 1:] >= -1e-8).all().all():
            raise ValueError("Os intervalos de previsão não estão ordenados corretamente.")
    observed["casos"] = pd.to_numeric(observed["casos"], errors="raise")

    if weekly.duplicated(["uf", "date"]).any():
        raise ValueError("Há previsões duplicadas para a mesma UF e data.")
    if observed.duplicated(["uf", "date"]).any():
        raise ValueError("Há observações duplicadas para a mesma UF e data.")
    if totals.duplicated(["uf"]).any():
        raise ValueError("Há acumulados previstos duplicados para a mesma UF.")
    if not {"BR"}.issubset(weekly["uf"]) or not {"BR"}.issubset(totals["uf"]):
        raise ValueError("Os arquivos de previsão devem conter uma linha/série para BR.")

    return (
        weekly.sort_values(["uf", "date"]).reset_index(drop=True),
        totals.sort_values("uf").reset_index(drop=True),
        observed.sort_values(["uf", "date"]).reset_index(drop=True),
    )


def read_medium_term_data() -> pd.DataFrame:
    """Reload dengue medium-term forecasts when their CSV contents change."""
    path = DATA_DIR / "medium_term_dengue_preds.csv.gz"
    return _read_medium_term_data_cached(csv_fingerprint(path))


@st.cache_data(show_spinner=False, max_entries=4)
def _read_medium_term_data_cached(file_version: str) -> pd.DataFrame:
    """Cache by the compressed file's content hash (file_version)."""
    path = DATA_DIR / "medium_term_dengue_preds.csv.gz"
    medium = pd.read_csv(path, parse_dates=["date"])
    check_columns(medium, MEDIUM_TERM_COLUMNS, path.name)
    medium["uf"] = medium["uf"].astype(str).str.strip().str.upper()
    medium[MEDIUM_TERM_QUANTILES] = medium[MEDIUM_TERM_QUANTILES].apply(
        pd.to_numeric, errors="raise"
    )
    if medium["date"].isna().any() or medium[MEDIUM_TERM_QUANTILES].isna().any().any():
        raise ValueError(f"{path.name}: foram encontrados valores ou datas ausentes.")
    if (medium[MEDIUM_TERM_QUANTILES] < 0).any().any():
        raise ValueError(f"{path.name}: há previsões negativas.")
    if not (medium[MEDIUM_TERM_QUANTILES].diff(axis=1).iloc[:, 1:] >= -1e-8).all().all():
        raise ValueError(f"{path.name}: os quantis estão fora de ordem.")
    if medium.duplicated(["uf", "date"]).any():
        raise ValueError(f"{path.name}: há datas duplicadas para uma UF.")
    if "BR" not in medium["uf"].values:
        raise ValueError(f"{path.name}: a previsão nacional BR está ausente.")
    return medium.sort_values(["uf", "date"]).reset_index(drop=True)


def observed_series(observed: pd.DataFrame, uf: str) -> pd.DataFrame:
    """BR is the sum across all 27 observed UFs; state uses its own data."""
    if uf == "BR":
        return observed.groupby("date", as_index=False, sort=True)["casos"].sum()
    return observed.loc[observed["uf"].eq(uf), ["date", "casos"]].sort_values("date")


def observed_national_current_season(
    observed: pd.DataFrame, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp
) -> float:
    """Observed BR cases in the current 2026–2027 season (from epiweek 202641)."""
    national = observed_series(observed, "BR")
    return float(national.loc[national["date"].between(forecast_start, forecast_end), "casos"].sum())


def preceding_season(forecast_start: pd.Timestamp, n_weeks: int = 52) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The n Sundays immediately before the forecast's first Sunday."""
    return forecast_start - pd.Timedelta(weeks=n_weeks), forecast_start - pd.Timedelta(weeks=1)


def build_cumulative_table(
    totals: pd.DataFrame, observed: pd.DataFrame, forecast_start: pd.Timestamp, n_weeks: int
) -> pd.DataFrame:
    """Aggregate observed cases separately by season, retaining ensemble quantiles."""
    prev_start, prev_end = preceding_season(forecast_start, n_weeks)
    forecast_end = forecast_start + pd.Timedelta(weeks=n_weeks - 1)
    previous_observed = observed.loc[observed["date"].between(prev_start, prev_end)]
    current_observed = observed.loc[observed["date"].between(forecast_start, forecast_end)]

    previous_by_state = previous_observed.groupby("uf")["casos"].sum()
    current_by_state = current_observed.groupby("uf")["casos"].sum()

    order = ["BR"] + [uf for ufs in REGIONS.values() for uf in ufs]
    base = pd.DataFrame({"uf": order})
    table = base.merge(totals, on="uf", how="left", validate="one_to_one")
    table["observed_prev"] = table["uf"].map(previous_by_state)
    table["observed_current"] = table["uf"].map(current_by_state)
    # BR é a soma das 27 UFs observadas (inclusive ES), sem duplicar linhas BR.
    # Uma temporada sem observações fica como ausente, não como zero.
    if not previous_observed.empty:
        table.loc[table["uf"].eq("BR"), "observed_prev"] = float(previous_by_state.sum())
    if not current_observed.empty:
        table.loc[table["uf"].eq("BR"), "observed_current"] = float(current_by_state.sum())
    table["region"] = table["uf"].map(
        {uf: region for region, states in REGIONS.items() for uf in states}
    ).fillna("Brasil")
    return table


def add_bands(
    fig: go.Figure,
    forecast: pd.DataFrame,
    row: int,
    col: int,
    color: str,
    intervals: Iterable[int],
    showlegend: bool,
    model_name: str = "Ensemble",
    opacity_factor: float = 1.0,
) -> None:
    bands = [(90, 0.105), (80, 0.20)]
    for coverage, alpha in bands:
        if coverage not in intervals:
            continue
        lower, upper = f"lower_{coverage}", f"upper_{coverage}"
        fig.add_trace(
            go.Scatter(
                x=forecast["date"], y=forecast[lower], mode="lines",
                line={"width": 0}, hoverinfo="skip", showlegend=False,
                legendgroup=f"{model_name}_ip{coverage}",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["date"], y=forecast[upper], mode="lines",
                line={"width": 0}, fill="tonexty",
                fillcolor=hex_rgba(color, alpha * opacity_factor),
                name=f"{model_name} · IP {coverage}%",
                legendgroup=f"{model_name}_ip{coverage}",
                showlegend=showlegend, hoverinfo="skip",
                legendrank=20 if coverage == 80 else 30,
            ),
            row=row, col=col,
        )


def weekly_figure(
    ufs: list[str],
    weekly: pd.DataFrame,
    observed: pd.DataFrame,
    forecast_start: pd.Timestamp,
    n_history_weeks: int,
    include_partial_observed: bool,
    intervals: Iterable[int],
    color: str,
    columns: int = 3,
    medium_term: pd.DataFrame | None = None,
    medium_term_intervals: bool = False,
) -> go.Figure:
    """Small multiples: ensemble and optionally dengue medium-term forecasts."""
    ncols = min(columns, len(ufs))
    nrows = math.ceil(len(ufs) / ncols)
    titles = ["Brasil (27 UFs)" if uf == "BR" else uf for uf in ufs]
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=titles,
        horizontal_spacing=0.075 if ncols > 1 else 0.03,
        vertical_spacing=min(0.18, 0.22 / max(nrows, 1)),
    )
    min_date = forecast_start - pd.Timedelta(weeks=n_history_weeks)

    for i, uf in enumerate(ufs):
        row, col = divmod(i, ncols)
        row += 1
        col += 1
        showlegend = i == 0
        forecast = weekly.loc[weekly["uf"].eq(uf)].sort_values("date")
        medium_forecast = (
            medium_term.loc[medium_term["uf"].eq(uf)].sort_values("date")
            if medium_term is not None
            else pd.DataFrame()
        )
        history = observed_series(observed, uf)
        history = history[history["date"].ge(min_date)]
        if not include_partial_observed:
            history = history[history["date"].lt(forecast_start)]

        if not forecast.empty:
            add_bands(fig, forecast, row, col, color, intervals, showlegend)
        if medium_term_intervals and not medium_forecast.empty:
            add_bands(
                fig, medium_forecast, row, col, MEDIUM_TERM_COLOR,
                intervals, showlegend, model_name="Médio prazo", opacity_factor=0.65,
            )

        if not forecast.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast["date"], y=forecast["pred"], mode="lines",
                    name="Ensemble · mediana", legendgroup="forecast", showlegend=showlegend,
                    legendrank=5, line={"color": color, "width": 2.6},
                    hovertemplate="%{x|%d/%m/%Y}<br>Ensemble: %{y:,.0f}<extra></extra>",
                ), row=row, col=col,
            )
        if not medium_forecast.empty:
            fig.add_trace(
                go.Scatter(
                    x=medium_forecast["date"], y=medium_forecast["pred"], mode="lines",
                    name="Médio prazo · mediana", legendgroup="medium_term",
                    showlegend=showlegend, legendrank=7,
                    line={"color": MEDIUM_TERM_COLOR, "width": 2.8, "dash": "dash"},
                    hovertemplate="%{x|%d/%m/%Y}<br>Médio prazo: %{y:,.0f}<extra></extra>",
                ), row=row, col=col,
            )
        if not history.empty:
            fig.add_trace(
                go.Scatter(
                    x=history["date"], y=history["casos"], mode="markers",
                    name="Casos observados", legendgroup="observed", showlegend=showlegend,
                    legendrank=10, marker={"color": OBS_COLOR, "size": 5.5, "opacity": 0.9},
                    hovertemplate="%{x|%d/%m/%Y}<br>Observados: %{y:,.0f}<extra></extra>",
                ), row=row, col=col,
            )

        if forecast.empty and medium_forecast.empty:
            fig.add_annotation(
                x=0.5, y=0.5, xref=f"x{'' if i == 0 else i + 1} domain",
                yref=f"y{'' if i == 0 else i + 1} domain",
                text="Sem previsão disponível", showarrow=False,
                font={"color": "#8a6671", "size": 12},
            )

        fig.update_xaxes(
            tickformat="%m/%Y", tickangle=-30, showgrid=True,
            gridcolor=GRID_COLOR, row=row, col=col,
        )
        fig.update_yaxes(
            rangemode="tozero", showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, tickformat="~s", row=row, col=col,
        )

    fig.update_layout(
        template="plotly_white", height=345 * nrows + 65, autosize=True,
        margin={"l": 12, "r": 16, "t": 55,
                "b": 125 if medium_term is not None and medium_term_intervals
                else (90 if medium_term is not None else 75)},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "size": 12, "color": TEXT_COLOR},
        hovermode="closest",
        legend={"orientation": "h", "x": 0, "y": -0.12 / nrows,
                "xanchor": "left", "yanchor": "top", "font": {"size": 12}},
    )
    if len(ufs) == 1:
        fig.update_layout(height=450)
    return fig


def accumulated_figure(
    table: pd.DataFrame, ufs: list[str], color: str, height: int = 450,
    log_scale: bool = False,
) -> go.Figure:
    """Three grouped bars: past observed, current observed and forecast ensemble.

    The current observed total covers only the weeks available so far. Forecast
    uncertainty is taken from the original cumulative ensemble quantiles.
    """
    df = table.set_index("uf").reindex(ufs).reset_index()
    pos = list(range(len(ufs)))
    bar_width = 0.23
    prev_x = [x - 0.27 for x in pos]
    current_x = pos
    pred_df = df.loc[df["pred"].notna()].copy()
    forecast_x = [pos[i] + 0.27 for i in df.index if pd.notna(df.loc[i, "pred"])]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=prev_x, y=df["observed_prev"], width=bar_width,
            name="Observado · 2025–2026", marker_color=OBS_BAR_COLOR,
            hovertext=[
                f"{uf}: {fmt_num(value)} casos"
                for uf, value in zip(df["uf"], df["observed_prev"])
            ],
            hovertemplate="%{hovertext}<extra>Temporada anterior (completa)</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=current_x, y=df["observed_current"], width=bar_width,
            name="Observado · 2026–2027 (parcial)", marker_color=OBS_CURRENT_BAR_COLOR,
            hovertext=[
                f"{uf}: {fmt_num(value)} casos"
                for uf, value in zip(df["uf"], df["observed_current"])
            ],
            hovertemplate="%{hovertext}<extra>Temporada atual (até a última observação)</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=forecast_x, y=pred_df["pred"], width=bar_width,
            name="Previsto · 2026–2027", marker_color=color,
            hovertext=[
                f"{uf}: {fmt_num(value)} casos"
                for uf, value in zip(pred_df["uf"], pred_df["pred"])
            ],
            hovertemplate="%{hovertext}<extra>Mediana prevista (temporada completa)</extra>",
        )
    )

    # Os intervalos originais do ensemble ficam sobre as barras previstas.
    for coverage, err_color, thickness in [(90, "#25445d", 1.4), (80, "#e3a33d", 3.2)]:
        lower, upper = f"lower_{coverage}", f"upper_{coverage}"
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=pred_df["pred"],
                mode="markers",
                marker={"size": 4, "color": err_color, "opacity": 0.85},
                name=f"Intervalo {coverage}%",
                error_y={
                    "type": "data", "symmetric": False,
                    "array": (pred_df[upper] - pred_df["pred"]).clip(lower=0).to_list(),
                    "arrayminus": (pred_df["pred"] - pred_df[lower]).clip(lower=0).to_list(),
                    "color": err_color, "thickness": thickness, "width": 4,
                },
                hovertext=[
                    f"{uf} · IP {coverage}%: {fmt_num(lo)} a {fmt_num(hi)} casos"
                    for uf, lo, hi in zip(pred_df["uf"], pred_df[lower], pred_df[upper])
                ],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white", autosize=True, height=height, barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "size": 12, "color": TEXT_COLOR},
        legend={"orientation": "h", "y": -0.18, "x": 0, "xanchor": "left", "yanchor": "top"},
        margin={"l": 15, "r": 20, "t": 22, "b": 105},
    )
    fig.update_xaxes(
        tickmode="array", tickvals=pos,
        ticktext=["Brasil" if x == "BR" else x for x in ufs],
        showgrid=False, zeroline=False, range=[-0.65, len(ufs) - 0.35],
    )
    if log_scale:
        # Escala logarítmica não representa zeros; valores não são transformados.
        fig.update_yaxes(
            title_text="Casos acumulados (escala log)", type="log", autorange=True,
            gridcolor=GRID_COLOR, tickformat="~s", zeroline=False,
        )
    else:
        fig.update_yaxes(
            title_text="Casos acumulados", type="linear", rangemode="tozero",
            gridcolor=GRID_COLOR, tickformat="~s", zeroline=False,
        )
    return fig


def render_dashboard() -> None:
    st.set_page_config(
        page_title="IMDC | Previsões 2026–2027", page_icon="📊", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background-color: #f8fafc; }
        .block-container { padding-top: 1.55rem; padding-bottom: 3rem; max-width: 1550px; }
        h1, h2, h3 { color: #183147; }
        div[data-testid="stMetric"] { background: white; padding: 0.85rem 1rem;
            border: 1px solid #e0e8f0; border-radius: 12px; }
        div[data-testid="stMetricLabel"] { font-weight: 600; }
        div[data-testid="stPlotlyChart"] { border: 1px solid #e4ebf1;
            border-radius: 12px; background-color: white; padding: 0.3rem; }
        /* Destaque para a comparação opcional de previsões semanais. */
        div.st-key-short_term_panel { background: #fff8ee;
            border-left: 5px solid #d97732; border-radius: 12px;
            padding: 0.8rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Previsões de casos prováveis | Brasil")
    st.caption("Ensemble probabilístico (IMDC) · Temporada 2026–2027 · Casos prováveis semanais e acumulados")

    disease = st.radio("Doença", list(DISEASES), horizontal=True, label_visibility="collapsed")
    config = DISEASES[disease]
    with st.sidebar:
        st.header("Configurações")
        n_history = st.slider("Semanas de histórico observado", 13, 104, 52, 13)
        include_partial_observed = st.toggle("Mostrar observações durante a previsão", value=True)
        intervals = st.multiselect("Bandas de incerteza (semanal)", options=[80, 90], default=[80, 90])
        selected_regions = st.multiselect("Regiões", options=list(REGIONS), default=list(REGIONS))
        st.divider()
        #st.caption("Os dados são carregados dos arquivos locais em `data/`. Altere `DASH_DATA_DIR` para usar outra pasta.")

    render_data_content(config, n_history, include_partial_observed, intervals, selected_regions)


def render_data_content(
    config: dict[str, str],
    n_history: int,
    include_partial_observed: bool,
    intervals: list[int],
    selected_regions: list[str],
) -> None:
    """Renderize a interface usando os dados atuais a cada execução do aplicativo."""
    try:
        weekly, totals, observed = read_data(config["key"])
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()

    start = weekly["date"].min()
    end = weekly["date"].max()
    n_weeks = weekly.loc[weekly["uf"].eq("BR"), "date"].nunique()
    prev_start, prev_end = preceding_season(start, n_weeks)
    total_table = build_cumulative_table(totals, observed, start, n_weeks)
    br = total_table.loc[total_table["uf"].eq("BR")].iloc[0]
    last_observed = observed["date"].max()
    current_br_observed = observed_national_current_season(observed, start, end)

    st.caption(
        f"Previsão: **{start:%d/%m/%Y} a {end:%d/%m/%Y}** ({n_weeks} semanas) · "
        f"Última semana observada: **{last_observed:%d/%m/%Y}**"
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Brasil · mediana prevista", fmt_num(br["pred"]))
    k2.metric("Intervalo 80%", f"{fmt_num(br['lower_80'])} – {fmt_num(br['upper_80'])}")
    k3.metric("Intervalo 90%", f"{fmt_num(br['lower_90'])} – {fmt_num(br['upper_90'])}")
    k4.metric("Brasil · observados desde SE 202641", fmt_num(current_br_observed))
    st.caption(
        f"Card de observados: soma das 27 UFs de {start:%d/%m/%Y} "
        f"até {min(last_observed, end):%d/%m/%Y} (temporada 2026–2027). "
        #f"Nos gráficos de barras, a comparação histórica permanece em "
        #f"{prev_start:%d/%m/%Y} a {prev_end:%d/%m/%Y}."
    )

    weekly_tab, cumulative_tab = st.tabs(["Séries semanais", "Casos acumulados"])

    with weekly_tab:
        medium_term = None
        medium_term_intervals = False
        # Bloco destacado e sempre visível, inclusive para explicar a
        # indisponibilidade desta comparação para chikungunya.
        with st.container(border=True, key="short_term_panel"):
            st.markdown("#### 🟠 Previsões de curto prazo")
            st.markdown(
                "**Disponíveis apenas para dengue.** "
                "Ative a comparação para sobrepor uma segunda previsão "
                "(linha laranja tracejada) às estimativas semanais do ensemble."
            )
            if config["key"] == "dengue":
                show_medium_term = st.toggle(
                    "🟠 Ativar comparação com as previsões de curto prazo",
                    value=False,
                    help=(
                        "Mostra a série adicional de dengue do arquivo "
                        "medium_term_dengue_preds.csv.gz, identificado como "
                        "modelo de médio prazo (16 semanas)."
                    ),
                    key="show_medium_term_dengue",
                )
                if show_medium_term:
                    try:
                        medium_term = read_medium_term_data()
                    except (FileNotFoundError, ValueError, OSError) as exc:
                        st.warning(f"Não foi possível carregar a previsão adicional: {exc}")
                    else:
                        medium_term_intervals = st.checkbox(
                            "Exibir também os intervalos de 80% e 90% do modelo adicional",
                            value=False,
                            key="show_medium_term_intervals",
                        )
                        st.caption(
                            "O arquivo disponível corresponde ao modelo de **médio prazo**, "
                            f"com {medium_term['date'].nunique()} semanas "
                            f"(até {medium_term['date'].max():%d/%m/%Y}). "
                            "A legenda dos gráficos mantém essa nomenclatura técnica."
                        )
            else:
                st.toggle(
                    "🟠 Ativar comparação com as previsões de curto prazo",
                    value=False,
                    disabled=True,
                    key="show_medium_term_chik_disabled",
                )
                st.caption("Selecione **Dengue** acima para habilitar esta comparação.")
        st.subheader("Brasil")
        st.plotly_chart(
            weekly_figure(["BR"], weekly, observed, start, n_history,
                          include_partial_observed, intervals, config["color"], columns=1,
                          medium_term=medium_term, medium_term_intervals=medium_term_intervals),
            use_container_width=True, config={"displaylogo": False, "responsive": True},
            key=f"br_weekly_{config['key']}",
        )
        st.caption(
            "Série observada do Brasil = soma semanal dos casos das 27 UFs; "
            "previsões nacionais = séries BR fornecidas por cada modelo."
        )
        st.subheader("Estados organizados por região")
        st.caption("Cada painel reúne os estados da região, com escalas verticais próprias para facilitar a leitura.")
        for region in selected_regions:
            st.markdown(f"#### {region} · {' · '.join(REGIONS[region])}")
            st.plotly_chart(
                weekly_figure(REGIONS[region], weekly, observed, start, n_history,
                              include_partial_observed, intervals, config["color"], columns=3,
                              medium_term=medium_term, medium_term_intervals=medium_term_intervals),
                use_container_width=True, config={"displaylogo": False, "responsive": True},
                key=f"week_{config['key']}_{region}",
            )
            if "ES" in REGIONS[region]:
                if medium_term is not None and "ES" in medium_term["uf"].values:
                    st.caption("ES: dados observados e previsão de médio prazo; o ensemble não possui previsão para o estado.")
                else:
                    st.caption("ES: somente dados observados; o ensemble não possui previsão para o estado.")

    with cumulative_tab:
        chart_scale = st.radio(
            "Escala dos gráficos de barras", ["Linear", "Logarítmica"],
            horizontal=True, key="cumulative_axis_scale",
        )
        log_scale = chart_scale == "Logarítmica"
        if log_scale:
            st.caption("Na escala logarítmica, valores iguais a zero não podem ser exibidos; os dados originais não são alterados.")
        observed_current_dates = observed.loc[observed["date"].between(start, end), "date"]
        if not observed_current_dates.empty:
            st.caption(
                f"Temporada atual: casos observados de {start:%d/%m/%Y} "
                f"até {observed_current_dates.max():%d/%m/%Y} "
                "(**acumulado parcial**). A previsão cobre a temporada completa."
            )
        st.subheader("Brasil · acumulado da temporada")
        st.plotly_chart(
            accumulated_figure(total_table, ["BR"], config["color"], height=470, log_scale=log_scale),
            use_container_width=True, config={"displaylogo": False, "responsive": True},
            key=f"br_total_{config['key']}",
        )
        st.caption(
            "**Cinza:** observado em 2025–2026 (temporada completa). "
            "**Verde:** observado em 2026–2027 (somente semanas disponíveis). "
            f"**{'Azul' if config['key'] == 'dengue' else 'Roxo'}:** "
            "mediana prevista para toda a temporada 2026–2027. "
            "Traços verticais: intervalos de 80% (ocre) e 90% (azul-escuro). "
            "**A barra verde ainda não representa uma temporada completa.**"
        )
        st.subheader("Acumulados por estado, agrupados por região")
        for region in selected_regions:
            st.markdown(f"#### {region}")
            st.plotly_chart(
                accumulated_figure(total_table, REGIONS[region], config["color"], height=480, log_scale=log_scale),
                use_container_width=True, config={"displaylogo": False, "responsive": True},
                key=f"total_{config['key']}_{region}",
            )
        if "Sudeste" in selected_regions:
            st.caption("ES não possui previsão no ensemble; por isso sua barra colorida está ausente.")

        with st.expander("Ver tabela e baixar os acumulados", expanded=False):
            output = total_table[
                ["region", "uf", "observed_prev", "observed_current", "pred", "lower_80", "upper_80", "lower_90", "upper_90"]
            ].copy()
            st.dataframe(
                output.rename(columns={
                    "region": "Região", "uf": "UF", "observed_prev": "Observado 2025–2026",
                    "observed_current": "Observado 2026–2027 (parcial)",
                    "pred": "Mediana 2026–2027", "lower_80": "Limite inf. 80%",
                    "upper_80": "Limite sup. 80%", "lower_90": "Limite inf. 90%",
                    "upper_90": "Limite sup. 90%",
                }).style.format(precision=0, thousands=".", na_rep="—"),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Baixar acumulados (CSV)",
                data=output.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"acumulados_{config['key']}_2026_2027.csv",
                mime="text/csv", key=f"download_{config['key']}",
            )

    with st.expander("Notas metodológicas"):
        st.markdown(
            """
            - **Atualização dos dados:** o conteúdo dos CSVs é verificado quando a
              página é aberta, recarregada ou quando um controle é alterado.
              Arquivos modificados invalidam o cache correspondente. Não há
              atualização periódica em segundo plano.
            - **Série semanal observada:** coluna `casos` dos arquivos `*_casprov.csv.gz`,
              agregada por data para o Brasil e sem agregação para cada UF;
              exibida como pontos pretos.
            - **Série semanal prevista (ensemble):** coluna `pred` e bandas de 80%/90%
              dos arquivos `ensemble_weekly_2027_*.csv.gz`.
            - **Médio prazo (somente dengue):** opcionalmente, coluna `pred` do arquivo
              `medium_term_dengue_preds.csv.gz`, em laranja tracejado. As bandas de
              80% e 90% podem ser exibidas separadamente com a caixa de seleção.
              São 16 semanas de previsão, inclusive para ES, onde não há ensemble;
              os valores de Brasil são fornecidos pelo próprio modelo de médio prazo.
              O arquivo de médio prazo não altera os acumulados nem os cards. 
            - **Acumulados previstos:** quantis originais de `ensemble_total_2027_*.csv.gz`.
              Quantis semanais **não são somados** para construir intervalos acumulados.
            - **Acumulados observados:** barras cinzas somam as 52 semanas da temporada
              anterior (2025–2026); barras verdes somam as semanas **já disponíveis**
              da temporada atual (2026–2027), a partir da SE 202641, por UF.
              O Brasil é obtido pela soma das 27 UFs, incluindo ES. A barra verde
              representa o acumulado **parcial**, não a temporada completa.
            - **Escala logarítmica:** opção disponível nos gráficos de barras;
              valores zero não são representáveis nessa escala.
            - **Brasil:** casos observados são somados de todas as UFs, incluindo ES;
              previsões nacionais utilizam a linha `BR` fornecida pelos arquivos do ensemble.
              O total nacional previsto pode diferir da soma das medianas estaduais.
            - **Comparabilidade:** o acumulado de 2025–2026 e a previsão do ensemble
              são para temporadas completas; o observado de 2026–2027 inclui apenas
              as semanas já publicadas. Não compare os totais como se tivessem
              a mesma duração.
            """
        )


if __name__ == "__main__":
    render_dashboard()
