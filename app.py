from datetime import date, timedelta
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st


DATA_PATH = "data.parquet"
DEFAULT_FOCUS = ["LATINO", "Global", "Brasil", "México", "Colombia", "Argentina", "Chile", "Perú"]
RISK_COUNTRIES = {"LATINO", "Global"}
PERIOD_LABELS = ["1D", "1M", "MTD", "QTD", "YTD"]
PALETTE = [
    "#66E3D4",
    "#F6B26B",
    "#9BD67D",
    "#FF8A80",
    "#8AB4F8",
    "#D7BDE2",
    "#F4D35E",
    "#8ED1FC",
    "#F2A7C6",
    "#B9C6AE",
]


st.set_page_config(
    page_title="LatAm EMBI Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1117;
            --bg-2: #101820;
            --panel: rgba(20, 30, 39, 0.92);
            --panel-2: rgba(15, 23, 31, 0.96);
            --line: rgba(184, 202, 214, 0.16);
            --ink: #edf4f2;
            --muted: #9eacb5;
            --teal: #66e3d4;
            --amber: #f6b26b;
        }
        .stApp {
            background:
                radial-gradient(circle at 16% 6%, rgba(102, 227, 212, 0.12), transparent 30rem),
                radial-gradient(circle at 86% 12%, rgba(246, 178, 107, 0.10), transparent 34rem),
                linear-gradient(135deg, #080d12 0%, #0d141b 48%, #111922 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 2.4rem;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101820 0%, #0b1117 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--ink);
        }
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] label p {
            color: var(--muted) !important;
        }
        [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            background-color: #111b24 !important;
            border-color: rgba(184, 202, 214, 0.22) !important;
            color: var(--ink) !important;
        }
        [data-baseweb="tag"] {
            background-color: #66e3d4 !important;
            border: 1px solid rgba(102, 227, 212, 0.55) !important;
            border-radius: 999px !important;
        }
        [data-baseweb="tag"],
        [data-baseweb="tag"] *,
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] svg {
            color: #061014 !important;
            fill: #061014 !important;
            font-weight: 700;
        }
        [data-testid="stMultiSelect"] input,
        [data-testid="stMultiSelect"] div,
        [data-testid="stSelectbox"] div {
            color: var(--ink) !important;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 760;
        }
        [data-testid="stMetricDelta"] {
            font-weight: 650;
        }
        .hero {
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 28px 30px;
            margin-bottom: 20px;
            background:
                linear-gradient(120deg, rgba(16, 24, 32, 0.96), rgba(9, 15, 20, 0.98)),
                radial-gradient(circle at 100% 0%, rgba(102, 227, 212, 0.20), transparent 24rem),
                repeating-linear-gradient(45deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 14px);
            color: var(--ink);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
        }
        .eyebrow {
            color: var(--amber);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.14em;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
        }
        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4.3rem);
            line-height: 0.95;
            margin: 0 0 0.8rem 0;
            letter-spacing: -0.055em;
        }
        .hero p {
            color: rgba(237, 244, 242, 0.78);
            font-size: 1.02rem;
            max-width: 880px;
            margin: 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.94rem;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
        }
        h2, h3 {
            letter-spacing: -0.03em;
            color: var(--ink);
        }
        p, li, span {
            color: inherit;
        }
        hr {
            border-color: var(--line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_region_name(region: str) -> str:
    return {
        "Brasil": "Brazil",
        "México": "Mexico",
        "Perú": "Peru",
        "Panamá": "Panama",
        "REP DOM": "Dominican Republic",
        "LATINO": "LatAm",
    }.get(region, region)


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pl.DataFrame:
    if not Path(path).exists():
        st.error(f"{path} not found.")
        st.stop()

    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    if "Date" in schema and "date" not in schema:
        lf = lf.rename({"Date": "date"})

    schema = lf.collect_schema()
    if {"date", "region", "value"}.issubset(schema):
        normalized = lf.select(["date", "region", "value"])
    else:
        value_cols = [col for col in schema if col != "date"]
        normalized = lf.unpivot(
            index="date",
            on=value_cols,
            variable_name="region",
            value_name="value",
        )

    return (
        normalized.with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("region").cast(pl.Utf8),
            pl.col("value").cast(pl.Float64),
        )
        .drop_nulls(["date", "region", "value"])
        .sort(["region", "date"])
        .collect()
    )


def observation_on_or_before(df: pl.DataFrame, cutoff: date) -> date | None:
    result = df.filter(pl.col("date") <= cutoff).select(pl.col("date").max()).item()
    return result


def period_cutoffs(df: pl.DataFrame, latest_date: date) -> dict[str, date | None]:
    latest_quarter = (latest_date.month - 1) // 3
    quarter_start_month = latest_quarter * 3 + 1
    anchors = {
        "1D": latest_date - timedelta(days=1),
        "1M": latest_date - timedelta(days=30),
        "MTD": date(latest_date.year, latest_date.month, 1),
        "QTD": date(latest_date.year, quarter_start_month, 1),
        "YTD": date(latest_date.year, 1, 1),
    }
    return {label: observation_on_or_before(df, cutoff) for label, cutoff in anchors.items()}


def build_metrics(df: pl.DataFrame, latest_date: date, cutoffs: dict[str, date | None]) -> pl.DataFrame:
    latest = (
        df.filter(pl.col("date") == latest_date)
        .select("region", pl.col("value").alias("spread_pct"))
    )

    metrics = latest
    for label, cutoff in cutoffs.items():
        if cutoff is None:
            metrics = metrics.with_columns(pl.lit(None, dtype=pl.Float64).alias(f"{label} bps"))
            continue
        reference = (
            df.filter(pl.col("date") == cutoff)
            .select("region", pl.col("value").alias(f"{label}_ref"))
        )
        metrics = metrics.join(reference, on="region", how="left").with_columns(
            ((pl.col("spread_pct") - pl.col(f"{label}_ref")) * 100).alias(f"{label} bps")
        ).drop(f"{label}_ref")

    latam = metrics.filter(pl.col("region") == "LATINO").select("spread_pct").to_series()
    global_spread = metrics.filter(pl.col("region") == "Global").select("spread_pct").to_series()
    latam_value = latam[0] if len(latam) else metrics.select(pl.col("spread_pct").mean()).item()
    global_value = global_spread[0] if len(global_spread) else None

    metrics = metrics.with_columns(
        (pl.col("spread_pct") * 100).alias("spread_bps"),
        ((pl.col("spread_pct") - latam_value) * 100).alias("vs LatAm bps"),
        pl.when(pl.lit(global_value is not None))
        .then((pl.col("spread_pct") - pl.lit(global_value or 0.0)) * 100)
        .otherwise(None)
        .alias("vs Global bps"),
        pl.col("region").map_elements(normalize_region_name, return_dtype=pl.Utf8).alias("Market"),
    )

    return metrics.sort("spread_bps", descending=True)


def add_percentile_rank(df: pl.DataFrame, metrics: pl.DataFrame) -> pl.DataFrame:
    country_regions = [
        region for region in metrics["region"].to_list() if region not in RISK_COUNTRIES
    ]
    percentiles = (
        df.filter(pl.col("region").is_in(country_regions))
        .group_by("region")
        .agg(
            (pl.col("value").rank("average").last() / pl.len() * 100).alias("historical percentile")
        )
    )
    return metrics.join(percentiles, on="region", how="left")


def format_bps(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f} bps"


def metric_delta(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:+,.0f} bps"


def delta_color(value: float | None) -> str:
    if value is None:
        return "off"
    return "inverse" if value > 0 else "normal"


def chart_theme_config() -> dict:
    return {
        "config": {
            "background": "transparent",
            "axis": {
                "labelColor": "#9eacb5",
                "titleColor": "#c9d6dc",
                "gridColor": "rgba(184, 202, 214, 0.12)",
                "domainColor": "rgba(184, 202, 214, 0.18)",
            },
            "legend": {"labelColor": "#edf4f2", "titleColor": "#9eacb5"},
            "view": {"stroke": "transparent"},
        }
    }


if hasattr(alt, "theme"):
    @alt.theme.register("embi_theme", enable=True)
    def embi_theme():
        return alt.theme.ThemeConfig(chart_theme_config())
else:
    alt.themes.register("embi_theme", chart_theme_config)
    alt.themes.enable("embi_theme")
inject_css()

df = load_data()
if not {"date", "region", "value"}.issubset(df.columns):
    st.error("Dataset missing required columns: date, region, value.")
    st.stop()

min_date = df.select(pl.col("date").min()).item()
latest_date = df.select(pl.col("date").max()).item()
cutoffs = period_cutoffs(df, latest_date)
metrics = add_percentile_rank(df, build_metrics(df, latest_date, cutoffs))
regions = metrics["region"].to_list()

latest_latam = metrics.filter(pl.col("region") == "LATINO")
latam_spread = latest_latam.select("spread_bps").item() if latest_latam.height else None
latam_1m = latest_latam.select("1M bps").item() if latest_latam.height else None
latest_global = metrics.filter(pl.col("region") == "Global")
global_spread = latest_global.select("spread_bps").item() if latest_global.height else None
high_risk = metrics.filter(~pl.col("region").is_in(RISK_COUNTRIES)).head(1)
largest_widener = metrics.filter(~pl.col("region").is_in(RISK_COUNTRIES)).sort("1M bps", descending=True).head(1)
largest_tightener = metrics.filter(~pl.col("region").is_in(RISK_COUNTRIES)).sort("1M bps").head(1)

with st.sidebar:
    st.header("Market Controls")
    st.caption("Benchmarks are always available for context. Limit selected markets to keep charts readable.")
    default_markets = [region for region in DEFAULT_FOCUS if region in regions]
    selected_regions = st.multiselect("Markets", regions, default=default_markets)
    selected_regions = selected_regions or default_markets or regions[:8]

    start_date, end_date = st.slider(
        "Date range",
        min_value=min_date,
        max_value=latest_date,
        value=(max(latest_date - timedelta(days=365 * 3), min_date), latest_date),
        format="YYYY-MM-DD",
    )
    comparison_period = st.selectbox("Movement ranking period", PERIOD_LABELS, index=1)
    detail_default = "LATINO" if "LATINO" in regions else regions[0]
    detail_region = st.selectbox(
        "Detail market",
        regions,
        index=regions.index(detail_default),
        format_func=normalize_region_name,
    )
    show_outliers = st.checkbox("Include Venezuela in ranked charts", value=False)

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Emerging Market Bond Index</div>
        <h1>LatAm Sovereign Risk Monitor</h1>
        <p>
        Daily EMBI spreads for Latin American sovereigns, focused on risk premia,
        benchmark gaps, and short-term widening or tightening. Latest observation:
        <strong>{latest_date:%d %b %Y}</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

card1, card2, card3, card4 = st.columns(4)
card1.metric(
    "LatAm EMBI",
    format_bps(latam_spread),
    metric_delta(latam_1m),
    delta_color=delta_color(latam_1m),
)
card2.metric("Global EMBI", format_bps(global_spread))
if high_risk.height:
    card3.metric(
        "Highest Country Spread",
        normalize_region_name(high_risk.select("region").item()),
        format_bps(high_risk.select("spread_bps").item()),
        delta_color="inverse",
    )
if largest_widener.height and largest_tightener.height:
    widener = largest_widener.select("region").item()
    card4.metric(
        f"{comparison_period} Widening",
        normalize_region_name(widener),
        metric_delta(largest_widener.select(f"{comparison_period} bps").item()),
        delta_color="inverse",
    )

st.markdown(
    """
    EMBI spreads measure the extra yield investors demand over comparable U.S. Treasuries.
    A wider spread usually signals higher perceived sovereign risk or tighter external financing conditions;
    a tighter spread suggests improving risk appetite, credit perception, or liquidity.
    """
)

st.subheader("Market Snapshot")
st.markdown(
    '<div class="section-note">Latest levels and period changes. Positive changes mean spreads widened. '
    "Hist. Percentile shows where the latest spread sits versus that market's own history: "
    "90% means today's spread is higher than roughly 90% of past observations.</div>",
    unsafe_allow_html=True,
)
display_cols = [
    "Market",
    "spread_bps",
    "1D bps",
    "1M bps",
    "MTD bps",
    "QTD bps",
    "YTD bps",
    "vs LatAm bps",
    "historical percentile",
]
snapshot = metrics.select(display_cols).rename(
    {
        "spread_bps": "Spread",
        "historical percentile": "Hist. Percentile",
    }
)
st.dataframe(
    snapshot.to_pandas(),
    width="stretch",
    hide_index=True,
    column_config={
        "Spread": st.column_config.NumberColumn("Spread", format="%.0f bps"),
        "1D bps": st.column_config.NumberColumn("1D", format="%+.0f bps"),
        "1M bps": st.column_config.NumberColumn("1M", format="%+.0f bps"),
        "MTD bps": st.column_config.NumberColumn("MTD", format="%+.0f bps"),
        "QTD bps": st.column_config.NumberColumn("QTD", format="%+.0f bps"),
        "YTD bps": st.column_config.NumberColumn("YTD", format="%+.0f bps"),
        "vs LatAm bps": st.column_config.NumberColumn("vs LatAm", format="%+.0f bps"),
        "Hist. Percentile": st.column_config.ProgressColumn(
            "Hist. Percentile",
            help="Latest spread percentile versus each market's own historical observations. Higher means the current spread is unusually wide for that market.",
            min_value=0,
            max_value=100,
            format="%.0f%%",
        ),
    },
)

rank_filter = ~pl.col("region").is_in(RISK_COUNTRIES)
if not show_outliers:
    rank_filter = rank_filter & (pl.col("region") != "Venezuela")
ranked = metrics.filter(rank_filter).sort("spread_bps", descending=True)
ranked_pd = ranked.select(["Market", "spread_bps", "vs LatAm bps"]).to_pandas()

bars = (
    alt.Chart(ranked_pd)
    .mark_bar(cornerRadiusEnd=5, color="#66E3D4")
    .encode(
        x=alt.X("spread_bps:Q", title="Spread (bps)"),
        y=alt.Y(
            "Market:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelLimit=180, labelOverlap=False, labelPadding=8),
        ),
        tooltip=[
            alt.Tooltip("Market:N"),
            alt.Tooltip("spread_bps:Q", title="Spread", format=",.0f"),
            alt.Tooltip("vs LatAm bps:Q", title="vs LatAm", format="+,.0f"),
        ],
    )
)
latam_rule = alt.Chart({"values": [{"x": latam_spread}]}).mark_rule(
    color="#F6B26B", strokeDash=[6, 4], strokeWidth=2
).encode(x="x:Q")
st.altair_chart((bars + latam_rule).properties(height=max(420, 34 * ranked.height)), width="stretch")

left, right = st.columns((1.35, 1))
with left:
    st.subheader("Trend: Selected Markets")
    st.markdown('<div class="section-note">Raw EMBI spread in percentage points. Keep market selection focused for readable comparison.</div>', unsafe_allow_html=True)
    series = df.filter(
        pl.col("region").is_in(selected_regions)
        & (pl.col("date") >= start_date)
        & (pl.col("date") <= end_date)
    ).with_columns(
        pl.col("region").map_elements(normalize_region_name, return_dtype=pl.Utf8).alias("Market")
    )
    trend = (
        alt.Chart(series.to_pandas())
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title="Spread (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("Market:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("Market:N"),
                alt.Tooltip("value:Q", title="Spread (%)", format=".2f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(trend, width="stretch")

with right:
    st.subheader(f"{comparison_period} Movers")
    st.markdown('<div class="section-note">Largest spread changes among country series.</div>', unsafe_allow_html=True)
    movers = (
        metrics.filter(~pl.col("region").is_in(RISK_COUNTRIES))
        .filter(pl.col(f"{comparison_period} bps").is_not_null())
        .with_columns(
            pl.when(pl.col(f"{comparison_period} bps") >= 0)
            .then(pl.lit("Widening"))
            .otherwise(pl.lit("Tightening"))
            .alias("Direction")
        )
    )
    if not show_outliers:
        movers = movers.filter(pl.col("region") != "Venezuela")
    mover_chart = (
        alt.Chart(movers.select(["Market", f"{comparison_period} bps", "Direction"]).to_pandas())
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(f"{comparison_period} bps:Q", title="Change (bps)"),
            y=alt.Y(
                "Market:N",
                sort=alt.EncodingSortField(field=f"{comparison_period} bps", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=180, labelOverlap=False, labelPadding=8),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Widening", "Tightening"], range=["#FF8A80", "#9BD67D"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Market:N"),
                alt.Tooltip(f"{comparison_period} bps:Q", title="Change", format="+,.0f"),
            ],
        )
    )
    st.altair_chart(mover_chart.properties(height=max(420, 32 * movers.height)), width="stretch")

st.subheader("Relative Performance")
st.markdown('<div class="section-note">Selected markets indexed to 100 at the beginning of the displayed window; useful for relative moves when spread levels differ.</div>', unsafe_allow_html=True)
indexed = (
    series.sort(["region", "date"])
    .with_columns((pl.col("value") / pl.col("value").first().over("region") * 100).alias("indexed"))
)
indexed_chart = (
    alt.Chart(indexed.to_pandas())
    .mark_line(strokeWidth=2.4)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("indexed:Q", title="Index, first visible date = 100", scale=alt.Scale(zero=False)),
        color=alt.Color("Market:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(orient="bottom")),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Market:N"),
            alt.Tooltip("indexed:Q", title="Index", format=".1f"),
        ],
    )
    .interactive()
)
st.altair_chart(indexed_chart, width="stretch")

st.subheader(f"Detail: {normalize_region_name(detail_region)}")
detail_metrics = metrics.filter(pl.col("region") == detail_region)
detail_series = df.filter(pl.col("region") == detail_region).sort("date").with_columns(
    (pl.col("value") * 100).alias("spread_bps"),
    (pl.col("value").rolling_mean(50) * 100).alias("rolling_50d_bps"),
)

d1, d2, d3, d4 = st.columns(4)
d1.metric("Latest Spread", format_bps(detail_metrics.select("spread_bps").item()))
detail_1m = detail_metrics.select("1M bps").item()
detail_vs_latam = detail_metrics.select("vs LatAm bps").item()
d2.metric("1M Change", metric_delta(detail_1m) or "-", delta_color=delta_color(detail_1m))
d3.metric("vs LatAm", metric_delta(detail_vs_latam) or "-", delta_color=delta_color(detail_vs_latam))
hist_pct = detail_metrics.select("historical percentile").item()
d4.metric("Historical Percentile", "-" if hist_pct is None else f"{hist_pct:.0f}%")

detail_chart = (
    alt.Chart(detail_series.to_pandas())
    .transform_fold(["spread_bps", "rolling_50d_bps"], as_=["Series", "Spread"])
    .mark_line(strokeWidth=2.5)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Spread:Q", title="Spread (bps)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=["spread_bps", "rolling_50d_bps"],
                range=["#66E3D4", "#F6B26B"],
            ),
            legend=alt.Legend(title=None, labelExpr="datum.label == 'spread_bps' ? 'Daily spread' : '50-day average'"),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Spread:Q", format=",.0f"),
            alt.Tooltip("Series:N"),
        ],
    )
    .interactive()
)
st.altair_chart(detail_chart, width="stretch")

with st.expander("Data notes and interpretation"):
    st.markdown(
        f"""
        - Source file: `{DATA_PATH}`, updated by the repository data pipeline.
        - Coverage in the current dataset: {min_date:%d %b %Y} to {latest_date:%d %b %Y}.
        - Dashboard levels convert stored percentage-point spreads into basis points for tables and cards.
        - Widening spreads are shown as positive basis-point changes and usually indicate higher perceived risk.
        - Historical percentile ranks the latest spread against each market's own history; a higher percentile means the current spread is unusually wide relative to that market's past.
        - Benchmarks: LATINO is the regional EMBI aggregate; Global is the broader EMBI aggregate.
        """
    )

st.caption("Source: Emerging Market Bond Index historical spread series published by BCRD.")
