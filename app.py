import streamlit as st
import polars as pl
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="EMBI Report", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path: str = "data.parquet") -> pl.LazyFrame:
    if not Path(path).exists():
        st.error(f"{path} not found")
        st.stop()
    lf = pl.scan_parquet(path)
    if "Date" in lf.collect_schema() and "date" not in lf.collect_schema():
        lf = lf.rename({"Date": "date"})
    if {"region", "value"}.isdisjoint(lf.collect_schema()):
        cols = [c for c in lf.collect_schema() if c != "date"]
        lf = lf.unpivot(
            index="date",
            on=cols,
            variable_name="region",
            value_name="value",
        )
    return lf.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("region").cast(pl.Utf8),
        pl.col("value").cast(pl.Float64),
    )

lf = load_data()
df = lf.collect()
if not {"date", "region", "value"}.issubset(df.columns):
    st.error("Dataset missing required columns")
    st.stop()

latest_date = df.select(pl.col("date").max()).item()
latest_year = latest_date.year
start_year = (
    df
    .filter(pl.col("date").dt.year() == latest_year)
    .select(pl.col("date").min())
    .item()
)
latest_quarter = (latest_date.month - 1) // 3
start_qtr = (
    df
    .filter (
        (pl.col("date").dt.year() == latest_year) &
        (pl.col("date").dt.month().is_between(3*latest_quarter + 1, 3*latest_quarter + 3))
    )
    .select(pl.col("date").min())
    .item()
)
start_month = (
    df
    .filter(
        (pl.col("date").dt.year() == latest_year) &
        (pl.col("date").dt.month() == latest_date.month)
    )
    .select(pl.col("date").min())
    .item()
)
month_ago = latest_date - timedelta(days=30)

periods = {"YTD": start_year, "QTD": start_qtr, "MTD": start_month, "1M": month_ago}
agg_exprs = [pl.col("value").filter(pl.col("date") <= latest_date).last().alias("Valor")]
for label, cutoff in periods.items():
    agg_exprs.append(
        pl.col("value").filter(pl.col("date") <= cutoff).last().alias(f"{label}_raw")
    )

metrics = (
    lf.group_by("region")
      .agg(*agg_exprs)
      .with_columns([
          ((pl.col("Valor") - pl.col(f"{label}_raw")) * 1e2).alias(f"{label} (bps)")
          for label in periods
      ])
      .drop([f"{label}_raw" for label in periods])
      .collect()
      .sort("Valor", descending=True)
)

latam_row = metrics.filter(pl.col("region").str.to_lowercase() == "latino")
latam_val = latam_row["Valor"][0] if latam_row.height > 0 else metrics["Valor"].mean()

metrics = metrics.with_columns(
    ((pl.col("Valor") - pl.lit(latam_val)) * 1e2).alias("Spread vs LatAm")
)

regions = metrics["region"].to_list()

# 1) region selector with "select all"
select_all = st.sidebar.checkbox("Select all regions", value=True)

if select_all:
    sel_regions = regions                # every region selected
else:
    sel_regions = st.sidebar.multiselect("Región", regions, default=regions)

# 2) time‑range slider (min/max dates are inferred from the data)
min_date = df.select(pl.col("date").min()).item()
max_date = df.select(pl.col("date").max()).item()

start_date, end_date = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY‑MM‑DD",
)

table = metrics.filter(pl.col("region").is_in(sel_regions))

st.title("EMBI Report")
st.subheader("Summary")
st.dataframe(
    table.to_pandas(),
    column_config={
        "Valor": st.column_config.NumberColumn(format="%.2f"),
        **{f"{label} (bps)": st.column_config.NumberColumn(format="%.2f") for label in periods},
        "Spread vs LatAm": st.column_config.NumberColumn(format="%.2f"),
    },
    hide_index=True,
)

bar_df = table.select(["region", "Valor"]).sort("Valor", descending=True)

st.altair_chart(
    alt.Chart(bar_df.to_pandas()).mark_bar().encode(
        x="Valor:Q",
        y=alt.Y("region:N", sort="-x"),
        tooltip=["region", "Valor"],
    ),
    use_container_width=True,
)

series = (
    lf.filter(pl.col("region").is_in(sel_regions))
      .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
      .select(["date", "region", "value"])
      .collect()
)

# ---------- RAW SERIES ----------
line_chart = (
    alt.Chart(series.to_pandas())
        # make the line thicker → easier hover/click
        .mark_line(strokeWidth=3)               # :contentReference[oaicite:0]{index=0}
        .encode(
            x="date:T",
            y="value:Q",
            color="region:N",
            tooltip=[
                alt.Tooltip("date:T",    title="Fecha"),
                alt.Tooltip("region:N",  title="Región"),
                alt.Tooltip("value:Q",   title="Valor"),
            ],                              # full tooltip per point :contentReference[oaicite:1]{index=1}
        )
        .interactive()                       # pan/zoom still works :contentReference[oaicite:2]{index=2}
)

st.altair_chart(line_chart, use_container_width=True)

roll = (
    series.with_columns(
        pl.col("value").rolling_mean(50).over("region").alias("rolling50")
    )
    .select(["date", "region", "rolling50"])
)

# ---------- 50‑DAY ROLLING MEAN ----------
roll_chart = (
    alt.Chart(roll.to_pandas())
        .mark_line(strokeWidth=3)               # same thicker line
        .encode(
            x="date:T",
            y="rolling50:Q",
            color="region:N",
            tooltip=[
                alt.Tooltip("date:T",        title="Fecha"),
                alt.Tooltip("region:N",      title="Región"),
                alt.Tooltip("rolling50:Q",   title="Media 50 d"),
            ],
        )
        .interactive()
)

st.altair_chart(roll_chart, use_container_width=True)

st.caption("Fuente: Emerging Market Bond Index")
