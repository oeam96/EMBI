import polars as pl
import requests
import io
import datetime as dt
from functools import reduce

def parse_mixed_date(date_str: str) -> str | None:
    """
    Attempt multiple date patterns and return a unified YYYY-MM-DD string.
    Returns None if none match.
    """
    patterns = ("%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y")
    for fmt in patterns:
        try:
            # Convert to a datetime, then back to ISO string for consistent parsing
            dt_obj = dt.datetime.strptime(date_str, fmt)
            return dt_obj.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


# URL to download EMBI file
url = "https://bcrdgdcprod.blob.core.windows.net/documents/entorno-internacional/documents/Serie_Historica_Spread_del_EMBI.xlsx"

# Download the file content
response = requests.get(url)
response.raise_for_status()

# Read the Excel content into Polars DataFrame
df = pl.read_excel(io.BytesIO(response.content),
                   sheet_name="Serie Histórica",
                   columns=[i for i in range(0,20)])

# Using first row as header
header = df.row(0)
header = list(header)
df.columns= header

# Removing the first row
df = df.filter(~pl.Series(range(len(df))).is_in([0]))

# Define a custom parser function that tries multiple formats.
def parse_mixed_date(s: str) -> str | None:
    patterns = [
        "%Y-%m-%d %H:%M:%S",  # e.g., "2025-03-13 00:00:00"
        "%Y-%m-%d",           # e.g., "2025-03-13"
        "%d-%b-%y",           # e.g., "20-Jan-24"
        "'%d-%b-%y",
        "%d-%b-%Y",           # e.g., "20-Jan-2024"
    ]
    for fmt in patterns:
        try:
            dt_obj = dt.datetime.strptime(s, fmt)
            return dt_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

df = df.with_columns(
    pl.col("Fecha")
      .map_elements(parse_mixed_date, return_dtype=pl.Utf8)
      .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
      .alias("Fecha_parsed")
)

# Drop the original "Fecha" column and rename "Fecha_parsed" to "Fecha"
df = df.drop("Fecha").rename({"Fecha_parsed": "Date"})

# Reorder columns so that "Fecha" is the first column.
cols = df.columns
new_order = ["Date"] + [col for col in cols if col != "Date"]
df = df.select(new_order)

# --- Filter out rows where any column contains "`" or "N/A" ---
# For each column, cast to Utf8 and check that it does not contain either unwanted value.
conditions = [
    ~pl.col(col).cast(pl.Utf8).str.contains("`") & ~pl.col(col).cast(pl.Utf8).str.contains("N/A")
    for col in df.columns
]
# Combine all conditions with logical AND.
filter_condition = reduce(lambda acc, expr: acc & expr, conditions)
df = df.filter(filter_condition)

# Convert all other columns except "Fecha" to Float64.
other_cols = [col for col in df.columns if col != "Date"]
df = df.with_columns([pl.col(c).cast(pl.Float64) for c in other_cols])

print(df.head)

# Save as Parquet file
df.write_parquet("data.parquet")