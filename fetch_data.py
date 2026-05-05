import datetime as dt
import io
from functools import reduce

import polars as pl
import requests


SOURCE_URL = "https://bcrdgdcprod.blob.core.windows.net/documents/entorno-internacional/documents/Serie_Historica_Spread_del_EMBI.xlsx"
SHEET_NAME = "Serie Histórica"
OUTPUT_PATH = "data.parquet"
EXPECTED_DATE_COLUMN = "Fecha"


def parse_mixed_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dt.datetime, dt.date)):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    patterns = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%b-%y",
        "'%d-%b-%y",
        "%d-%b-%Y",
    )
    for pattern in patterns:
        try:
            return dt.datetime.strptime(text, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def download_source() -> bytes:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    return response.content


def read_source_excel(content: bytes) -> pl.DataFrame:
    raw = pl.read_excel(
        io.BytesIO(content),
        sheet_name=SHEET_NAME,
        columns=list(range(20)),
    )
    if raw.height < 2:
        raise ValueError("Source workbook does not contain enough rows.")

    header = [str(col).strip() for col in raw.row(0)]
    raw.columns = header
    return raw.slice(1)


def clean_embi_data(df: pl.DataFrame) -> pl.DataFrame:
    if EXPECTED_DATE_COLUMN not in df.columns:
        raise ValueError(f"Source workbook is missing the expected '{EXPECTED_DATE_COLUMN}' column.")

    df = df.with_columns(
        pl.col(EXPECTED_DATE_COLUMN)
        .map_elements(parse_mixed_date, return_dtype=pl.Utf8)
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        .alias("Date")
    ).drop(EXPECTED_DATE_COLUMN)

    value_columns = [col for col in df.columns if col != "Date"]
    if not value_columns:
        raise ValueError("Source workbook has no EMBI spread columns.")

    invalid_tokens = ["`", "N/A"]
    valid_conditions = [pl.col("Date").is_not_null()]
    for col in value_columns:
        as_text = pl.col(col).cast(pl.Utf8).str.strip_chars()
        valid_conditions.extend(~as_text.str.contains(token, literal=True) for token in invalid_tokens)
        valid_conditions.append(as_text != "")

    df = df.filter(reduce(lambda acc, expr: acc & expr, valid_conditions))
    df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in value_columns])
    df = df.drop_nulls(["Date", *value_columns])

    return (
        df.select(["Date", *value_columns])
        .unique(subset=["Date"], keep="last")
        .sort("Date")
    )


def main() -> None:
    content = download_source()
    df = clean_embi_data(read_source_excel(content))
    if df.height == 0:
        raise ValueError("Cleaned EMBI dataset is empty.")

    df.write_parquet(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}: {df.height} rows, {len(df.columns) - 1} series.")
    print(f"Latest observation: {df.select(pl.col('Date').max()).item()}")


if __name__ == "__main__":
    main()
