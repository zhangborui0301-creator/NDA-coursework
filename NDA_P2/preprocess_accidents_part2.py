from pathlib import Path
import pandas as pd

from config_part2 import RAW_ACCIDENTS_DIR, PROCESSED_DIR, YEARS, SELECTED_AREA_NAME, BBOX_BNG


def load_one_year(year: int) -> pd.DataFrame:
    """
    Load one accident file for a given year.
    Expected filenames: 2012.csv, 2013.csv, ..., 2016.csv
    """
    file_path = RAW_ACCIDENTS_DIR / f"{year}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    df = pd.read_csv(file_path, dtype=str, encoding="cp1252")
    df.columns = [str(c).strip() for c in df.columns]

    # Harmonise the small set of column name differences across 2012-2016
    rename_map = {}

    if "Reference Number" in df.columns:
        rename_map["Reference Number"] = "reference_number"

    if "Accident Date" in df.columns:
        rename_map["Accident Date"] = "accident_date"
    elif "Date" in df.columns:
        rename_map["Date"] = "accident_date"

    if "Easting" in df.columns:
        rename_map["Easting"] = "easting"
    elif "Grid Ref: Easting" in df.columns:
        rename_map["Grid Ref: Easting"] = "easting"

    if "Northing" in df.columns:
        rename_map["Northing"] = "northing"
    elif "Grid Ref: Northing" in df.columns:
        rename_map["Grid Ref: Northing"] = "northing"

    if "Time (24hr)" in df.columns:
        rename_map["Time (24hr)"] = "time_24hr"
    elif "Time" in df.columns:
        rename_map["Time"] = "time_24hr"

    df = df.rename(columns=rename_map)

    required_cols = ["reference_number", "accident_date", "easting", "northing"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {missing}")

    keep_cols = ["reference_number", "accident_date", "easting", "northing"]
    if "time_24hr" in df.columns:
        keep_cols.append("time_24hr")

    df = df[keep_cols].copy()
    df["year"] = year
    df["source_file"] = file_path.name

    # Basic cleaning
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "NaN", "None"]), col] = pd.NA

    df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
    df["northing"] = pd.to_numeric(df["northing"], errors="coerce")
    df["accident_date"] = pd.to_datetime(df["accident_date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["reference_number", "accident_date", "easting", "northing"]).copy()

    # Create a year-specific unique accident identifier
    df["accident_uid"] = df["year"].astype(str) + "_" + df["reference_number"].astype(str)

    return df


def load_all_years() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        print(f"Reading {year}.csv")
        frames.append(load_one_year(year))
    return pd.concat(frames, ignore_index=True)


def deduplicate_accidents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate at accident level using accident_uid.
    """
    df = df.drop_duplicates(subset=["accident_uid"]).copy()
    df = df.reset_index(drop=True)
    return df


def filter_selected_area(df: pd.DataFrame) -> pd.DataFrame:
    west = BBOX_BNG["west"]
    east = BBOX_BNG["east"]
    south = BBOX_BNG["south"]
    north = BBOX_BNG["north"]

    area_df = df[
        (df["easting"] >= west) &
        (df["easting"] <= east) &
        (df["northing"] >= south) &
        (df["northing"] <= north)
    ].copy()

    area_df["selected_area_name"] = SELECTED_AREA_NAME
    return area_df


def build_yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby("year")
        .size()
        .reset_index(name="unique_accidents_in_selected_area")
        .sort_values("year")
        .reset_index(drop=True)
    )

    total_row = pd.DataFrame(
        [{"year": "TOTAL", "unique_accidents_in_selected_area": len(df)}]
    )

    return pd.concat([counts, total_row], ignore_index=True)


def save_outputs(
    all_standardised_df: pd.DataFrame,
    deduplicated_df: pd.DataFrame,
    area_df: pd.DataFrame,
    yearly_counts_df: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_standardised_df.to_csv(
        PROCESSED_DIR / "accidents_all_years_standardised.csv",
        index=False
    )

    deduplicated_df.to_csv(
        PROCESSED_DIR / "accidents_all_years_deduplicated.csv",
        index=False
    )

    area_df.to_csv(
        PROCESSED_DIR / "accidents_in_selected_area_2012_2016.csv",
        index=False
    )

    yearly_counts_df.to_csv(
        PROCESSED_DIR / "yearly_accident_counts_in_area.csv",
        index=False
    )


def main():
    all_df = load_all_years()
    dedup_df = deduplicate_accidents(all_df)
    area_df = filter_selected_area(dedup_df)
    yearly_counts_df = build_yearly_counts(area_df)

    if all_df.empty:
        raise ValueError("The combined dataframe is empty.")
    if dedup_df.empty:
        raise ValueError("The deduplicated accident dataframe is empty.")
    if area_df.empty:
        raise ValueError("No accidents fall inside the selected area. Check the bbox or coordinate fields.")

    save_outputs(all_df, dedup_df, area_df, yearly_counts_df)

    print("=" * 70)
    print("Part 2 accident preprocessing completed")
    print("=" * 70)
    print(f"All rows after standardisation: {len(all_df):,}")
    print(f"Unique accidents after deduplication: {len(dedup_df):,}")
    print(f"Unique accidents in selected area: {len(area_df):,}")
    print("\nYearly counts in selected area:")
    print(yearly_counts_df.to_string(index=False))


if __name__ == "__main__":
    main()