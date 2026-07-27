"""Readers for the SaTScan artifact formats.

SaTScan's .cas / .geo / .pop files are headerless and whitespace-delimited,
and the ZIP column has to be forced back to a zero-padded string on every
read - pandas otherwise turns 01001 into the integer 1001 and every downstream
merge silently returns nothing.
"""
import pandas as pd


def clean_zip(series: pd.Series) -> pd.Series:
    """Standardize a ZIP column into 5-digit strings.

    Handles the three shapes these files arrive in: integers that lost their
    leading zero, floats that gained a '.0', and strings with stray whitespace.
    """
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(5)
    )


def read_satscan_cases(path) -> pd.DataFrame:
    """Read a .cas file into columns [zip, n_case, date]."""
    cases = pd.read_csv(path, header=None, names=["zip", "n_case", "date"], sep=r"\s+")
    cases["zip"] = clean_zip(cases["zip"])
    cases["date"] = pd.to_datetime(cases["date"])
    return cases


def read_satscan_geo(path) -> pd.DataFrame:
    """Read a .geo file into columns [zip, lat, lon]."""
    geo = pd.read_csv(path, header=None, names=["zip", "lat", "lon"], sep=r"\s+")
    geo["zip"] = clean_zip(geo["zip"])
    return geo


def read_satscan_pop(path) -> pd.DataFrame:
    """Read a .pop file into columns [zip, year, population]."""
    pop = pd.read_csv(path, header=None, names=["zip", "year", "population"], sep=r"\s+")
    pop["zip"] = clean_zip(pop["zip"])
    return pop


def daily_case_totals(cases: pd.DataFrame) -> pd.Series:
    """Collapse a case frame to one total per date."""
    return cases.groupby("date")["n_case"].sum()
