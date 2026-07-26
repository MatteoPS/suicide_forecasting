"""Shared pytest configuration and fixtures.

Two things drive the design of this file.

1. `src.utils.config` raises at *import* time when the Census / Redivis
    credentials are missing, so stub credentials have to be in the environment
    before any test module imports anything under `src`. conftest.py is imported
    before test modules, which makes this the right place for it. `setdefault`
    plus `load_dotenv`'s no-override behaviour means a real `.env` sitting next
    to the repo can never leak a live API key into the suite.

2. No test in this suite reads a real dataset. NVDRS and BRFSS are large and
    restricted; every fixture here builds a handful of synthetic rows with the
    same *shape* as the real files. Anything that would hit the Census API,
    Redivis, or a multi-GB CSV is monkeypatched at the boundary.
"""
import os

os.environ.setdefault("CENSUS_API_KEY", "test-census-key")
os.environ.setdefault("REDIVIS_USERNAME", "test-user")
os.environ.setdefault("REDIVIS_ORGANIZATION", "test-org")
os.environ.setdefault("BRFSS_PATH", "/nonexistent/brfss")

import matplotlib

matplotlib.use("Agg")  # no interactive windows during tests

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


# --------------------------------------------------------------------------
# NVDRS-shaped synthetic data
# --------------------------------------------------------------------------

@pytest.fixture
def nvdrs_frame():
    """A minimal NVDRS-shaped frame covering every incident category branch.

    Five incidents, deliberately chosen:
        1  single suicide                       -> kept
        2  single suicide                       -> kept
        3  homicide followed by suicide, 2 rows -> only the 'Both victim and suspect' row should survive
        4  single homicide                      -> dropped (no 'suicide' in label)
        5  multiple suicides, 2 rows            -> both are genuine suicides
    """
    rows = [
        # IncidentID, Category, PersonType, DeathDate, InjuryDate, State, FIPS, InjuryZip, ResidenceZip
        ("1", "Single suicide", "Victim", "2015-03-01", "2015-02-28", "New York", "36061", "10001", "10001"),
        ("2", "Single suicide", "Victim", "2015-03-01", "2015-03-01", "New York", "36061", "10002", "10002"),
        ("3", "Homicide followed by suicide", "Victim", "2016-07-04", "2016-07-04", "Utah", "49035", "84101", "84101"),
        ("3", "Homicide followed by suicide", "Both victim and suspect", "2016-07-04", "2016-07-04", "Utah", "49035", "84102", "84102"),
        ("4", "Single homicide", "Victim", "2016-08-01", "2016-08-01", "Utah", "49035", "84103", "84103"),
        ("5", "Multiple suicides", "Victim", "2017-01-15", "2017-01-15", "Iowa", "19153", "50301", "50301"),
        ("5", "Multiple suicides", "Victim", "2017-01-15", "2017-01-15", "Iowa", "19153", "50302", "50302"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "IncidentID", "IncidentCategory_c", "PersonType", "DeathDate",
            "InjuryDate", "DeathState", "DeathFIPS", "InjuryZip", "ResidenceZip",
        ],
    )
    # Columns the aggregators expect alongside the ones above.
    df["InjuryState"] = df["DeathState"]
    df["InjuryFIPS"] = df["DeathFIPS"]
    df["DeathDate_myr"] = pd.to_datetime(df["DeathDate"]).dt.strftime("%Y-%m-01")
    df["Sex"] = "Male"
    df["AgeYears_c"] = "40"
    return df


@pytest.fixture
def suicides_frame(nvdrs_frame):
    """`nvdrs_frame` already passed through the suicide filter."""
    from src.etl.transform import filter_nvdrs_suicides

    return filter_nvdrs_suicides(nvdrs_frame)


# --------------------------------------------------------------------------
# SaTScan artifact files (.cas / .geo / .pop)
# --------------------------------------------------------------------------

@pytest.fixture
def write_satscan_files(tmp_path):
    """Factory writing headerless, whitespace-delimited SaTScan input files.

    That format is what `run_small_st_dbscan` reads, so the fixture has to
    match it exactly (no header row, whitespace separator).
    """

    def _write(cases, geo, pop=None, stem="test"):
        paths = {}

        cas_path = tmp_path / f"{stem}.cas"
        cas_path.write_text(
            "\n".join(f"{z} {n} {d}" for z, n, d in cases) + "\n"
        )
        paths["cas"] = cas_path

        geo_path = tmp_path / f"{stem}.geo"
        geo_path.write_text(
            "\n".join(f"{z} {lat} {lon}" for z, lat, lon in geo) + "\n"
        )
        paths["geo"] = geo_path

        if pop is not None:
            pop_path = tmp_path / f"{stem}.pop"
            pop_path.write_text(
                "\n".join(f"{z} {y} {p}" for z, y, p in pop) + "\n"
            )
            paths["pop"] = pop_path

        return paths

    return _write


# --------------------------------------------------------------------------
# Census API doubles
# --------------------------------------------------------------------------

class FakeResponse:
    """Stand-in for `requests.Response` covering only what ingest.py uses."""

    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def fake_response():
    return FakeResponse


# --------------------------------------------------------------------------
# Redivis doubles
# --------------------------------------------------------------------------

class FakeProps(dict):
    """redivis objects expose `.properties` as a dict-like with `.get`."""


class FakeVariable:
    def __init__(self, name, type_="string"):
        self.name = name
        self.properties = FakeProps(type=type_, label=name, description="")


class FakeTable:
    def __init__(self, name, ref, variables, rows=100):
        self.name = name
        self.qualified_reference = f"org.dataset.{ref}"
        self.properties = FakeProps(
            numRows=rows, variableCount=len(variables), numBytes=rows * 10,
            updatedAt=1_600_000_000_000, description="",
        )
        self._variables = [FakeVariable(v) for v in variables]

    def list_variables(self):
        return self._variables


class FakeDataset:
    def __init__(self, name, ref, tables):
        self.name = name
        self.qualified_reference = f"org.{ref}"
        self.properties = FakeProps(updatedAt=1_600_000_000_000)
        self._tables = tables
        self.queries = []          # every SQL string this dataset was asked to run
        self.query_results = {}    # substring -> DataFrame, or substring -> Exception

    def list_tables(self):
        return self._tables

    def table(self, ref):
        for t in self._tables:
            if t.qualified_reference.split(".")[-1] == ref:
                return t
        raise KeyError(ref)

    def query(self, sql):
        self.queries.append(sql)
        outer = self

        class _Query:
            def to_pandas_dataframe(self):
                for needle, result in outer.query_results.items():
                    if needle in sql:
                        if isinstance(result, Exception):
                            raise result
                        return result.copy()
                return pd.DataFrame({"KEY": ["k1"], "ZIP": ["10001"]})

        return _Query()


class FakeOrg:
    def __init__(self, datasets):
        self._datasets = datasets
        self.list_datasets_calls = 0

    def list_datasets(self):
        self.list_datasets_calls += 1
        return self._datasets

    def dataset(self, ref):
        for d in self._datasets:
            if d.qualified_reference.split(".")[-1] == ref:
                return d
        raise KeyError(ref)

    def exists(self):
        return True


@pytest.fixture
def fake_org():
    """A two-table HCUP-like organization.

    `ny_sedd_2015q1q3_core` has ECODE1/ECODE2 and an AGE column;
    `ny_sedd_2015q4_core` has I10_DX1/I10_DX2 and no AGE. The asymmetry is the
    point: it exercises the `NULL AS <col>` padding in `fetch_hcup`.
    """
    core_9 = FakeTable(
        "NY SEDD 2015q1q3 CORE", "ny_sedd_2015q1q3_core:aaaa",
        ["KEY", "AGE", "ZIP", "AMONTH", "AYEAR", "DSHOSPID", "HOSPID", "ECODE1", "ECODE2"],
    )
    core_9_prior = FakeTable(
        "NY SEDD 2014 CORE", "ny_sedd_2014_core:eeee",
        ["KEY", "AGE", "ZIP", "AMONTH", "AYEAR", "DSHOSPID", "HOSPID", "ECODE1"],
    )
    core_10 = FakeTable(
        "NY SEDD 2015q4 CORE", "ny_sedd_2015q4_core:bbbb",
        ["KEY", "ZIP", "AMONTH", "AYEAR", "DSHOSPID", "HOSPID", "I10_DX1", "I10_DX2"],
    )
    ahal = FakeTable(
        "NY SEDD 2015 AHAL", "ny_sedd_2015_ahal:cccc",
        ["HOSPID", "YEAR", "HFIPSSTCO"],
    )
    ahal_keyed = FakeTable(
        "IA SEDD 2015 AHAL", "ia_sedd_2015_ahal:dddd",
        ["KEY", "YEAR", "HFIPSSTCO"],
    )
    ny = FakeDataset(
        "New York HCUP", "new_york_hcup:1234", [core_9_prior, core_9, core_10, ahal]
    )
    ia = FakeDataset("Iowa HCUP", "iowa_hcup:5678", [ahal_keyed])
    return FakeOrg([ny, ia])


@pytest.fixture
def catalog(fake_org, monkeypatch):
    """A `RedivisCatalog` wired to `fake_org` instead of the live API."""
    import redivis

    from src.utils.redivis_client import RedivisCatalog

    monkeypatch.setattr(redivis, "organization", lambda name: fake_org)
    return RedivisCatalog("test-org")


# --------------------------------------------------------------------------
# uszipcode doubles
# --------------------------------------------------------------------------

class FakeZipResult:
    def __init__(self, lat=None, lng=None, major_city=None, county=None,
                state=None, zipcode_type="Standard", population=None):
        self.lat = lat
        self.lng = lng
        self.major_city = major_city
        self.county = county
        self.state = state
        self.zipcode_type = zipcode_type
        self.population = population


class FakeSearchEngine:
    """Drop-in for `uszipcode.SearchEngine` backed by an in-memory dict.

    The real thing downloads and opens a SQLite database on construction,
    which is far too heavy (and too networked) for a unit test.
    """

    lookup = {}

    def __init__(self, *args, **kwargs):
        pass

    def by_zipcode(self, zipcode):
        return self.lookup.get(str(zipcode))


@pytest.fixture
def fake_search_engine():
    def _make(lookup):
        cls = type("PatchedSearchEngine", (FakeSearchEngine,), {"lookup": dict(lookup)})
        return cls

    return _make


# --------------------------------------------------------------------------
# prep_satscan_gui environment
#
# Shared by test_prep_satscan.py (which tests the function's branches) and
# test_data_pipeline.py (which tests properties of what it produces).
# --------------------------------------------------------------------------

@pytest.fixture
def nvdrs_rows():
    """Every row exists to exercise one branch of prep_satscan_gui."""
    return pd.DataFrame(
        [
            # normal case, injury ZIP present
            ("1", "Single suicide", "Victim", "2015-03-01", "10001", "10001"),
            # blank injury ZIP -> falls back to residence
            ("2", "Single suicide", "Victim", "2015-03-02", "", "10002"),
            # sentinel injury ZIP -> falls back to residence
            ("3", "Single suicide", "Victim", "2015-03-03", "99999", "10003"),
            # before the 2010-01-01 study start
            ("4", "Single suicide", "Victim", "2009-06-01", "10001", "10001"),
            # ZIP with no coordinates anywhere (a PO Box range)
            ("5", "Single suicide", "Victim", "2015-03-04", "00501", "00501"),
            # not a suicide
            ("6", "Single homicide", "Victim", "2015-03-05", "10001", "10001"),
        ],
        columns=["IncidentID", "IncidentCategory_c", "PersonType",
                "DeathDate", "InjuryZip", "ResidenceZip"],
    )


@pytest.fixture
def gazetteer_file(tmp_path):
    """Census Gazetteer extract. The real file has trailing whitespace in its
    header, which `prep_satscan_gui` strips; reproduce that here."""
    path = tmp_path / "gazetteer.txt"
    path.write_text(
        "GEOID\tINTPTLAT\tINTPTLONG   \n"
        "10001\t40.7500\t-73.9900\n"
        "10002\t40.7510\t-73.9900\n"
        "10003\t40.7520\t-73.9900\n"
    )
    return path


@pytest.fixture
def zcta_crosswalk_file(tmp_path):
    """Identity 2010->2020 mapping, so harmonization is a no-op and the other
    assertions are not confounded by apportionment."""
    path = tmp_path / "zcta_crosswalk.txt"
    rows = "\n".join(
        f"{z}|{z}|1000|1000" for z in ("10001", "10002", "10003", "00501")
    )
    path.write_text(
        "GEOID_ZCTA5_10|GEOID_ZCTA5_20|AREALAND_ZCTA5_10|AREALAND_PART\n" + rows + "\n"
    )
    return path


@pytest.fixture
def prep_env(tmp_path, monkeypatch, nvdrs_rows, gazetteer_file,
            zcta_crosswalk_file, fake_search_engine):
    """Wires every external dependency of prep_satscan_gui to a local stub.

    Mutate `prep_env['state']['populations']` before calling the function to
    change what the stubbed Census API returns.
    """
    from src.etl import transform
    from src.geospatial import prep_satscan

    state = {"populations": {"10001": 10_000, "10002": 10_000, "10003": 10_000}}

    monkeypatch.setattr(prep_satscan, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prep_satscan, "load_nvdrs", lambda **kwargs: nvdrs_rows.copy())

    def fake_fetch_census(variables_dict, years, geo_level="county", states="*"):
        rows = [
            {"ZTCA5": f"ZCTA5 {z}", "Population": str(p), "ZIP": z,
            "Year": y, "state": None}
            for y in years for z, p in state["populations"].items()
        ]
        # fetch_census hands back the raw Census column name; prep renames it.
        return pd.DataFrame(rows).rename(columns={"ZIP": "zip code tabulation area"})

    monkeypatch.setattr(prep_satscan, "fetch_census", fake_fetch_census)
    monkeypatch.setattr(prep_satscan, "get_data_path", lambda *a, **k: gazetteer_file)
    monkeypatch.setattr(transform, "get_data_path", lambda *a, **k: zcta_crosswalk_file)
    monkeypatch.setattr(prep_satscan, "SearchEngine", fake_search_engine({}))

    return {"out_dir": tmp_path / "data" / "processed" / "satscan",
            "root": tmp_path, "state": state}
