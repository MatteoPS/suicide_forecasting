"""Tests for `src.etl.ingest`.

Nothing here touches a real dataset or a live API:

* `load_nvdrs` runs against a three-row cp1252 file written to tmp_path. The
  real extract is multi-GB and restricted.
* `load_brfss` never actually parses an XPORT file; `pd.read_sas` is stubbed.
* Both Census fetchers run against a stubbed `requests.get`.
* The HCUP fetchers run against the fake Redivis catalog in conftest, which
  records the SQL they generate. The SQL string builder is the part worth
  testing - it is where the year/column/ICD mismatches actually bite.
"""
import warnings

import pandas as pd
import pytest

from src.etl import ingest
from src.etl.ingest import (
    fetch_2010_decennial_population,
    fetch_census,
    fetch_hcup,
    fetch_hcup_ahal,
    load_brfss,
    load_nvdrs,
)


# ==========================================================================
# load_nvdrs
# ==========================================================================

@pytest.fixture
def tiny_nvdrs_csv(tmp_path, monkeypatch):
    """A cp1252 NVDRS stand-in containing a byte that is not valid UTF-8.

    U+2019 (curly apostrophe) encodes to the single byte 0x92 in cp1252, which
    is invalid UTF-8. It appears constantly in NVDRS narrative text and is why
    `load_nvdrs` pins encoding='cp1252'.
    """
    path = tmp_path / "nvdrs.csv"
    body = (
        "IncidentID,IncidentCategory_c,DeathDate,DeathFIPS,NarrativeCME\n"
        "1,Single suicide,2015-03-01,36061,victim’s note\n"
        "2,Single suicide,2015-03-02,06037,plain text\n"
        "3,Single homicide,2015-03-03,49035,plain text\n"
    )
    path.write_bytes(body.encode("cp1252"))
    assert b"\x92" in path.read_bytes(), "fixture must actually be non-UTF-8"
    monkeypatch.setattr(ingest, "get_data_path", lambda *a, **k: path)
    return path


def test_load_nvdrs_reads_windows_encoded_text(tiny_nvdrs_csv):
    df = load_nvdrs("nvdrs", "raw")
    assert len(df) == 3
    assert df.loc[0, "NarrativeCME"] == "victim’s note"


def test_load_nvdrs_returns_everything_as_string(tiny_nvdrs_csv):
    """dtype=str is load-bearing: FIPS and ZIP codes lose leading zeros
    the moment pandas infers them as integers."""
    df = load_nvdrs("nvdrs", "raw")
    assert (df.dtypes == object).all()
    assert df.loc[1, "DeathFIPS"] == "06037"


def test_load_nvdrs_respects_nrows(tiny_nvdrs_csv):
    assert len(load_nvdrs("nvdrs", "raw", nrows=2)) == 2


def test_load_nvdrs_respects_usecols(tiny_nvdrs_csv):
    df = load_nvdrs("nvdrs", "raw", usecols=["IncidentID", "DeathDate"])
    assert list(df.columns) == ["IncidentID", "DeathDate"]


def test_load_nvdrs_nrows_zero_gives_column_names_only(tiny_nvdrs_csv):
    """The notebooks use nrows=0 to inspect the schema without loading 2 GB."""
    df = load_nvdrs("nvdrs", "raw", nrows=0)
    assert len(df) == 0
    assert "IncidentCategory_c" in df.columns


def test_load_nvdrs_passes_file_key_and_folder_through(tmp_path, monkeypatch):
    seen = {}

    def spy(file_key, data_folder):
        seen["args"] = (file_key, data_folder)
        path = tmp_path / "x.csv"
        path.write_text("a\n1\n")
        return path

    monkeypatch.setattr(ingest, "get_data_path", spy)
    load_nvdrs("nvdrs_narratives", "processed")
    assert seen["args"] == ("nvdrs_narratives", "processed")


# ==========================================================================
# load_brfss
# ==========================================================================

@pytest.fixture
def stub_read_sas(monkeypatch):
    calls = []

    def fake(path, format=None):
        calls.append((path, format))
        return pd.DataFrame({"_STATE": [36.0], "ADDEPEV3": [1.0]})

    monkeypatch.setattr(ingest.pd, "read_sas", fake)
    return calls


def test_load_brfss_reads_base_file(tmp_path, stub_read_sas):
    (tmp_path / "LLCP2024.XPT").write_bytes(b"")
    df = load_brfss(2024, brfss_path=str(tmp_path))
    assert len(df) == 1
    assert stub_read_sas[0][1] == "xport"
    assert stub_read_sas[0][0].endswith("LLCP2024.XPT")


def test_load_brfss_missing_file_raises(tmp_path, stub_read_sas):
    with pytest.raises(FileNotFoundError, match="LLCP2024.XPT"):
        load_brfss(2024, brfss_path=str(tmp_path))


def test_load_brfss_warns_about_versioned_files(tmp_path, stub_read_sas, capsys):
    """BRFSS ships LLCP2020V1.XPT alongside LLCP2020.XPT in some years, and
    they are not interchangeable. The loader must say which one it picked."""
    (tmp_path / "LLCP2020.XPT").write_bytes(b"")
    (tmp_path / "LLCP2020V1.XPT").write_bytes(b"")
    load_brfss(2020, brfss_path=str(tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "LLCP2020V1.XPT" in out
    assert "LLCP2020.XPT" in out


def test_load_brfss_falls_back_to_env_path(tmp_path, monkeypatch, stub_read_sas):
    (tmp_path / "LLCP2024.XPT").write_bytes(b"")
    monkeypatch.setattr(ingest, "BRFSS_PATH", str(tmp_path))
    assert len(load_brfss(2024)) == 1


def test_load_brfss_without_any_path_raises_valueerror(monkeypatch):
    monkeypatch.setattr(ingest, "BRFSS_PATH", None)
    with pytest.raises(ValueError, match="BRFSS_PATH is missing"):
        load_brfss(2024)


def test_load_brfss_accepts_string_year(tmp_path, stub_read_sas):
    (tmp_path / "LLCP2024.XPT").write_bytes(b"")
    assert len(load_brfss("2024", brfss_path=str(tmp_path))) == 1


# ==========================================================================
# fetch_census
# ==========================================================================

@pytest.fixture
def census_stub(monkeypatch, fake_response):
    """Records every request and replies with a two-row ACS payload."""
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append({"url": url, "params": params})
        return fake_response(
            [
                ["NAME", "DP05_0001E", "zip code tabulation area"],
                ["ZCTA5 10001", "21102", "10001"],
                ["ZCTA5 10002", "81410", "10002"],
            ]
        )

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    return calls


VARS = {"NAME": "ZCTA5", "DP05_0001E": "Population"}


def test_fetch_census_renames_columns(census_stub):
    df = fetch_census(VARS, [2020], geo_level="zip")
    assert "Population" in df.columns
    assert "ZCTA5" in df.columns


def test_fetch_census_tags_each_row_with_its_year(census_stub):
    df = fetch_census(VARS, [2019, 2020], geo_level="zip")
    assert sorted(df["Year"].unique()) == [2019, 2020]
    assert len(df) == 4, "one request per year, concatenated"


def test_fetch_census_requests_the_right_variables(census_stub):
    fetch_census(VARS, [2020], geo_level="zip")
    assert census_stub[0]["params"]["get"] == "NAME,DP05_0001E"


def test_fetch_census_zip_geography(census_stub):
    fetch_census(VARS, [2020], geo_level="zip")
    assert census_stub[0]["params"]["for"] == "zip code tabulation area:*"
    assert "in" not in census_stub[0]["params"]


def test_fetch_census_county_geography_defaults_to_all_states(census_stub):
    fetch_census(VARS, [2020], geo_level="county")
    assert census_stub[0]["params"]["for"] == "county:*"
    assert "in" not in census_stub[0]["params"]


def test_fetch_census_county_geography_scopes_to_states(census_stub):
    fetch_census(VARS, [2020], geo_level="county", states=["36", "34"])
    assert census_stub[0]["params"]["in"] == "state:36,34"


def test_fetch_census_state_geography(census_stub):
    fetch_census(VARS, [2020], geo_level="state", states=["36", "34"])
    assert census_stub[0]["params"]["for"] == "state:36,34"


def test_fetch_census_accepts_a_state_string(census_stub):
    fetch_census(VARS, [2020], geo_level="state", states="36")
    assert census_stub[0]["params"]["for"] == "state:36"


def test_fetch_census_uses_the_acs5_profile_endpoint(census_stub):
    fetch_census(VARS, [2020], geo_level="zip")
    assert census_stub[0]["url"] == "https://api.census.gov/data/2020/acs/acs5/profile"


def test_fetch_census_skips_failed_years(monkeypatch, fake_response, capsys):
    def fake_get(url, params=None, **kwargs):
        if "/2020/" in url:
            return fake_response(status_code=404, text="not available")
        return fake_response([["NAME"], ["ZCTA5 10001"]])

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    df = fetch_census({"NAME": "ZCTA5"}, [2019, 2020], geo_level="zip")
    assert df["Year"].unique().tolist() == [2019]
    assert "not available" in capsys.readouterr().out


def test_fetch_census_returns_empty_frame_when_everything_fails(monkeypatch, fake_response):
    monkeypatch.setattr(
        ingest.requests, "get",
        lambda *a, **k: fake_response(status_code=500, text="boom"),
    )
    df = fetch_census({"NAME": "ZCTA5"}, [2019, 2020])
    assert df.empty
    assert isinstance(df, pd.DataFrame)


# ==========================================================================
# fetch_2010_decennial_population
# ==========================================================================

def test_fetch_2010_strips_the_zcta5_prefix(monkeypatch, fake_response):
    monkeypatch.setattr(
        ingest.requests, "get",
        lambda *a, **k: fake_response(
            [
                ["NAME", "P001001", "zip code tabulation area"],
                ["ZCTA5 10001", "21102", "10001"],
            ]
        ),
    )
    df = fetch_2010_decennial_population()
    assert df.loc[0, "ZIP"] == "10001"


def test_fetch_2010_matches_fetch_census_output_shape(monkeypatch, fake_response):
    monkeypatch.setattr(
        ingest.requests, "get",
        lambda *a, **k: fake_response(
            [["NAME", "P001001", "zip code tabulation area"], ["ZCTA5 10001", "21102", "10001"]]
        ),
    )
    df = fetch_2010_decennial_population()
    assert list(df.columns) == ["ZIP", "Population", "Year", "state"]
    assert df.loc[0, "Year"] == 2010
    assert df.loc[0, "Population"] == 21102


def test_fetch_2010_raises_on_api_error(monkeypatch, fake_response):
    monkeypatch.setattr(
        ingest.requests, "get",
        lambda *a, **k: fake_response(status_code=403, text="forbidden"),
    )
    with pytest.raises(ConnectionError, match="403"):
        fetch_2010_decennial_population()


# ==========================================================================
# fetch_hcup - SQL generation
# ==========================================================================

ICD9_COLS = ["KEY", "AGE", "ZIP", "AMONTH", "AYEAR", "DSHOSPID", "HOSPID"]
E_CODES = [f"E{i}" for i in range(950, 960)]


def _queries(fake_org, dataset_ref="new_york_hcup:1234"):
    return fake_org.dataset(dataset_ref).queries


def test_fetch_hcup_selects_requested_columns(catalog, fake_org):
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", E_CODES)
    sql = _queries(fake_org)[0]
    for col in ICD9_COLS:
        assert col in sql


def test_fetch_hcup_pads_absent_columns_with_null(catalog, fake_org):
    """The 2015q4 CORE table has no AGE column. Selecting it verbatim would
    fail the whole year; the builder must emit `NULL AS AGE` instead so the
    per-year frames stay concatenable."""
    fetch_hcup(catalog, "New York", "sedd", ["2015q4"], ICD9_COLS, "I10_DX", ["T1491XA"])
    sql = _queries(fake_org)[0]
    assert "NULL AS AGE" in sql


def test_fetch_hcup_anchors_the_icd_regex(catalog, fake_org):
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", ["E950", "E951"])
    sql = _queries(fake_org)[0]
    assert "r'^(E950|E951)'" in sql


def test_fetch_hcup_filters_on_every_icd_column(catalog, fake_org):
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", E_CODES)
    sql = _queries(fake_org)[0]
    assert "REGEXP_CONTAINS(CAST(ECODE1 AS STRING)" in sql
    assert "REGEXP_CONTAINS(CAST(ECODE2 AS STRING)" in sql
    assert " OR " in sql


def test_fetch_hcup_casts_icd_columns_to_string(catalog, fake_org):
    """HCUP types drift between numeric and string across states and years;
    the CAST is what stops BigQuery rejecting the REGEXP."""
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", E_CODES)
    assert "CAST(" in _queries(fake_org)[0]


def test_fetch_hcup_skips_tables_without_matching_icd_columns(catalog, fake_org):
    """Year pattern '2015' matches both CORE tables, but only the ICD-9 one
    carries ECODE columns, so exactly one query should be issued."""
    fetch_hcup(catalog, "New York", "sedd", ["2015"], ICD9_COLS, "ECODE", E_CODES)
    queries = _queries(fake_org)
    assert len(queries) == 1
    assert "ny_sedd_2015q1q3_core" in queries[0]


def test_fetch_hcup_qualifies_the_table_reference(catalog, fake_org):
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", E_CODES)
    assert "`test-org.new_york_hcup:1234.ny_sedd_2015q1q3_core:aaaa`" in _queries(fake_org)[0]


def test_fetch_hcup_can_return_the_icd_columns(catalog, fake_org):
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE",
               E_CODES, return_icd_cols=True)
    sql = _queries(fake_org)[0]
    assert "SELECT" in sql and "ECODE1, ECODE2" in sql.split("FROM")[0]


def test_fetch_hcup_omits_icd_columns_by_default(catalog, fake_org):
    fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", E_CODES)
    select_clause = _queries(fake_org)[0].split("FROM")[0]
    assert "ECODE1" not in select_clause


def test_fetch_hcup_no_matching_tables_raises(catalog):
    with pytest.raises(ValueError, match="No tables matched"):
        fetch_hcup(catalog, "New York", "sedd", ["1999"], ICD9_COLS, "ECODE", E_CODES)


def test_fetch_hcup_unknown_state_raises(catalog):
    with pytest.raises(IndexError):
        fetch_hcup(catalog, "Atlantis", "sedd", ["2015"], ICD9_COLS, "ECODE", E_CODES)


def test_fetch_hcup_warns_and_continues_when_one_year_fails(catalog, fake_org):
    """A single oversized or access-denied year must not sink the whole state."""
    ny = fake_org.dataset("new_york_hcup:1234")
    ny.query_results = {
        "ny_sedd_2014_core": RuntimeError("result too large"),
        "ny_sedd_2015q1q3_core": pd.DataFrame({"KEY": ["a", "b"]}),
    }
    with pytest.warns(UserWarning, match="too large or access was denied"):
        df = fetch_hcup(catalog, "New York", "sedd", ["2014", "2015q1q3"],
                        ICD9_COLS, "ECODE", E_CODES)
    assert len(df) == 2, "the surviving year must still come back"


def test_fetch_hcup_raises_when_every_table_fails(catalog, fake_org):
    ny = fake_org.dataset("new_york_hcup:1234")
    ny.query_results = {"ny_sedd": RuntimeError("denied")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="No data could be retrieved"):
            fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS,
                       "ECODE", E_CODES)


def test_fetch_hcup_concatenates_multiple_tables(catalog, fake_org):
    ny = fake_org.dataset("new_york_hcup:1234")
    ny.query_results = {
        "ny_sedd_2015q1q3_core": pd.DataFrame({"KEY": ["a", "b"]}),
    }
    df = fetch_hcup(catalog, "New York", "sedd", ["2015q1q3"], ICD9_COLS, "ECODE", E_CODES)
    assert len(df) == 2


# ==========================================================================
# fetch_hcup_ahal
# ==========================================================================

def test_fetch_ahal_hospital_level_downloads_whole_table(catalog, fake_org):
    fetch_hcup_ahal(catalog, "New York", ["2015"])
    sql = _queries(fake_org)[0]
    assert sql.startswith("SELECT * FROM")
    assert "WHERE" not in sql


def test_fetch_ahal_patient_level_requires_keys(catalog):
    with pytest.raises(ValueError, match="patient-level"):
        fetch_hcup_ahal(catalog, "Iowa", ["2015"])


def test_fetch_ahal_patient_level_filters_by_key(catalog, fake_org):
    fetch_hcup_ahal(catalog, "Iowa", ["2015"], core_keys=["k1", "k2"])
    sql = _queries(fake_org, "iowa_hcup:5678")[0]
    assert "WHERE CAST(KEY AS STRING) IN ('k1', 'k2')" in sql


def test_fetch_ahal_chunks_keys_to_stay_under_the_query_limit(catalog, fake_org):
    """BigQuery rejects statements past a character limit, so 25k keys have to
    go out as three 10k-max chunks."""
    keys = [f"k{i}" for i in range(25_000)]
    fetch_hcup_ahal(catalog, "Iowa", ["2015"], core_keys=keys)
    assert len(_queries(fake_org, "iowa_hcup:5678")) == 3


def test_fetch_ahal_normalises_quarter_suffixed_years(catalog, fake_org):
    """'2015q1q3' and '2015q4' both live in the single 2015 AHAL table."""
    fetch_hcup_ahal(catalog, "Iowa", ["2015q1q3", "2015q4"], core_keys=["k1"])
    assert len(_queries(fake_org, "iowa_hcup:5678")) == 1


def test_fetch_ahal_no_matching_tables_raises(catalog):
    with pytest.raises(ValueError, match="No AHAL tables matched"):
        fetch_hcup_ahal(catalog, "New York", ["1999"])


def test_fetch_ahal_raises_when_every_chunk_fails(catalog, fake_org):
    ia = fake_org.dataset("iowa_hcup:5678")
    ia.query_results = {"ia_sedd_2015_ahal": RuntimeError("denied")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="No AHAL data could be retrieved"):
            fetch_hcup_ahal(catalog, "Iowa", ["2015"], core_keys=["k1"])
