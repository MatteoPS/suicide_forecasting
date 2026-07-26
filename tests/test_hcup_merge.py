"""Tests for the HCUP helpers moved out of notebooks/0.4.

The notebook markdown says the AHAL merge "still messes up when trying to get
the hospital fips". These tests pin down what each branch actually does, so
that debugging can start from a known baseline rather than from a live
Redivis session.

Behaviour is unchanged from the notebook version, including the two warts
called out below.
"""
import numpy as np
import pandas as pd
import pytest

from src.etl.transform import (
    HCUP_MISSING_CODES,
    clean_hcup_missing_codes,
    smart_merge_ahal,
)


# ==========================================================================
# clean_hcup_missing_codes
# ==========================================================================

@pytest.mark.parametrize("code", HCUP_MISSING_CODES)
def test_every_sentinel_becomes_nan(code):
    df = pd.DataFrame({"AGE": [code, 42]})
    clean_hcup_missing_codes(df)
    assert pd.isna(df.loc[0, "AGE"])
    assert df.loc[1, "AGE"] == 42


def test_strips_float_suffix_from_identifiers():
    df = pd.DataFrame({"HOSPID": ["12345.0"]})
    clean_hcup_missing_codes(df)
    assert df.loc[0, "HOSPID"] == "12345"


def test_strips_leading_zeros_from_identifiers():
    """The same hospital appears as '0123' in one file and '123' in another."""
    df = pd.DataFrame({"HOSPID": ["0123"]})
    clean_hcup_missing_codes(df)
    assert df.loc[0, "HOSPID"] == "123"


def test_keeps_a_lone_zero():
    """The regex is `^0+(?!$)`, so an identifier of exactly '0' survives."""
    df = pd.DataFrame({"HOSPID": ["0"]})
    clean_hcup_missing_codes(df)
    assert df.loc[0, "HOSPID"] == "0"


def test_strips_whitespace_from_identifiers():
    df = pd.DataFrame({"KEY": ["  abc "]})
    clean_hcup_missing_codes(df)
    assert df.loc[0, "KEY"] == "abc"


def test_leaves_non_identifier_columns_alone():
    df = pd.DataFrame({"ZIP": ["01001"], "HOSPID": ["01001"]})
    clean_hcup_missing_codes(df)
    assert df.loc[0, "ZIP"] == "01001", "ZIP must keep its leading zero"
    assert df.loc[0, "HOSPID"] == "1001"


def test_tolerates_absent_identifier_columns():
    df = pd.DataFrame({"ZIP": ["10001"]})
    clean_hcup_missing_codes(df)  # must not raise
    assert list(df.columns) == ["ZIP"]


def test_operates_in_place_and_returns_the_frame():
    df = pd.DataFrame({"HOSPID": ["0123"]})
    returned = clean_hcup_missing_codes(df)
    assert returned is df


def test_missing_identifiers_become_the_string_nan():
    """Wart carried over from the notebook.

    The final `astype(str)` turns NaN into the literal 'nan', so two rows with
    a missing HOSPID compare equal and will merge to each other. Filter on NaN
    before merging if that matters.
    """
    df = pd.DataFrame({"HOSPID": [np.nan, "123"]})
    clean_hcup_missing_codes(df)
    assert df.loc[0, "HOSPID"] == "nan"


def test_custom_identifier_columns():
    df = pd.DataFrame({"MYID": ["0123"], "HOSPID": ["0456"]})
    clean_hcup_missing_codes(df, id_cols=["MYID"])
    assert df.loc[0, "MYID"] == "123"
    assert df.loc[0, "HOSPID"] == "0456", "not in id_cols, so left alone"


# ==========================================================================
# smart_merge_ahal - patient-level crosswalk
# ==========================================================================

def test_patient_level_merges_on_key():
    """The Iowa shape: AHAL is one row per discharge, keyed by KEY."""
    core = pd.DataFrame({"KEY": ["a", "b"], "ZIP": ["10001", "10002"]})
    ahal = pd.DataFrame({"KEY": ["a", "b"], "HFIPSSTCO": ["19153", "19113"]})
    out = smart_merge_ahal(core, ahal)
    assert list(out["HFIPSSTCO"]) == ["19153", "19113"]


def test_patient_level_drops_overlapping_columns_from_ahal():
    """Without the drop, pandas appends _x/_y suffixes and downstream code that
    reads plain 'ZIP' silently gets nothing."""
    core = pd.DataFrame({"KEY": ["a"], "ZIP": ["10001"]})
    ahal = pd.DataFrame({"KEY": ["a"], "ZIP": ["99999"], "HFIPSSTCO": ["19153"]})
    out = smart_merge_ahal(core, ahal)
    assert out.loc[0, "ZIP"] == "10001"
    assert "ZIP_x" not in out.columns


def test_patient_level_unmatched_key_yields_nan():
    core = pd.DataFrame({"KEY": ["a", "z"], "ZIP": ["10001", "10002"]})
    ahal = pd.DataFrame({"KEY": ["a"], "HFIPSSTCO": ["19153"]})
    out = smart_merge_ahal(core, ahal)
    assert pd.isna(out.loc[1, "HFIPSSTCO"])


def test_patient_level_preserves_row_count():
    core = pd.DataFrame({"KEY": ["a", "b", "c"], "ZIP": ["1", "2", "3"]})
    ahal = pd.DataFrame({"KEY": ["a"], "HFIPSSTCO": ["19153"]})
    assert len(smart_merge_ahal(core, ahal)) == 3


# ==========================================================================
# smart_merge_ahal - hospital-level crosswalk
# ==========================================================================

def test_hospital_level_merges_on_hospid_and_year():
    core = pd.DataFrame({"KEY": ["a"], "HOSPID": ["100"], "AYEAR": ["2015"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert out.loc[0, "HFIPSSTCO"] == "36061"


def test_hospital_level_renames_ayear_to_year():
    core = pd.DataFrame({"HOSPID": ["100"], "AYEAR": ["2015"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert "YEAR" in out.columns
    assert "AYEAR" not in out.columns


def test_hospital_level_does_not_match_across_years():
    """A hospital can change its FIPS between years, so the year is part of the key."""
    core = pd.DataFrame({"HOSPID": ["100"], "AYEAR": ["2016"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert pd.isna(out.loc[0, "HFIPSSTCO"])


def test_falls_back_to_dshospid_when_hospid_is_empty():
    """Which hospital identifier is populated varies by state and year."""
    core = pd.DataFrame({"HOSPID": [np.nan], "DSHOSPID": ["ds1"], "AYEAR": ["2015"]})
    ahal = pd.DataFrame({"DSHOSPID": ["ds1"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert out.loc[0, "HFIPSSTCO"] == "36061"


def test_prefers_hospid_when_both_are_populated():
    core = pd.DataFrame({"HOSPID": ["100"], "DSHOSPID": ["ds1"], "AYEAR": ["2015"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "DSHOSPID": ["other"],
                         "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert out.loc[0, "HFIPSSTCO"] == "36061"
    assert out.loc[0, "DSHOSPID"] == "ds1", "the AHAL copy is dropped, not merged"


def test_no_usable_hospital_id_degrades_to_empty_fips():
    """Better an empty column than a raise: enrich_fips_data handles NaN."""
    core = pd.DataFrame({"ZIP": ["10001"], "AYEAR": ["2015"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert "HFIPSSTCO" in out.columns
    assert out["HFIPSSTCO"].isna().all()


def test_all_null_hospid_counts_as_unusable():
    core = pd.DataFrame({"HOSPID": [np.nan, np.nan], "AYEAR": ["2015", "2015"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert out["HFIPSSTCO"].isna().all()


# ==========================================================================
# smart_merge_ahal - degenerate inputs
# ==========================================================================

def test_empty_core_is_returned_unchanged():
    core = pd.DataFrame({"KEY": []})
    out = smart_merge_ahal(core, pd.DataFrame({"KEY": ["a"], "HFIPSSTCO": ["1"]}))
    assert out.empty


def test_empty_ahal_returns_core_unchanged():
    """A state whose AHAL fetch came back empty must not lose its discharges."""
    core = pd.DataFrame({"KEY": ["a"], "ZIP": ["10001"]})
    out = smart_merge_ahal(core, pd.DataFrame())
    pd.testing.assert_frame_equal(out, core)


def test_type_mismatch_between_id_columns_is_reconciled():
    """CORE often types HOSPID numerically and AHAL as text, or vice versa."""
    core = pd.DataFrame({"HOSPID": [100], "AYEAR": ["2015"]})
    ahal = pd.DataFrame({"HOSPID": ["100"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})
    out = smart_merge_ahal(core, ahal)
    assert out.loc[0, "HFIPSSTCO"] == "36061"


def test_end_to_end_clean_then_merge():
    """The order the notebook uses: normalise identifiers, then merge on them.

    Without the cleaning step '0100' and '100' are different hospitals.
    """
    core = pd.DataFrame({"HOSPID": ["0100"], "AYEAR": ["2015"], "ZIP": ["10001"]})
    ahal = pd.DataFrame({"HOSPID": ["100.0"], "YEAR": ["2015"], "HFIPSSTCO": ["36061"]})

    unmerged = smart_merge_ahal(core.copy(), ahal.copy())
    assert pd.isna(unmerged.loc[0, "HFIPSSTCO"])

    clean_hcup_missing_codes(core)
    clean_hcup_missing_codes(ahal)
    merged = smart_merge_ahal(core, ahal)
    assert merged.loc[0, "HFIPSSTCO"] == "36061"
