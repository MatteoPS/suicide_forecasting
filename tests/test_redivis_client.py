"""Tests for `src.utils.redivis_client`.

`RedivisCatalog` exists to stop the HCUP fetchers hammering the Redivis API
with the same metadata calls, so the caching is the behaviour under test.
The fake organization in conftest counts its own calls.
"""
import pandas as pd
import pytest
import redivis

from src.utils.redivis_client import (
    RedivisCatalog,
    display_hcup_variables,
    print_redivis_tree,
)


# ==========================================================================
# datasets
# ==========================================================================

def test_datasets_lists_every_dataset(catalog):
    assert set(catalog.datasets["Dataset_Name"]) == {"New York HCUP", "Iowa HCUP"}


def test_datasets_strips_the_organization_prefix_from_references(catalog):
    """`qualified_reference` is 'org.dataset'; only the tail is usable as a ref."""
    assert set(catalog.datasets["Reference"]) == {"new_york_hcup:1234", "iowa_hcup:5678"}


def test_datasets_parses_the_update_timestamp(catalog):
    assert pd.api.types.is_datetime64_any_dtype(catalog.datasets["Updated_At"])


def test_datasets_is_fetched_once(catalog, fake_org):
    catalog.datasets
    catalog.datasets
    catalog.datasets
    assert fake_org.list_datasets_calls == 1


# ==========================================================================
# tables
# ==========================================================================

def test_get_tables_returns_metadata(catalog):
    tables = catalog.get_tables("new_york_hcup:1234")
    assert "NY SEDD 2015q4 CORE" in set(tables["Table_Name"])
    for column in ("Table_Name", "Reference", "Rows", "Columns", "Size_Bytes",
                   "Updated_At", "Description"):
        assert column in tables.columns


def test_get_tables_is_cached_per_dataset(catalog, monkeypatch):
    catalog.get_tables("new_york_hcup:1234")
    # Break the underlying accessor: a cached call must not touch it again.
    monkeypatch.setattr(catalog.org, "dataset", _boom)
    assert not catalog.get_tables("new_york_hcup:1234").empty


def test_get_tables_caches_datasets_independently(catalog):
    ny = catalog.get_tables("new_york_hcup:1234")
    ia = catalog.get_tables("iowa_hcup:5678")
    assert len(ny) == 4
    assert len(ia) == 1


# ==========================================================================
# variables
# ==========================================================================

def test_get_variables_returns_the_schema(catalog):
    variables = catalog.get_variables("new_york_hcup:1234", "ny_sedd_2015q1q3_core:aaaa")
    assert "ECODE1" in set(variables["Variable"])
    assert list(variables.columns) == ["Variable", "Type", "Label", "Description"]


def test_get_variables_is_cached_per_table(catalog, monkeypatch):
    catalog.get_variables("new_york_hcup:1234", "ny_sedd_2015q1q3_core:aaaa")
    monkeypatch.setattr(catalog.org, "dataset", _boom)
    assert not catalog.get_variables(
        "new_york_hcup:1234", "ny_sedd_2015q1q3_core:aaaa"
    ).empty


# ==========================================================================
# cache invalidation
# ==========================================================================

def test_clear_cache_forces_a_refetch(catalog, fake_org):
    catalog.datasets
    catalog.get_tables("new_york_hcup:1234")
    catalog.clear_cache()
    catalog.datasets
    assert fake_org.list_datasets_calls == 2


def test_clear_cache_empties_the_table_and_schema_caches(catalog):
    catalog.get_tables("new_york_hcup:1234")
    catalog.get_variables("new_york_hcup:1234", "ny_sedd_2015q1q3_core:aaaa")
    catalog.clear_cache()
    assert catalog._tables_cache == {}
    assert catalog._schema_cache == {}


def test_org_name_is_retained(fake_org, monkeypatch):
    """`fetch_hcup` builds `org.dataset.table` references from this."""
    monkeypatch.setattr(redivis, "organization", lambda name: fake_org)
    assert RedivisCatalog("columbia").org_name == "columbia"


# ==========================================================================
# display_hcup_variables
# ==========================================================================

def test_display_hcup_variables_finds_the_table(catalog):
    variables = display_hcup_variables(catalog, "New York", "2015q1q3", "sedd")
    assert "ECODE1" in set(variables["Variable"])


def test_display_hcup_variables_unknown_state_raises(catalog):
    with pytest.raises(ValueError, match="No dataset found matching state"):
        display_hcup_variables(catalog, "Atlantis", 2015, "sedd")


def test_display_hcup_variables_unknown_table_raises(catalog):
    with pytest.raises(ValueError, match="No table found matching year"):
        display_hcup_variables(catalog, "New York", 1999, "sedd")


def test_display_hcup_variables_can_target_ahal(catalog):
    variables = display_hcup_variables(catalog, "New York", 2015, "sedd", table="ahal")
    assert "HFIPSSTCO" in set(variables["Variable"])


# ==========================================================================
# print_redivis_tree
# ==========================================================================

def test_print_redivis_tree_lists_datasets_and_tables(fake_org, monkeypatch, capsys):
    monkeypatch.setattr(redivis, "organization", lambda name: fake_org)
    print_redivis_tree("test-org", limit_tables=2)
    out = capsys.readouterr().out
    assert "new_york_hcup:1234" in out
    assert "ny_sedd_2014_core:eeee" in out


def test_print_redivis_tree_reports_the_truncated_count(fake_org, monkeypatch, capsys):
    monkeypatch.setattr(redivis, "organization", lambda name: fake_org)
    print_redivis_tree("test-org", limit_tables=1)
    assert "+3 more tables" in capsys.readouterr().out


def _boom(*args, **kwargs):
    raise AssertionError("cache miss: the API should not have been called again")
