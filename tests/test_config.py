"""Tests for `src.utils.config` and the YAML files it reads.

The most valuable test in here is `test_every_get_data_path_key_exists`: it
scans the source for literal `get_data_path("...")` calls and checks each key
resolves in data.yaml. That catches the failure mode where someone adds a new
data file to the pipeline and forgets to register its path, which otherwise
only surfaces halfway through a long ETL run.
"""
import re
from pathlib import Path

import pytest
import yaml

from src.utils import config as cfg
from src.utils.config import PROJECT_ROOT, get_data_path, load_config


@pytest.fixture(autouse=True)
def clear_config_cache():
    """`load_config` is lru_cached at module scope; keep tests independent."""
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# --------------------------------------------------------------------------
# PROJECT_ROOT
# --------------------------------------------------------------------------

def test_project_root_is_repo_root():
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "config" / "data.yaml").is_file()


# --------------------------------------------------------------------------
# load_config
# --------------------------------------------------------------------------

def test_load_config_reads_data_yaml():
    conf = load_config("data.yaml")
    assert isinstance(conf, dict)
    assert "paths" in conf


def test_load_config_is_cached():
    first = load_config("data.yaml")
    second = load_config("data.yaml")
    assert first is second, "load_config should return the memoized object"
    assert load_config.cache_info().hits >= 1


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Config file missing"):
        load_config("does_not_exist.yaml")


@pytest.mark.parametrize(
    "name",
    sorted(p.name for p in (Path(__file__).parent.parent / "config").glob("*.yaml")),
)
def test_every_config_file_is_valid_yaml(name):
    """Scaffolding configs are mostly empty, but they must still parse."""
    conf = load_config(name)
    assert conf is None or isinstance(conf, dict)


# --------------------------------------------------------------------------
# get_data_path
# --------------------------------------------------------------------------

def test_get_data_path_returns_absolute_path_under_project_root():
    path = get_data_path("fips_crosswalk", "raw")
    assert path.is_absolute()
    assert str(path).startswith(str(PROJECT_ROOT))


def test_get_data_path_defaults_to_raw():
    assert get_data_path("fips_crosswalk") == get_data_path("fips_crosswalk", "raw")


def test_get_data_path_unknown_key_raises_keyerror():
    with pytest.raises(KeyError, match="raw -> not_a_real_file"):
        get_data_path("not_a_real_file", "raw")


def test_get_data_path_unknown_folder_raises_keyerror():
    with pytest.raises(KeyError, match="not_a_folder"):
        get_data_path("fips_crosswalk", "not_a_folder")


def test_get_data_path_does_not_require_the_file_to_exist():
    """Documented behaviour: it resolves a path, it does not validate one.

    Callers that need an existence guarantee have to check themselves.
    """
    conf = load_config("data.yaml")
    key = next(iter(conf["paths"]["raw"]))
    assert isinstance(get_data_path(key, "raw"), Path)


# --------------------------------------------------------------------------
# code <-> config contract
# --------------------------------------------------------------------------

GET_DATA_PATH_CALL = re.compile(
    r"""get_data_path\(\s*["']([^"']+)["']\s*(?:,\s*["']([^"']+)["'])?"""
)


def _literal_get_data_path_calls():
    """Every `get_data_path("key"[, "folder"])` written with literal strings."""
    found = set()
    for root in ("src", "scripts"):
        for py in (PROJECT_ROOT / root).rglob("*.py"):
            for key, folder in GET_DATA_PATH_CALL.findall(py.read_text()):
                found.add((key, folder or "raw", py.relative_to(PROJECT_ROOT)))
    return sorted(found)


def test_source_actually_calls_get_data_path():
    """Guards the guard: if the regex stops matching, the test below goes vacuous."""
    assert _literal_get_data_path_calls(), "expected literal get_data_path calls in src/"


@pytest.mark.parametrize(
    "key,folder,source", _literal_get_data_path_calls(),
    ids=lambda v: str(v),
)
def test_every_get_data_path_key_exists(key, folder, source):
    paths = load_config("data.yaml")["paths"]
    assert folder in paths, f"{source} asks for folder '{folder}', absent from data.yaml"
    assert key in paths[folder], f"{source} asks for '{folder}/{key}', absent from data.yaml"


def test_data_yaml_paths_are_relative():
    """Paths are joined onto PROJECT_ROOT, so an absolute entry would silently
    escape the repo and break on every other machine."""
    for folder, entries in load_config("data.yaml")["paths"].items():
        for key, value in entries.items():
            assert not Path(value).is_absolute(), f"{folder}/{key} is an absolute path"


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def test_credentials_are_loaded_as_module_constants():
    """conftest installs stub credentials before import; assert they landed.

    This also pins the fact that `config` reads credentials once at import
    time, so tests must never rely on mutating os.environ afterwards.
    """
    assert cfg.CENSUS_API_KEY
    assert cfg.USERNAME
    assert cfg.ORGNAME
