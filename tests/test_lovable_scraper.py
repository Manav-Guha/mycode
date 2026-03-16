"""Tests for lovable_scraper — GPT-Engineer-App org repo discovery."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from lovable_scraper import (
    _list_org_repos,
    _passes_basic_filters,
    _has_package_json,
    scrape,
    main,
    _ORG,
    _ORG_REPOS_URL,
    _PER_PAGE,
)

# Patch time.sleep globally for all tests in this module — _api_get sleeps
# on every call and scrape() passes real delays (2.5s authenticated).
pytestmark = pytest.mark.usefixtures("_no_sleep")


@pytest.fixture(autouse=True)
def _no_sleep():
    with mock.patch("time.sleep"):
        yield


# ── Fixtures ──


def _repo(
    name="my-app",
    archived=False,
    size=1000,
    fork=False,
    language="TypeScript",
    stars=0,
    owner="GPT-Engineer-App",
    repo_id=1,
):
    """Build a minimal GitHub repo dict."""
    return {
        "id": repo_id,
        "name": name,
        "full_name": f"{owner}/{name}",
        "html_url": f"https://github.com/{owner}/{name}",
        "clone_url": f"https://github.com/{owner}/{name}.git",
        "owner": {"login": owner},
        "archived": archived,
        "size": size,
        "fork": fork,
        "language": language,
        "stargazers_count": stars,
        "pushed_at": "2026-03-01T00:00:00Z",
        "description": "A Lovable-generated app",
    }


# ── _passes_basic_filters ──


class TestBasicFilters:
    def test_normal_repo_passes(self):
        assert _passes_basic_filters(_repo()) is True

    def test_archived_repo_rejected(self):
        assert _passes_basic_filters(_repo(archived=True)) is False

    def test_empty_repo_rejected(self):
        assert _passes_basic_filters(_repo(size=0)) is False

    def test_fork_rejected(self):
        assert _passes_basic_filters(_repo(fork=True)) is False

    def test_no_language_still_passes(self):
        # Lovable repos might not have language detected yet
        assert _passes_basic_filters(_repo(language=None)) is True


# ── _list_org_repos ──


class TestListOrgRepos:
    def test_single_page(self):
        repos = [_repo(name=f"app-{i}", repo_id=i) for i in range(30)]
        with mock.patch("lovable_scraper._api_get", return_value=repos) as mock_get:
            result = _list_org_repos(token="tok", delay=0)
        assert len(result) == 30
        # Fewer than _PER_PAGE → single page, one call
        mock_get.assert_called_once()
        url = mock_get.call_args[0][0]
        assert "GPT-Engineer-App" in url
        assert "type=public" in url

    def test_multi_page_pagination(self):
        page1 = [_repo(name=f"app-{i}", repo_id=i) for i in range(_PER_PAGE)]
        page2 = [_repo(name=f"app-{i}", repo_id=i) for i in range(_PER_PAGE, _PER_PAGE + 20)]

        call_count = 0

        def fake_get(url, token, delay):
            nonlocal call_count
            call_count += 1
            if "page=1" in url:
                return page1
            elif "page=2" in url:
                return page2
            return []

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get):
            result = _list_org_repos(token="tok", delay=0)
        assert len(result) == _PER_PAGE + 20
        assert call_count == 2

    def test_stops_on_none_response(self):
        with mock.patch("lovable_scraper._api_get", return_value=None):
            result = _list_org_repos(token="tok", delay=0)
        assert result == []

    def test_stops_on_empty_page(self):
        with mock.patch("lovable_scraper._api_get", return_value=[]):
            result = _list_org_repos(token="tok", delay=0)
        assert result == []

    def test_handles_non_list_response(self):
        # GitHub could return an error dict
        with mock.patch("lovable_scraper._api_get", return_value={"message": "error"}):
            result = _list_org_repos(token="tok", delay=0)
        assert result == []


# ── _has_package_json ──


class TestHasPackageJson:
    def test_returns_true_when_found(self):
        with mock.patch("lovable_scraper._api_get", return_value={"name": "package.json"}):
            assert _has_package_json("owner", "repo", "tok", 0) is True

    def test_returns_false_when_404(self):
        with mock.patch("lovable_scraper._api_get", return_value=None):
            assert _has_package_json("owner", "repo", "tok", 0) is False


# ── scrape (integration with mocks) ──


class TestScrape:
    def test_full_pipeline_with_deep_filter(self, tmp_path):
        repos = [
            _repo(name="good-app", repo_id=1, size=5000),
            _repo(name="archived-app", repo_id=2, archived=True),
            _repo(name="empty-app", repo_id=3, size=0),
        ]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            # Org listing
            if "/orgs/" in url and "repos" in url:
                return repos
            # package.json check
            if "/contents/package.json" in url:
                if "good-app" in url:
                    return {"name": "package.json", "content": ""}
                return None
            # Languages endpoint
            if "/languages" in url:
                return {"TypeScript": 20000, "CSS": 5000}
            # Dep file: package.json content
            if "/contents/requirements.txt" in url:
                return None
            if "/contents/package.json" in url:
                return None
            if "/contents/" in url:
                return None
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get), \
             mock.patch("lovable_scraper._get_dep_count", return_value=8):
            result = scrape(
                token="tok",
                output=output,
                max_repos=100,
                min_loc=100,
                min_deps=3,
            )

        assert len(result) == 1
        assert result[0]["repo_url"] == "https://github.com/GPT-Engineer-App/good-app"
        assert result[0]["clone_url"] == "https://github.com/GPT-Engineer-App/good-app.git"
        assert result[0]["loc_estimate"] > 0
        assert result[0]["dep_count"] == 8

        # Verify output file
        written = json.loads(output.read_text())
        assert len(written) == 1

    def test_skip_deep_filter(self, tmp_path):
        repos = [_repo(name=f"app-{i}", repo_id=i) for i in range(5)]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            if "/contents/package.json" in url:
                return {"name": "package.json"}
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get):
            result = scrape(
                token="tok",
                output=output,
                max_repos=100,
                skip_deep_filter=True,
            )

        assert len(result) == 5
        # loc_estimate and dep_count should be 0 when skipping deep filter
        for r in result:
            assert r["loc_estimate"] == 0
            assert r["dep_count"] == 0

    def test_max_repos_cap(self, tmp_path):
        repos = [_repo(name=f"app-{i}", repo_id=i) for i in range(20)]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            if "/contents/package.json" in url:
                return {"name": "package.json"}
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get):
            result = scrape(
                token="tok",
                output=output,
                max_repos=3,
                skip_deep_filter=True,
            )

        assert len(result) == 3

    def test_no_package_json_filtered_out(self, tmp_path):
        repos = [_repo(name="no-pkg", repo_id=1)]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            return None  # package.json not found

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get):
            result = scrape(
                token="tok",
                output=output,
                skip_deep_filter=True,
            )

        assert len(result) == 0

    def test_loc_too_low_filtered(self, tmp_path):
        repos = [_repo(name="tiny", repo_id=1)]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            if "/contents/package.json" in url:
                return {"name": "package.json"}
            if "/languages" in url:
                return {"TypeScript": 200}  # ~5 LOC
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get), \
             mock.patch("lovable_scraper._get_dep_count", return_value=10):
            result = scrape(
                token="tok",
                output=output,
                min_loc=100,
            )

        assert len(result) == 0

    def test_deps_too_low_filtered(self, tmp_path):
        repos = [_repo(name="few-deps", repo_id=1)]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            if "/contents/package.json" in url:
                return {"name": "package.json"}
            if "/languages" in url:
                return {"TypeScript": 40000}  # ~1000 LOC
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get), \
             mock.patch("lovable_scraper._get_dep_count", return_value=1):
            result = scrape(
                token="tok",
                output=output,
                min_deps=3,
            )

        assert len(result) == 0


# ── Output format ──


class TestOutputFormat:
    """Verify output is compatible with batch_mine.py."""

    def test_required_fields(self, tmp_path):
        repos = [_repo(name="compat", repo_id=1)]
        output = tmp_path / "out.json"

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            if "/contents/package.json" in url:
                return {"name": "package.json"}
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get):
            result = scrape(token="tok", output=output, skip_deep_filter=True)

        entry = result[0]
        required_fields = {
            "repo_url", "clone_url", "stars", "last_commit_date",
            "language", "description", "loc_estimate", "dep_count",
        }
        assert required_fields.issubset(set(entry.keys()))
        assert isinstance(entry["repo_url"], str)
        assert isinstance(entry["clone_url"], str)
        assert entry["clone_url"].endswith(".git")
        assert isinstance(entry["stars"], int)


# ── CLI ──


class TestCLI:
    def test_main_writes_output(self, tmp_path):
        output = tmp_path / "cli_out.json"
        repos = [_repo(name="cli-app", repo_id=1)]

        def fake_get(url, token, delay):
            if "/orgs/" in url:
                return repos
            if "/contents/package.json" in url:
                return {"name": "package.json"}
            return None

        with mock.patch("lovable_scraper._api_get", side_effect=fake_get):
            main([
                "--token", "fake",
                "--output", str(output),
                "--max-repos", "10",
                "--skip-deep-filter",
            ])

        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data) == 1

    def test_main_default_args(self, tmp_path, monkeypatch):
        """Ensure main() runs with minimal args."""
        output = tmp_path / "default_out.json"

        with mock.patch("lovable_scraper._api_get", return_value=[]):
            main(["--output", str(output), "--skip-deep-filter"])

        assert output.exists()
        assert json.loads(output.read_text()) == []
