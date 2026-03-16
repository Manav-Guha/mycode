"""Tests for batch_mine repo deduplication."""

import json
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

# batch_mine is a script, not a package — add scripts/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import batch_mine


# ── _load_processed_repos ──


class TestLoadProcessedRepos:
    def test_missing_file(self, tmp_path):
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", tmp_path / "nope.txt"):
            assert batch_mine._load_processed_repos() == set()

    def test_reads_urls(self, tmp_path):
        f = tmp_path / "processed.txt"
        f.write_text("https://github.com/a/b\nhttps://github.com/c/d\n")
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            result = batch_mine._load_processed_repos()
        assert result == {"https://github.com/a/b", "https://github.com/c/d"}

    def test_strips_whitespace_and_trailing_slash(self, tmp_path):
        f = tmp_path / "processed.txt"
        f.write_text("  https://github.com/a/b/  \n\n  \nhttps://github.com/c/d\n")
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            result = batch_mine._load_processed_repos()
        assert result == {"https://github.com/a/b", "https://github.com/c/d"}

    def test_handles_duplicates(self, tmp_path):
        f = tmp_path / "processed.txt"
        f.write_text("https://github.com/a/b\nhttps://github.com/a/b\n")
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            result = batch_mine._load_processed_repos()
        assert result == {"https://github.com/a/b"}
        assert len(result) == 1

    def test_oserror_returns_empty(self, tmp_path):
        f = tmp_path / "processed.txt"
        f.write_text("data")
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                # read_text uses open internally — mock at Path level
                with mock.patch.object(Path, "read_text", side_effect=OSError("denied")):
                    result = batch_mine._load_processed_repos()
        assert result == set()


# ── _record_processed_repo ──


class TestRecordProcessedRepo:
    def test_creates_file_and_appends(self, tmp_path):
        f = tmp_path / "sub" / "processed.txt"
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            batch_mine._record_processed_repo("https://github.com/a/b")
            batch_mine._record_processed_repo("https://github.com/c/d/")
        lines = f.read_text().splitlines()
        assert lines == ["https://github.com/a/b", "https://github.com/c/d"]

    def test_normalises_trailing_slash(self, tmp_path):
        f = tmp_path / "processed.txt"
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            batch_mine._record_processed_repo("https://github.com/a/b/")
        assert f.read_text().strip() == "https://github.com/a/b"

    def test_oserror_is_warning(self, tmp_path, caplog):
        f = tmp_path / "processed.txt"
        with mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", f):
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                batch_mine._record_processed_repo("https://github.com/a/b")
        # Should not raise — just log a warning
        assert "Could not write" in caplog.text


# ── Dedup integration in mine() ──


def _make_repos_json(tmp_path, urls):
    """Write a minimal discovered_repos.json."""
    repos = [{"repo_url": u, "clone_url": u + ".git"} for u in urls]
    p = tmp_path / "repos.json"
    p.write_text(json.dumps(repos))
    return p


class TestMineDedup:
    """Test that mine() skips already-processed repos."""

    def _run_mine(self, tmp_path, urls, processed, force=False, max_repos=0):
        """Run mine() with mocked clone/mycode, return (summary, clone_calls)."""
        input_path = _make_repos_json(tmp_path, urls)
        results_dir = tmp_path / "results"
        processed_file = tmp_path / "processed.txt"
        if processed:
            processed_file.write_text("\n".join(processed) + "\n")

        clone_calls = []

        def fake_clone(url, dest):
            clone_calls.append(url)
            dest.mkdir(parents=True, exist_ok=True)
            return True

        def fake_run_mycode(path, timeout=300):
            # Write a minimal report so it counts as success
            report = {"findings": [], "unrecognized_dependencies": []}
            (path / "mycode-report.json").write_text(json.dumps(report))
            return 0, "{}", ""

        with (
            mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", processed_file),
            mock.patch.object(batch_mine, "_clone_repo", side_effect=fake_clone),
            mock.patch.object(batch_mine, "_run_mycode", side_effect=fake_run_mycode),
            mock.patch.object(batch_mine, "_snapshot_discoveries", return_value=set()),
            mock.patch.object(batch_mine, "_collect_new_discoveries", return_value=[]),
        ):
            summary = batch_mine.mine(
                input_path=input_path,
                results_dir=results_dir,
                max_repos=max_repos,
                force=force,
            )
        return summary, clone_calls

    def test_skips_processed_repos(self, tmp_path):
        urls = ["https://github.com/a/b", "https://github.com/c/d", "https://github.com/e/f"]
        processed = ["https://github.com/a/b"]
        summary, clone_calls = self._run_mine(tmp_path, urls, processed)

        # Only c/d and e/f should be cloned
        assert len(clone_calls) == 2
        assert "https://github.com/a/b.git" not in clone_calls
        assert summary["total_repos"] == 2
        assert summary["repos_tested"] == 2

    def test_force_ignores_processed(self, tmp_path):
        urls = ["https://github.com/a/b", "https://github.com/c/d"]
        processed = ["https://github.com/a/b", "https://github.com/c/d"]
        summary, clone_calls = self._run_mine(tmp_path, urls, processed, force=True)

        assert len(clone_calls) == 2
        assert summary["total_repos"] == 2

    def test_no_processed_file(self, tmp_path):
        urls = ["https://github.com/a/b"]
        summary, clone_calls = self._run_mine(tmp_path, urls, processed=[])

        assert len(clone_calls) == 1
        assert summary["total_repos"] == 1

    def test_max_repos_applies_after_dedup(self, tmp_path):
        urls = [f"https://github.com/u/r{i}" for i in range(10)]
        processed = [f"https://github.com/u/r{i}" for i in range(5)]
        summary, clone_calls = self._run_mine(
            tmp_path, urls, processed, max_repos=3,
        )

        # 5 remaining after dedup, limited to 3
        assert len(clone_calls) == 3
        assert summary["total_repos"] == 3

    def test_records_successful_repos(self, tmp_path):
        urls = ["https://github.com/a/b", "https://github.com/c/d"]
        input_path = _make_repos_json(tmp_path, urls)
        results_dir = tmp_path / "results"
        processed_file = tmp_path / "processed.txt"

        def fake_clone(url, dest):
            dest.mkdir(parents=True, exist_ok=True)
            return True

        def fake_run_mycode(path, timeout=300):
            report = {"findings": [], "unrecognized_dependencies": []}
            (path / "mycode-report.json").write_text(json.dumps(report))
            return 0, "{}", ""

        with (
            mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", processed_file),
            mock.patch.object(batch_mine, "_clone_repo", side_effect=fake_clone),
            mock.patch.object(batch_mine, "_run_mycode", side_effect=fake_run_mycode),
            mock.patch.object(batch_mine, "_snapshot_discoveries", return_value=set()),
            mock.patch.object(batch_mine, "_collect_new_discoveries", return_value=[]),
        ):
            batch_mine.mine(input_path=input_path, results_dir=results_dir, max_repos=0)

            # Both should be recorded (read while mock is still active)
            recorded = batch_mine._load_processed_repos()
            assert "https://github.com/a/b" in recorded
            assert "https://github.com/c/d" in recorded

    def test_clone_failure_not_recorded(self, tmp_path):
        urls = ["https://github.com/a/b"]
        input_path = _make_repos_json(tmp_path, urls)
        results_dir = tmp_path / "results"
        processed_file = tmp_path / "processed.txt"

        with (
            mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", processed_file),
            mock.patch.object(batch_mine, "_clone_repo", return_value=False),
            mock.patch.object(batch_mine, "_snapshot_discoveries", return_value=set()),
            mock.patch.object(batch_mine, "_collect_new_discoveries", return_value=[]),
        ):
            batch_mine.mine(input_path=input_path, results_dir=results_dir, max_repos=0)

        # Should NOT be recorded
        assert not processed_file.exists() or processed_file.read_text().strip() == ""

    def test_mycode_error_not_recorded(self, tmp_path):
        urls = ["https://github.com/a/b"]
        input_path = _make_repos_json(tmp_path, urls)
        results_dir = tmp_path / "results"
        processed_file = tmp_path / "processed.txt"

        def fake_clone(url, dest):
            dest.mkdir(parents=True, exist_ok=True)
            return True

        with (
            mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", processed_file),
            mock.patch.object(batch_mine, "_clone_repo", side_effect=fake_clone),
            mock.patch.object(batch_mine, "_run_mycode", return_value=(-1, "", "timed out")),
            mock.patch.object(batch_mine, "_snapshot_discoveries", return_value=set()),
            mock.patch.object(batch_mine, "_collect_new_discoveries", return_value=[]),
        ):
            batch_mine.mine(input_path=input_path, results_dir=results_dir, max_repos=0)

        assert not processed_file.exists() or processed_file.read_text().strip() == ""

    def test_trailing_slash_normalisation(self, tmp_path):
        """Repo URL with trailing slash matches processed entry without it."""
        urls = ["https://github.com/a/b/"]
        processed = ["https://github.com/a/b"]
        summary, clone_calls = self._run_mine(tmp_path, urls, processed)

        assert len(clone_calls) == 0
        assert summary["total_repos"] == 0


# ── CLI --force flag ──


class TestCLIForceFlag:
    def test_force_flag_parsed(self):
        with mock.patch.object(batch_mine, "mine", return_value={
            "repos_tested": 0, "repos_failed": 0, "total_repos": 0,
        }) as mock_mine:
            with mock.patch("pathlib.Path.is_file", return_value=True):
                batch_mine.main(["--input", "x.json", "--force"])
        _, kwargs = mock_mine.call_args
        assert kwargs.get("force") is True

    def test_no_force_by_default(self):
        with mock.patch.object(batch_mine, "mine", return_value={
            "repos_tested": 0, "repos_failed": 0, "total_repos": 0,
        }) as mock_mine:
            with mock.patch("pathlib.Path.is_file", return_value=True):
                batch_mine.main(["--input", "x.json"])
        _, kwargs = mock_mine.call_args
        assert kwargs.get("force") is False
