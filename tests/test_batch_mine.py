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


# ── _parse_phase_timing ──


class TestParsePhraseTiming:
    def test_full_pipeline_phases(self):
        stderr = textwrap.dedent("""\
            10:00:00 INFO Language detected: python
            10:00:02 INFO Session ready: /tmp/mycode_abc
            10:00:45 INFO Ingestion complete: 12 files, 5 dependencies
            10:00:46 INFO Library matching: 3 recognized, 2 unrecognized
            10:01:00 INFO Generated 8 scenarios (model: offline)
            10:01:01 INFO Scenario review: 8 approved, 0 skipped
            10:03:30 INFO Execution complete: 8 completed, 2 failed, 0 skipped
            10:03:31 INFO Report generated (model: offline)
        """)
        result = batch_mine._parse_phase_timing(stderr, 211.0, timed_out=False)
        assert "PHASE detect:" in result
        assert "PHASE deps:" in result
        assert "PHASE ingest:" in result
        assert "PHASE execute:" in result
        assert "PHASE report:" in result
        assert "TIMEOUT" not in result

    def test_timeout_shows_stuck_phase(self):
        stderr = textwrap.dedent("""\
            10:00:00 INFO Language detected: python
            10:00:02 INFO Session ready: /tmp/mycode_abc
            10:00:45 INFO Ingestion complete: 12 files, 5 dependencies
            10:00:46 INFO Library matching: 3 recognized, 2 unrecognized
        """)
        result = batch_mine._parse_phase_timing(stderr, 310.0, timed_out=True)
        assert "PHASE detect:" in result
        assert "PHASE deps:" in result
        assert "PHASE ingest:" in result
        # Should show scenario_gen as the stuck phase (next after library)
        assert "PHASE scenario_gen: TIMEOUT" in result

    def test_timeout_during_deps(self):
        stderr = textwrap.dedent("""\
            10:00:00 INFO Language detected: python
        """)
        result = batch_mine._parse_phase_timing(stderr, 300.0, timed_out=True)
        assert "PHASE detect:" in result
        assert "PHASE deps: TIMEOUT" in result

    def test_no_stderr(self):
        result = batch_mine._parse_phase_timing("", 300.0, timed_out=True)
        assert "no stderr" in result

    def test_no_phase_markers(self):
        stderr = "10:00:00 DEBUG Some random log line\n"
        result = batch_mine._parse_phase_timing(stderr, 300.0, timed_out=True)
        assert "no phase markers" in result

    def test_timeout_during_execute(self):
        stderr = textwrap.dedent("""\
            10:00:00 INFO Language detected: python
            10:00:02 INFO Session ready: /tmp/abc
            10:00:10 INFO Ingestion complete: 5 files
            10:00:11 INFO Library matching: 2 recognized
            10:00:15 INFO Generated 4 scenarios (model: offline)
            10:00:16 INFO Scenario review: 4 approved, 0 skipped
        """)
        result = batch_mine._parse_phase_timing(stderr, 310.0, timed_out=True)
        assert "PHASE execute: TIMEOUT" in result


# ── _classify_failure ──


class TestClassifyFailure:
    def test_timeout_from_stderr(self):
        status, _ = batch_mine._classify_failure(
            returncode=-1, stderr="mycode timed out",
            report={}, elapsed=305.0, timeout=300,
        )
        assert status == "timeout"

    def test_timeout_from_elapsed(self):
        status, _ = batch_mine._classify_failure(
            returncode=-1, stderr="some error",
            report={}, elapsed=298.0, timeout=300,
        )
        assert status == "timeout"

    def test_skip_baseline_failed(self):
        report = {
            "summary": "No testable code found",
            "statistics": {"scenarios_run": 0},
        }
        status, error = batch_mine._classify_failure(
            returncode=1, stderr="",
            report=report, elapsed=10.0, timeout=300,
        )
        assert status == "skip"
        assert "No testable code" in error

    def test_crash_on_unknown_error(self):
        status, _ = batch_mine._classify_failure(
            returncode=-1, stderr="Traceback: something broke",
            report={}, elapsed=10.0, timeout=300,
        )
        assert status == "mycode_crash"

    def test_crash_with_no_report(self):
        status, _ = batch_mine._classify_failure(
            returncode=-1, stderr="segfault",
            report={}, elapsed=50.0, timeout=300,
        )
        assert status == "mycode_crash"


# ── mine() failure classification integration ──


class TestMineFailureClassification:
    """Test that mine() uses the new status labels."""

    def _run_mine_with_mycode_result(self, tmp_path, returncode, stdout, stderr,
                                     report=None):
        urls = ["https://github.com/a/b"]
        input_path = _make_repos_json(tmp_path, urls)
        results_dir = tmp_path / "results"
        processed_file = tmp_path / "processed.txt"

        def fake_clone(url, dest):
            dest.mkdir(parents=True, exist_ok=True)
            if report:
                (dest / "mycode-report.json").write_text(json.dumps(report))
            return True

        with (
            mock.patch.object(batch_mine, "_PROCESSED_REPOS_PATH", processed_file),
            mock.patch.object(batch_mine, "_clone_repo", side_effect=fake_clone),
            mock.patch.object(
                batch_mine, "_run_mycode",
                return_value=(returncode, stdout, stderr),
            ),
            mock.patch.object(batch_mine, "_snapshot_discoveries", return_value=set()),
            mock.patch.object(batch_mine, "_collect_new_discoveries", return_value=[]),
        ):
            return batch_mine.mine(
                input_path=input_path, results_dir=results_dir, max_repos=0,
            )

    def test_timeout_status(self, tmp_path):
        summary = self._run_mine_with_mycode_result(
            tmp_path, returncode=-1, stdout="", stderr="mycode timed out",
        )
        repo = summary["repos"][0]
        assert repo["status"] == "timeout"
        assert summary["timeouts"] == 1

    def test_crash_status(self, tmp_path):
        summary = self._run_mine_with_mycode_result(
            tmp_path, returncode=-1, stdout="",
            stderr="Traceback: ValueError: boom",
        )
        repo = summary["repos"][0]
        assert repo["status"] == "mycode_crash"
        assert summary["crashes"] == 1

    def test_skip_status(self, tmp_path):
        report = {
            "summary": "No testable code found",
            "statistics": {"scenarios_run": 0},
            "findings": [],
            "unrecognized_dependencies": [],
        }
        summary = self._run_mine_with_mycode_result(
            tmp_path, returncode=1, stdout="", stderr="",
            report=report,
        )
        repo = summary["repos"][0]
        assert repo["status"] == "skip"
        assert summary["skips"] == 1
        # Skips count as "tested" (processed)
        assert summary["repos_tested"] == 1
