"""Tests for the library taxonomy auto-classifiers."""

import pytest

from mycode.classifiers import (
    architectural_pattern_classifier,
    classify_finding,
    classify_project,
    failure_domain_classifier,
    failure_pattern_classifier,
    operational_trigger_classifier,
    vertical_classifier,
)


# ══════════════════════════════════════════════════════════════════
# 1. failure_domain_classifier
# ══════════════════════════════════════════════════════════════════


class TestFailureDomainClassifier:
    """Tests for failure_domain_classifier."""

    def test_memory_error_type_override(self):
        result = failure_domain_classifier(
            "some_scenario", "data_volume_scaling", "MemoryError", "",
        )
        assert result == "resource_exhaustion"

    def test_timeout_error_type(self):
        result = failure_domain_classifier(
            "some_scenario", "", "TimeoutError", "",
        )
        assert result == "scaling_collapse"

    def test_import_error_type(self):
        result = failure_domain_classifier(
            "some_scenario", "", "ImportError", "",
        )
        assert result == "configuration_environment_failure"

    def test_module_not_found_error(self):
        result = failure_domain_classifier(
            "some_scenario", "", "ModuleNotFoundError", "",
        )
        assert result == "configuration_environment_failure"

    def test_type_error(self):
        result = failure_domain_classifier(
            "some_scenario", "", "TypeError", "",
        )
        assert result == "input_handling_failure"

    def test_value_error(self):
        result = failure_domain_classifier(
            "some_scenario", "", "ValueError", "",
        )
        assert result == "input_handling_failure"

    def test_connection_error(self):
        result = failure_domain_classifier(
            "some_scenario", "", "ConnectionError", "",
        )
        assert result == "integration_failure"

    def test_error_in_details_fallback(self):
        result = failure_domain_classifier(
            "some_scenario", "", "", "Traceback: MemoryError at line 42",
        )
        assert result == "resource_exhaustion"

    def test_scenario_name_memory_keyword(self):
        result = failure_domain_classifier(
            "flask_memory_under_load", "", "", "",
        )
        assert result == "resource_exhaustion"

    def test_scenario_name_concurrent_keyword(self):
        result = failure_domain_classifier(
            "express_concurrent_request_load", "", "", "",
        )
        assert result == "concurrency_failure"

    def test_scenario_name_scaling_keyword(self):
        result = failure_domain_classifier(
            "data_scaling_test", "", "", "",
        )
        assert result == "scaling_collapse"

    def test_scenario_name_validation_keyword(self):
        result = failure_domain_classifier(
            "input_validation_test", "", "", "",
        )
        assert result == "input_handling_failure"

    def test_scenario_name_version_keyword(self):
        result = failure_domain_classifier(
            "version_check_test", "", "", "",
        )
        assert result == "dependency_failure"

    def test_scenario_name_timeout_keyword(self):
        result = failure_domain_classifier(
            "cascading_timeout_test", "", "", "",
        )
        assert result == "integration_failure"

    def test_scenario_name_install_keyword(self):
        result = failure_domain_classifier(
            "install_failure_test", "", "", "",
        )
        assert result == "configuration_environment_failure"

    def test_category_memory_profiling(self):
        result = failure_domain_classifier(
            "some_generic_test", "memory_profiling", "", "",
        )
        assert result == "resource_exhaustion"

    def test_category_concurrent_execution(self):
        result = failure_domain_classifier(
            "some_generic_test", "concurrent_execution", "", "",
        )
        assert result == "concurrency_failure"

    def test_category_data_volume_scaling(self):
        result = failure_domain_classifier(
            "some_generic_test", "data_volume_scaling", "", "",
        )
        assert result == "scaling_collapse"

    def test_category_edge_case_input(self):
        result = failure_domain_classifier(
            "some_generic_test", "edge_case_input", "", "",
        )
        assert result == "input_handling_failure"

    def test_category_blocking_io(self):
        result = failure_domain_classifier(
            "some_generic_test", "blocking_io", "", "",
        )
        assert result == "concurrency_failure"

    def test_unclassified_fallback(self):
        result = failure_domain_classifier(
            "generic_test", "", "", "",
        )
        assert result == "unclassified"

    def test_error_type_priority_over_category(self):
        """Error type should take priority over scenario category."""
        result = failure_domain_classifier(
            "some_test", "data_volume_scaling", "MemoryError", "",
        )
        assert result == "resource_exhaustion"

    def test_error_type_priority_over_name(self):
        """Error type should take priority over name keywords."""
        result = failure_domain_classifier(
            "concurrent_test", "", "MemoryError", "",
        )
        assert result == "resource_exhaustion"


# ══════════════════════════════════════════════════════════════════
# 2. failure_pattern_classifier
# ══════════════════════════════════════════════════════════════════


class TestFailurePatternClassifier:
    """Tests for failure_pattern_classifier."""

    def test_cache_growth_pattern(self):
        result = failure_pattern_classifier(
            "resource_exhaustion", "flask_cache_test", "",
        )
        assert result == "unbounded_cache_growth"

    def test_memory_leak_pattern(self):
        result = failure_pattern_classifier(
            "resource_exhaustion", "memory_leak_test", "",
        )
        assert result == "memory_accumulation_over_sessions"

    def test_oom_pattern(self):
        result = failure_pattern_classifier(
            "resource_exhaustion", "large_payload_oom_test", "",
        )
        assert result == "large_payload_oom"

    def test_deadlock_pattern(self):
        result = failure_pattern_classifier(
            "concurrency_failure", "request_deadlock_test", "",
        )
        assert result == "request_deadlock"

    def test_race_condition_pattern(self):
        result = failure_pattern_classifier(
            "concurrency_failure", "race_condition_check", "",
        )
        assert result == "race_condition"

    def test_gil_contention_pattern(self):
        result = failure_pattern_classifier(
            "concurrency_failure", "gil_contention_test", "",
        )
        assert result == "gil_contention"

    def test_throughput_pattern(self):
        result = failure_pattern_classifier(
            "scaling_collapse", "throughput_test", "",
        )
        assert result == "throughput_plateau"

    def test_empty_input_pattern(self):
        result = failure_pattern_classifier(
            "input_handling_failure", "empty_input_test", "",
        )
        assert result == "empty_input_crash"

    def test_encoding_pattern(self):
        result = failure_pattern_classifier(
            "input_handling_failure", "encoding_test", "",
        )
        assert result == "encoding_failure"

    def test_version_incompatibility_pattern(self):
        result = failure_pattern_classifier(
            "dependency_failure", "version_check", "",
        )
        assert result == "version_incompatibility"

    def test_cascading_timeout_pattern(self):
        result = failure_pattern_classifier(
            "integration_failure", "cascading_timeout", "",
        )
        assert result == "cascading_timeout"

    def test_install_failure_pattern(self):
        result = failure_pattern_classifier(
            "configuration_environment_failure", "pip_install_test", "",
        )
        assert result == "dependency_install_failure"

    def test_unclassified_domain_returns_none(self):
        result = failure_pattern_classifier(
            "unclassified", "some_scenario", "",
        )
        assert result is None

    def test_no_match_returns_none(self):
        result = failure_pattern_classifier(
            "resource_exhaustion", "totally_unrelated_scenario", "",
        )
        assert result is None

    def test_error_signature_match(self):
        result = failure_pattern_classifier(
            "resource_exhaustion", "generic_test", "cache overflow detected",
        )
        assert result == "unbounded_cache_growth"


# ══════════════════════════════════════════════════════════════════
# 3. operational_trigger_classifier
# ══════════════════════════════════════════════════════════════════


class TestOperationalTriggerClassifier:
    """Tests for operational_trigger_classifier."""

    def test_data_volume_scaling_category(self):
        result = operational_trigger_classifier("data_volume_scaling")
        assert result == "large_input"

    def test_memory_profiling_category(self):
        result = operational_trigger_classifier("memory_profiling")
        assert result == "sustained_load"

    def test_concurrent_execution_category(self):
        result = operational_trigger_classifier("concurrent_execution")
        assert result == "concurrent_access"

    def test_edge_case_input_category(self):
        result = operational_trigger_classifier("edge_case_input")
        assert result == "format_variation"

    def test_blocking_io_category(self):
        result = operational_trigger_classifier("blocking_io")
        assert result == "concurrent_access"

    def test_name_override_burst(self):
        result = operational_trigger_classifier(
            "memory_profiling", "burst_traffic_test",
        )
        assert result == "burst_traffic"

    def test_name_override_long_running(self):
        result = operational_trigger_classifier(
            "memory_profiling", "long_running_session_test",
        )
        assert result == "long_session"

    def test_name_override_concurrent(self):
        result = operational_trigger_classifier(
            "data_volume_scaling", "concurrent_access_test",
        )
        assert result == "concurrent_access"

    def test_name_override_large_payload(self):
        result = operational_trigger_classifier(
            "concurrent_execution", "large_payload_test",
        )
        assert result == "large_input"

    def test_unknown_category_defaults_sustained(self):
        result = operational_trigger_classifier("unknown_category")
        assert result == "sustained_load"

    def test_empty_category_defaults_sustained(self):
        result = operational_trigger_classifier("")
        assert result == "sustained_load"


# ══════════════════════════════════════════════════════════════════
# 4. vertical_classifier
# ══════════════════════════════════════════════════════════════════


class TestVerticalClassifier:
    """Tests for vertical_classifier."""

    def test_web_app_flask(self):
        result = vertical_classifier(["flask", "sqlalchemy"])
        assert result == "web_app"

    def test_web_app_django(self):
        result = vertical_classifier(["django", "gunicorn"])
        assert result == "web_app"

    def test_web_app_express(self):
        result = vertical_classifier(["express", "mongoose"])
        assert result == "web_app"

    def test_web_app_react(self):
        result = vertical_classifier(["react", "react-dom"])
        assert result == "web_app"

    def test_dashboard_streamlit(self):
        result = vertical_classifier(["streamlit", "pandas"])
        assert result == "dashboard"

    def test_dashboard_gradio(self):
        result = vertical_classifier(["gradio", "pandas"])
        assert result == "dashboard"

    def test_chatbot_langchain(self):
        result = vertical_classifier(["langchain", "openai", "chromadb"])
        assert result == "chatbot"

    def test_chatbot_llamaindex(self):
        result = vertical_classifier(["llama-index", "openai"])
        assert result == "chatbot"

    def test_data_pipeline_pandas(self):
        result = vertical_classifier(["pandas", "numpy", "scipy"])
        assert result == "data_pipeline"

    def test_ml_model_sklearn(self):
        result = vertical_classifier(["scikit-learn", "joblib", "mlflow"])
        assert result == "ml_model"

    def test_ml_model_pytorch(self):
        result = vertical_classifier(["torch", "torchvision"])
        assert result == "ml_model"

    def test_api_service_fastapi(self):
        result = vertical_classifier(["fastapi", "uvicorn"])
        assert result == "api_service"

    def test_utility_no_deps(self):
        result = vertical_classifier([])
        assert result == "utility"

    def test_utility_generic_deps(self):
        result = vertical_classifier(["pyyaml", "toml"])
        assert result == "utility"

    def test_file_structure_hints(self):
        result = vertical_classifier(
            ["pyyaml"], ["routes.py", "middleware.py", "templates/"],
        )
        assert result == "web_app"

    def test_dashboard_priority_over_web_app(self):
        """Dashboard (streamlit) should take priority over web_app."""
        result = vertical_classifier(["streamlit", "flask"])
        assert result == "dashboard"


# ══════════════════════════════════════════════════════════════════
# 5. architectural_pattern_classifier
# ══════════════════════════════════════════════════════════════════


class TestArchitecturalPatternClassifier:
    """Tests for architectural_pattern_classifier."""

    def test_flask_framework(self):
        result = architectural_pattern_classifier(
            ["flask"], framework="flask",
        )
        assert result == "web_app"

    def test_fastapi_framework(self):
        result = architectural_pattern_classifier(
            ["fastapi"], framework="fastapi",
        )
        assert result == "api_service"

    def test_streamlit_framework(self):
        result = architectural_pattern_classifier(
            ["streamlit"], framework="streamlit",
        )
        assert result == "dashboard"

    def test_langchain_dep(self):
        result = architectural_pattern_classifier(
            ["langchain", "openai"],
        )
        assert result == "chatbot"

    def test_pandas_dep(self):
        result = architectural_pattern_classifier(
            ["pandas", "numpy"],
        )
        assert result == "data_pipeline"

    def test_torch_dep(self):
        result = architectural_pattern_classifier(
            ["torch", "torchvision"],
        )
        assert result == "ml_model"

    def test_frontend_and_backend(self):
        result = architectural_pattern_classifier(
            [], file_count=20, has_frontend=True, has_backend=True,
        )
        assert result == "web_app"

    def test_frontend_only_small(self):
        result = architectural_pattern_classifier(
            [], file_count=5, has_frontend=True, has_backend=False,
        )
        assert result == "portfolio"

    def test_backend_only(self):
        result = architectural_pattern_classifier(
            [], file_count=20, has_frontend=False, has_backend=True,
        )
        assert result == "api_service"

    def test_no_signals_utility(self):
        result = architectural_pattern_classifier(
            [], file_count=10, has_frontend=False, has_backend=False,
        )
        assert result == "utility"


# ══════════════════════════════════════════════════════════════════
# Convenience functions
# ══════════════════════════════════════════════════════════════════


class TestClassifyFinding:
    """Tests for the classify_finding convenience function."""

    def test_returns_all_fields(self):
        result = classify_finding(
            "flask_memory_test", "memory_profiling", "MemoryError", "",
        )
        assert "failure_domain" in result
        assert "failure_pattern" in result
        assert "operational_trigger" in result

    def test_domain_and_trigger_populated(self):
        result = classify_finding(
            "concurrent_test", "concurrent_execution", "", "",
        )
        assert result["failure_domain"] == "concurrency_failure"
        assert result["operational_trigger"] == "concurrent_access"


class TestClassifyProject:
    """Tests for the classify_project convenience function."""

    def test_returns_all_fields(self):
        result = classify_project(["flask", "sqlalchemy"])
        assert "vertical" in result
        assert "architectural_pattern" in result

    def test_streamlit_project(self):
        result = classify_project(["streamlit", "pandas"])
        assert result["vertical"] == "dashboard"
        assert result["architectural_pattern"] == "dashboard"

    def test_empty_deps(self):
        result = classify_project([])
        assert result["vertical"] == "utility"
        assert result["architectural_pattern"] == "utility"


# ══════════════════════════════════════════════════════════════════
# Integration: classifiers wired into report
# ══════════════════════════════════════════════════════════════════


class TestReportIntegration:
    """Test that classifiers are wired into the report generator."""

    def test_finding_has_classification_fields(self):
        from mycode.report import Finding
        f = Finding(title="test", severity="warning")
        assert hasattr(f, "failure_domain")
        assert hasattr(f, "failure_pattern")
        assert hasattr(f, "operational_trigger")

    def test_report_has_classification_fields(self):
        from mycode.report import DiagnosticReport
        r = DiagnosticReport()
        assert hasattr(r, "vertical")
        assert hasattr(r, "architectural_pattern")

    def test_report_dict_includes_classification(self):
        from mycode.report import DiagnosticReport, Finding
        r = DiagnosticReport()
        f = Finding(
            title="test", severity="warning", category="memory_profiling",
            failure_domain="resource_exhaustion",
            failure_pattern="unbounded_cache_growth",
            operational_trigger="sustained_load",
        )
        r.findings = [f]
        r.vertical = "web_app"
        r.architectural_pattern = "web_app"
        d = r.as_dict()
        assert d["findings"][0]["failure_domain"] == "resource_exhaustion"
        assert d["findings"][0]["failure_pattern"] == "unbounded_cache_growth"
        assert d["findings"][0]["operational_trigger"] == "sustained_load"
        assert d["vertical"] == "web_app"
        assert d["architectural_pattern"] == "web_app"
