"""Threshold provider for measurement-based pattern gates.

Static dict lookup in v1. The signature accepts pattern_name only —
no project_metadata parameter. A future version can swap the
implementation for per-project calibration without changing call sites.

Follows the same swappable-backend pattern as llm.py.
"""

_THRESHOLDS = {
    "data_volume": {
        "peak_memory_mb": 47,
        "error_count": 11,
    },
    "cascading_timeout": {
        "error_count": 3,
    },
    "unbounded_cache_growth": {
        "peak_memory_mb": 50,
    },
    "input_handling_failure": {
        "error_count": 39,
    },
    "flask_concurrency": {
        "load_level": 5,
    },
    "requests_concurrent": {
        "load_level": 2,
        "error_count": 91,
    },
}

_CALIBRATION_DESCRIPTIONS = {
    "data_volume": (
        "Thresholds set at p25 of corpus distribution (9,297 reports, "
        "N=2,921 data_volume_scaling findings). peak_memory_mb=47 is "
        "the 25th percentile; error_count=11 is the 25th percentile."
    ),
    "cascading_timeout": (
        "Threshold set at p25 of corpus error_count distribution "
        "(N=467 cascading_timeout findings). error_count=3 captures "
        "75% of real cascading timeouts."
    ),
    "unbounded_cache_growth": (
        "Threshold set at 50MB — below p25 of general memory findings. "
        "Requires has_cache_decorator to confirm a cache mechanism "
        "exists in user code."
    ),
    "input_handling_failure": (
        "error_count=39 is the p25 of corpus distribution (N=143 "
        "unvalidated_type_crash findings). Exception marker requirement "
        "ensures the diagnosis names the actual exception type."
    ),
    "flask_concurrency": (
        "load_level=5 is the p25 of corpus distribution (N=886 flask "
        "concurrency findings). Thread saturation at <5 concurrent "
        "requests is implausible."
    ),
    "requests_concurrent": (
        "load_level=2 and error_count=91 are from corpus distribution "
        "(N=219). Requires I/O marker in details to confirm "
        "requests-specific failure."
    ),
}


def get_thresholds(pattern_name: str) -> dict:
    """Return threshold values for a given pattern.

    Returns a dict of threshold key -> value, e.g.:
        {"peak_memory_mb": 47, "error_count": 11}
    """
    return _THRESHOLDS[pattern_name]


def describe_calibration(pattern_name: str) -> str:
    """Return a human-readable description of how thresholds were set.

    Used for explainability in diagnostic output (Product Architecture
    Section 2 user-comprehension principle).
    """
    return _CALIBRATION_DESCRIPTIONS[pattern_name]
