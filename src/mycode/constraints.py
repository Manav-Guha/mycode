"""Operational Constraints — structured extraction from user conversation.

The constraint object is the bridge between user intent (Component 4) and
scenario generation (Component 5) / report generation (Component 7).

Produced by the Conversational Interface from the user's plain-language
answers.  Each field may be ``None`` if the user didn't provide enough
information to extract it.  Downstream components use defaults for ``None``
values and document this in the report.

Per spec §4, the constraint object schema is:
    user_scale, usage_pattern, max_payload_mb, data_type,
    deployment_context, availability_requirement, data_sensitivity,
    growth_expectation, raw_answers.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Data Class ──


@dataclass
class OperationalConstraints:
    """Structured constraints extracted from user conversation.

    Attributes:
        user_scale: Expected concurrent users (e.g., 20).  When
            ``max_users`` is set, ``user_scale`` mirrors it for
            backward compatibility.
        current_users: User's current scale (baseline).
        max_users: User's growth target (ceiling).
        usage_pattern: How the app is used — ``"steady"``,
            ``"burst"``, ``"on_demand"``, or ``"growing"``.
        max_payload_mb: Largest expected input/file size in megabytes.
        per_user_data: Typical data per user — ``"small"``,
            ``"medium"``, ``"large"``, or free text.
        max_total_data: Maximum total data for the application —
            e.g. ``"10GB"``, ``"1M rows"``, or free text.
        data_type: Kind of data handled — ``"tabular"``, ``"text"``,
            ``"images"``, ``"mixed"``, or ``"api_responses"``.
        deployment_context: Where it runs — ``"single_server"``,
            ``"local_only"``, ``"cloud"``, or ``"shared_hosting"``.
        availability_requirement: Uptime expectation — ``"always_on"``,
            ``"business_hours"``, or ``"occasional"``.
        data_sensitivity: Data classification — ``"public"``,
            ``"internal"``, ``"customer_data"``, ``"financial"``,
            or ``"medical"``.
        growth_expectation: Expected trajectory — ``"stable"``,
            ``"slow_growth"``, or ``"rapid_growth"``.
        project_description: Free-text description of what the project
            does (from Section 1 of the web form or Turn 1 of CLI).
        assumptions_used: True when non-interactive mode applied
            corpus-derived or hardcoded defaults instead of user input.
        assumed_values: Dict of field→value for assumptions made in
            non-interactive mode.
        raw_answers: Original user answers preserved for report
            contextualisation.
    """

    user_scale: Optional[int] = None
    current_users: Optional[int] = None
    max_users: Optional[int] = None
    usage_pattern: Optional[str] = None
    max_payload_mb: Optional[float] = None
    per_user_data: Optional[str] = None
    max_total_data: Optional[str] = None
    data_type: Optional[str] = None
    data_type_detail: Optional[str] = None
    deployment_context: Optional[str] = None
    availability_requirement: Optional[str] = None
    data_sensitivity: Optional[str] = None
    growth_expectation: Optional[str] = None
    timeout_per_scenario: Optional[int] = None
    analysis_depth: Optional[str] = None  # "quick", "standard", "deep"
    project_description: Optional[str] = None
    assumptions_used: bool = False
    assumed_values: dict = field(default_factory=dict)
    raw_answers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sync user_scale ↔ max_users for backward compatibility."""
        if self.max_users is not None and self.user_scale is None:
            self.user_scale = self.max_users
        elif self.user_scale is not None and self.max_users is None:
            self.max_users = self.user_scale

    def as_summary(self) -> str:
        """Human-readable summary of extracted constraints.

        Used as a backward-compatible Turn 2 equivalent when populating
        ``OperationalIntent.stress_priorities``.
        """
        parts: list[str] = []
        if self.current_users is not None and self.max_users is not None:
            parts.append(
                f"Current users: {self.current_users}, "
                f"max target: {self.max_users}"
            )
        elif self.user_scale is not None:
            parts.append(f"Expected concurrent users: {self.user_scale}")
        if self.data_type:
            labels = {
                "tabular": "tabular data (CSV/spreadsheets)",
                "text": "text/strings/logs",
                "documents": "documents/PDFs",
                "images": "images/media/files",
                "api_responses": "API responses/JSON",
                "mixed": "mixed data types",
            }
            parts.append(
                f"Data: {labels.get(self.data_type, self.data_type)}"
            )
        if self.per_user_data:
            parts.append(f"Typical data per user: {self.per_user_data}")
        if self.max_total_data:
            parts.append(f"Max total data: {self.max_total_data}")
        if self.usage_pattern:
            labels = {
                "sustained": "steady, continuous use",
                "steady": "steady, continuous use",
                "burst": "burst/peak usage patterns",
                "on_demand": "occasional, on-demand use",
                "periodic": "occasional, on-demand use",
                "growing": "growing usage over time",
            }
            parts.append(
                f"Usage: {labels.get(self.usage_pattern, self.usage_pattern)}"
            )
        if self.max_payload_mb is not None:
            parts.append(f"Typical input size: up to {self.max_payload_mb} MB")
        if self.deployment_context:
            parts.append(f"Deployment: {self.deployment_context}")
        if self.availability_requirement:
            parts.append(f"Availability: {self.availability_requirement}")
        if self.data_sensitivity:
            parts.append(f"Data sensitivity: {self.data_sensitivity}")
        if self.growth_expectation:
            parts.append(f"Growth: {self.growth_expectation}")
        if self.analysis_depth:
            _depth_labels = {
                "quick": "quick scan (~2 minutes)",
                "standard": "standard analysis (~5 minutes)",
                "deep": "deep analysis (~10 minutes)",
            }
            parts.append(
                f"Analysis depth: "
                f"{_depth_labels.get(self.analysis_depth, self.analysis_depth)}"
            )
        elif self.timeout_per_scenario is not None:
            parts.append(f"Time limit per test: {self.timeout_per_scenario}s")
        return ". ".join(parts) if parts else "No specific constraints provided"


# ── Skip Detection ──

_SKIP_WORDS = frozenset({
    "not sure", "no idea", "don't know", "dont know", "idk",
    "n/a", "na", "skip", "unsure", "no clue", "pass", "none",
})


def _is_skip(text: str) -> bool:
    """Return True if the user is declining to answer."""
    stripped = text.strip().lower()
    return not stripped or stripped in _SKIP_WORDS


# ── Parsers ──


def parse_user_scale(text: str) -> Optional[int]:
    """Extract a concurrent-user count from free text.

    Handles plain numbers (``"20"``), numbers with context
    (``"about 50 users"``), ranges (``"10-20"``), k/m suffixes
    (``"5k"``), and word-based estimates (``"a few hundred"``).

    Returns ``None`` when the input is empty, a skip word, or
    contains no recognisable quantity.
    """
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    # Range: "10-20 users", "10 to 20"
    range_match = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)', text_lower)
    if range_match:
        high = int(range_match.group(2))
        if high > 0:
            return high

    # k/m suffix: "5k", "2m"
    suffix_match = re.search(r'(\d+(?:\.\d+)?)\s*([km])\b', text_lower)
    if suffix_match:
        value = float(suffix_match.group(1))
        suffix = suffix_match.group(2)
        if suffix == "k":
            return int(value * 1_000)
        return int(value * 1_000_000)

    # Plain number
    num_match = re.search(r'\b(\d{1,7})\b', text_lower)
    if num_match:
        n = int(num_match.group(1))
        if n > 0:
            return n

    # Word-based estimates (longest phrases first)
    _WORD_SCALES: list[tuple[str, int]] = [
        ("tens of thousands", 20_000),
        ("hundred thousand", 100_000),
        ("a few thousand", 3_000),
        ("a few hundred", 300),
        ("a few dozen", 30),
        ("thousands", 2_000),
        ("hundreds", 200),
        ("million", 1_000_000),
        ("a thousand", 1_000),
        ("a hundred", 100),
        ("a dozen", 12),
        ("dozen", 12),
        ("just me", 1),
        ("only me", 1),
        ("myself", 1),
        ("couple", 2),
        ("a few", 5),
        ("handful", 5),
        ("tens", 30),
    ]
    for phrase, value in _WORD_SCALES:
        if phrase in text_lower:
            return value

    return None


# ── Data Type ──

_DATA_TYPE_CHOICES: dict[str, str] = {
    "1": "tabular",
    "2": "text",
    "3": "documents",
    "4": "images",
    "5": "api_responses",
    "6": "mixed",
}

_DATA_TYPE_KEYWORDS: dict[str, list[str]] = {
    "tabular": [
        "csv", "spreadsheet", "excel", "xls", "xlsx", "table", "tabular",
        "dataframe", "pandas", "column", "row", "sql", "database",
    ],
    "text": [
        "text", "document", "string", "log", "email",
        "chat", "message", "markdown", "txt",
    ],
    "documents": [
        "pdf", "pdfs", "word", "docx", "doc", "pptx",
        "powerpoint", "odt",
    ],
    "images": [
        "image", "photo", "picture", "video", "media",
        "audio", "png", "jpg", "jpeg", "gif", "svg",
    ],
    "api_responses": [
        "api", "json", "endpoint", "rest",
        "webhook", "graphql", "xml",
    ],
    "mixed": [
        "mix", "various", "different types", "everything",
        "all kinds", "multiple", "combination",
    ],
}


def parse_data_type(text: str) -> Optional[str]:
    """Extract data type from a numbered choice or keyword match."""
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    # Numbered choice
    stripped = text_lower.rstrip(".").strip()
    if stripped in _DATA_TYPE_CHOICES:
        return _DATA_TYPE_CHOICES[stripped]

    # Keyword scoring — highest match count wins, but if keywords hit
    # two or more *distinct* categories the user is describing mixed data.
    scores: dict[str, int] = {}
    for dtype, keywords in _DATA_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[dtype] = score

    if not scores:
        return None

    # When keywords span two or more distinct categories the user is
    # describing mixed data (e.g. "PDFs, TXT, XLS" hits text + tabular).
    non_mixed = {k: v for k, v in scores.items() if k != "mixed"}
    if len(non_mixed) >= 2:
        return "mixed"

    return max(scores, key=scores.get)


_GENERIC_DATA_KEYWORDS: frozenset[str] = frozenset({
    "text", "document", "image", "media", "string", "photo", "picture",
    "video", "audio", "message", "log", "email", "chat",
})
"""Category-level words too vague for a user-facing detail label.

When the user says "PDFs, JPEGs, XLS documents", the matched keywords are
``pdf``, ``jpeg``, ``xls``, ``document``.  We want the detail to read
"PDF, JPEG and XLS files" — not "PDF, JPEG, XLS and document files".
Generic words are kept only when no specific keywords were found.
"""


def extract_data_type_keywords(text: str) -> list[str]:
    """Return the specific file-type / data-type keywords found in *text*.

    Used to build a human-readable description that preserves the user's
    original phrasing (e.g. ``["PDFs", "XLS"]`` instead of ``"mixed"``).

    The returned strings use the ORIGINAL casing from *text* — not the
    lower-case keyword form — so they read naturally in reports.
    Generic category words (``"document"``, ``"image"``, …) are dropped
    when more specific terms are present.
    """
    if not text:
        return []
    text_lower = text.lower()
    all_keywords: list[str] = []
    for keywords in _DATA_TYPE_KEYWORDS.values():
        all_keywords.extend(keywords)

    found: list[str] = []
    seen_lower: set[str] = set()
    for kw in all_keywords:
        if kw in text_lower and kw not in seen_lower:
            seen_lower.add(kw)
            idx = text_lower.index(kw)
            found.append(text[idx : idx + len(kw)])

    # Drop generic words only when enough specific terms remain to be
    # meaningful.  "PDFs, JPEGs, XLS documents" → drop "documents" (3
    # specific remain).  "TXT and images" → keep "images" (only 1 specific
    # without it).
    specific = [w for w in found if w.lower() not in _GENERIC_DATA_KEYWORDS]
    if len(specific) >= 2:
        return specific
    return found


def _format_data_type_detail(keywords: list[str]) -> str:
    """Format extracted keywords into a readable phrase.

    ``["PDFs", "XLS"]`` → ``"PDFs and XLS files"``
    ``["csv"]`` → ``"CSV files"``
    """
    if not keywords:
        return ""
    # Uppercase short extensions (pdf → PDF, xls → XLS, csv → CSV)
    parts = [kw.upper() if len(kw) <= 4 and kw.isalpha() else kw for kw in keywords]
    if len(parts) == 1:
        return f"{parts[0]} files"
    return ", ".join(parts[:-1]) + " and " + parts[-1] + " files"


# ── Usage Pattern ──

_USAGE_PATTERN_CHOICES: dict[str, str] = {
    "1": "steady",
    "2": "burst",
    "3": "on_demand",
    "4": "growing",
}

# Canonical aliases — old values map to new canonical forms.
_USAGE_PATTERN_ALIASES: dict[str, str] = {
    "sustained": "steady",
    "periodic": "on_demand",
}

_USAGE_PATTERN_KEYWORDS: dict[str, list[str]] = {
    "steady": [
        "steady", "constant", "continuous", "all day",
        "always", "throughout", "nonstop", "24/7", "sustained",
    ],
    "burst": [
        "burst", "peak", "rush", "spike", "end of month",
        "deadline", "morning", "evening",
    ],
    "on_demand": [
        "occasional", "sometimes", "now and then", "weekly",
        "monthly", "periodic", "on demand", "when needed",
        "once in a while", "rarely", "infrequent",
    ],
    "growing": [
        "growing", "scaling", "increasing", "more and more",
        "ramping", "expanding", "growth",
    ],
}


def parse_usage_pattern(text: str) -> Optional[str]:
    """Extract usage pattern from a numbered choice or keyword match.

    Returns canonical values: ``"steady"``, ``"burst"``, ``"on_demand"``,
    ``"growing"``.  Old values (``"sustained"``, ``"periodic"``) are
    normalised to their canonical equivalents.
    """
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    # Direct value (e.g. from web pill)
    stripped = text_lower.rstrip(".").strip()
    if stripped in _USAGE_PATTERN_ALIASES:
        return _USAGE_PATTERN_ALIASES[stripped]
    if stripped in _USAGE_PATTERN_CHOICES:
        return _USAGE_PATTERN_CHOICES[stripped]

    scores: dict[str, int] = {}
    for pattern, keywords in _USAGE_PATTERN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[pattern] = score

    if scores:
        return max(scores, key=scores.get)

    return None


def infer_availability(usage_pattern: Optional[str]) -> Optional[str]:
    """Derive ``availability_requirement`` from the extracted usage pattern.

    Mapping:
        sustained → always_on,  burst → business_hours,
        periodic  → occasional, growing → always_on.
    """
    if usage_pattern is None:
        return None
    return {
        "sustained": "always_on",
        "steady": "always_on",
        "burst": "business_hours",
        "periodic": "occasional",
        "on_demand": "occasional",
        "growing": "always_on",
    }.get(usage_pattern)


# ── Max Payload ──

_PAYLOAD_CHOICES: dict[str, float] = {
    "1": 1.0,      # small (under 1 MB)
    "2": 50.0,     # medium (1–50 MB)
    "3": 100.0,    # large (over 50 MB)
}


def parse_max_payload(text: str) -> Optional[float]:
    """Extract maximum payload size in megabytes.

    Handles numbered choices, explicit sizes with units
    (``"50 MB"``, ``"2 GB"``), and descriptive keywords
    (``"small"``, ``"large"``).
    """
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    # Numbered choice
    stripped = text_lower.rstrip(".").strip()
    if stripped in _PAYLOAD_CHOICES:
        return _PAYLOAD_CHOICES[stripped]

    # Number + unit: "50mb", "2 gb", "500 kb"
    unit_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(gb|mb|kb|g|m|k)\b', text_lower,
    )
    if unit_match:
        value = float(unit_match.group(1))
        unit = unit_match.group(2)
        if unit in ("gb", "g"):
            return value * 1_000.0
        if unit in ("mb", "m"):
            return value
        if unit in ("kb", "k"):
            return value / 1_000.0

    # Descriptive keywords
    if any(w in text_lower for w in ("small", "tiny", "little", "light")):
        return 1.0
    if any(w in text_lower for w in ("medium", "moderate", "average")):
        return 50.0
    if any(w in text_lower for w in ("large", "big", "heavy", "huge", "massive")):
        return 100.0

    # Plain number without unit — assume MB
    plain_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text_lower)
    if plain_match:
        value = float(plain_match.group(1))
        if value > 0:
            return value

    return None


# ── Timeout Per Scenario ──

_TIMEOUT_FLOOR = 60

_TIMEOUT_CHOICES: dict[str, int] = {
    "1": 90,    # default
    "2": 180,   # 3 minutes
    "3": 300,   # 5 minutes
}


def parse_timeout_per_scenario(text: str) -> Optional[int]:
    """Extract per-scenario timeout in seconds.

    Handles numbered choices, explicit durations with units
    (``"2 min"``, ``"120s"``, ``"5 minutes"``), and plain numbers
    (assumed seconds).  Enforces a 60-second floor.

    Returns ``None`` when the input is a skip word or empty.
    """
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    # Numbered choice
    stripped = text_lower.rstrip(".").strip()
    if stripped in _TIMEOUT_CHOICES:
        return _TIMEOUT_CHOICES[stripped]

    # Number + unit: "2 min", "120s", "5 minutes", "3m"
    unit_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|m|sec(?:ond)?s?|s)\b', text_lower,
    )
    if unit_match:
        value = float(unit_match.group(1))
        unit = unit_match.group(2)
        if unit.startswith("m"):
            seconds = int(value * 60)
        else:
            seconds = int(value)
        return max(seconds, _TIMEOUT_FLOOR)

    # Plain number — assume seconds
    plain_match = re.search(r'\b(\d+)\b', text_lower)
    if plain_match:
        value = int(plain_match.group(1))
        if value > 0:
            return max(value, _TIMEOUT_FLOOR)

    return None


# ── Analysis Depth ──

_DEPTH_CHOICES: dict[str, str] = {
    "1": "quick",
    "2": "standard",
    "3": "deep",
}

_DEPTH_KEYWORDS: dict[str, str] = {
    "quick": "quick",
    "fast": "quick",
    "quick scan": "quick",
    "standard": "standard",
    "normal": "standard",
    "default": "standard",
    "deep": "deep",
    "thorough": "deep",
    "comprehensive": "deep",
    "full": "deep",
}


def parse_analysis_depth(text: str) -> Optional[str]:
    """Extract analysis depth from user input.

    Returns ``"quick"``, ``"standard"``, ``"deep"``, or ``None``.
    """
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    stripped = text_lower.rstrip(".").strip()
    if stripped in _DEPTH_CHOICES:
        return _DEPTH_CHOICES[stripped]

    for keyword, depth in _DEPTH_KEYWORDS.items():
        if keyword in text_lower:
            return depth

    return None


def depth_to_timeout(depth: str) -> int:
    """Map analysis depth to per-scenario timeout in seconds."""
    return {"quick": 120, "standard": 300, "deep": 600}.get(depth, 300)


def depth_to_coupling_cap(depth: str) -> Optional[int]:
    """Map analysis depth to coupling scenario cap.

    Returns ``None`` for deep (no cap).
    """
    return {"quick": 5, "standard": 10, "deep": None}.get(depth, 10)


# ── Context-Only Parsers ──
# These are run against the Turn 1 free-text answer to extract
# fields that don't have dedicated structured questions.

_DEPLOYMENT_KEYWORDS: dict[str, list[str]] = {
    "local_only": [
        "local", "localhost", "my machine", "my computer",
        "my laptop", "desktop app",
    ],
    "single_server": [
        "server", "vps", "droplet", "ec2 instance",
        "single server", "one server",
    ],
    "cloud": [
        "cloud", "aws", "gcp", "azure", "heroku", "vercel",
        "netlify", "railway", "fly.io", "render",
    ],
    "shared_hosting": [
        "shared hosting", "cpanel", "shared server",
    ],
}


def parse_deployment_context(text: str) -> Optional[str]:
    """Extract deployment context from free text (Turn 1 scan)."""
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for ctx, keywords in _DEPLOYMENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[ctx] = score

    if scores:
        return max(scores, key=scores.get)
    return None


_SENSITIVITY_KEYWORDS: dict[str, list[str]] = {
    "medical": [
        "medical", "health", "patient", "hipaa", "clinical",
        "hospital", "diagnosis",
    ],
    "financial": [
        "financial", "payment", "bank", "money", "invoice",
        "budget", "transaction", "billing", "accounting",
        "credit card", "stripe",
    ],
    "customer_data": [
        "customer", "user data", "personal", "pii",
        "registration", "sign up",
    ],
    "internal": [
        "internal", "company", "team", "enterprise",
        "staff", "employee",
    ],
    "public": [
        "public", "open source", "blog", "portfolio",
        "landing page",
    ],
}


def parse_data_sensitivity(text: str) -> Optional[str]:
    """Extract data sensitivity level from free text (Turn 1 scan)."""
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for level, keywords in _SENSITIVITY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[level] = score

    if scores:
        return max(scores, key=scores.get)
    return None


_GROWTH_KEYWORDS: dict[str, list[str]] = {
    "rapid_growth": [
        "rapid growth", "fast growth", "scaling up", "viral",
        "hockey stick", "doubling", "explosive",
    ],
    "slow_growth": [
        "slow growth", "gradual", "steady growth", "organic",
    ],
    "stable": [
        "stable", "not changing", "fixed", "consistent",
        "flat", "no growth",
    ],
}


def parse_growth_expectation(text: str) -> Optional[str]:
    """Extract growth expectation from free text (Turn 1 scan)."""
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for exp, keywords in _GROWTH_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[exp] = score

    if scores:
        return max(scores, key=scores.get)
    return None


# ── Per-User Data ──

_PER_USER_DATA_CATEGORIES: dict[str, str] = {
    "1": "small",
    "2": "medium",
    "3": "large",
}


def parse_per_user_data(text: str) -> Optional[str]:
    """Extract per-user data size category or description.

    Accepts numbered choices (``"1"`` = small), category keywords
    (``"small"``, ``"medium"``, ``"large"``), or free text that is
    stored verbatim.
    """
    text_stripped = text.strip()
    if _is_skip(text_stripped.lower()):
        return None

    stripped = text_stripped.lower().rstrip(".").strip()
    if stripped in _PER_USER_DATA_CATEGORIES:
        return _PER_USER_DATA_CATEGORIES[stripped]
    if stripped in ("small", "medium", "large"):
        return stripped

    # Store free text as-is (e.g. "about 50 rows", "a few hundred KB")
    if text_stripped:
        return text_stripped
    return None


def parse_max_total_data(text: str) -> Optional[str]:
    """Extract maximum total data description.

    Accepts numbered choices, size expressions (``"10GB"``,
    ``"1M rows"``), category keywords, or free text.
    """
    text_stripped = text.strip()
    if _is_skip(text_stripped.lower()):
        return None

    stripped = text_stripped.lower().rstrip(".").strip()
    if stripped in _PER_USER_DATA_CATEGORIES:
        return _PER_USER_DATA_CATEGORIES[stripped]
    if stripped in ("small", "medium", "large"):
        return stripped

    if text_stripped:
        return text_stripped
    return None


# ── Data Size Conversion ──

_PER_USER_ITEM_DEFAULTS: dict[str, int] = {
    "small": 50,
    "medium": 500,
    "large": 5000,
}

_MAX_TOTAL_ITEM_DEFAULTS: dict[str, int] = {
    "small": 1000,
    "medium": 10000,
    "large": 100000,
}


def per_user_data_to_items(per_user_data: Optional[str]) -> int:
    """Convert per-user data description to an item count for testing."""
    if not per_user_data:
        return 100  # sensible default
    lower = per_user_data.lower().strip()
    if lower in _PER_USER_ITEM_DEFAULTS:
        return _PER_USER_ITEM_DEFAULTS[lower]

    # Try to extract a number
    m = re.search(r'(\d+(?:\.\d+)?)\s*([km])?\b', lower)
    if m:
        val = float(m.group(1))
        suffix = m.group(2)
        if suffix == "k":
            return int(val * 1000)
        if suffix == "m":
            return int(val * 1_000_000)
        return max(1, int(val))

    return 100


def max_total_data_to_items(max_total_data: Optional[str]) -> int:
    """Convert max total data description to an item count for testing."""
    if not max_total_data:
        return 10000  # sensible default
    lower = max_total_data.lower().strip()
    if lower in _MAX_TOTAL_ITEM_DEFAULTS:
        return _MAX_TOTAL_ITEM_DEFAULTS[lower]

    # Try to extract a number with optional suffix
    m = re.search(r'(\d+(?:\.\d+)?)\s*([km])?\b', lower)
    if m:
        val = float(m.group(1))
        suffix = m.group(2)
        if suffix == "k":
            return int(val * 1000)
        if suffix == "m":
            return int(val * 1_000_000)
        return max(1, int(val))

    return 10000
