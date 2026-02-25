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
        user_scale: Expected concurrent users (e.g., 20).
        usage_pattern: How the app is used — ``"sustained"``,
            ``"burst"``, ``"periodic"``, or ``"growing"``.
        max_payload_mb: Largest expected input/file size in megabytes.
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
        raw_answers: Original user answers preserved for report
            contextualisation.
    """

    user_scale: Optional[int] = None
    usage_pattern: Optional[str] = None
    max_payload_mb: Optional[float] = None
    data_type: Optional[str] = None
    deployment_context: Optional[str] = None
    availability_requirement: Optional[str] = None
    data_sensitivity: Optional[str] = None
    growth_expectation: Optional[str] = None
    raw_answers: list[str] = field(default_factory=list)

    def as_summary(self) -> str:
        """Human-readable summary of extracted constraints.

        Used as a backward-compatible Turn 2 equivalent when populating
        ``OperationalIntent.stress_priorities``.
        """
        parts: list[str] = []
        if self.user_scale is not None:
            parts.append(f"Expected concurrent users: {self.user_scale}")
        if self.data_type:
            labels = {
                "tabular": "tabular data (CSV/spreadsheets)",
                "text": "text/documents",
                "images": "images/media/files",
                "api_responses": "API responses/JSON",
                "mixed": "mixed data types",
            }
            parts.append(
                f"Data: {labels.get(self.data_type, self.data_type)}"
            )
        if self.usage_pattern:
            labels = {
                "sustained": "steady, continuous use",
                "burst": "burst/peak usage patterns",
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
    "3": "images",
    "4": "api_responses",
    "5": "mixed",
}

_DATA_TYPE_KEYWORDS: dict[str, list[str]] = {
    "tabular": [
        "csv", "spreadsheet", "excel", "table", "tabular",
        "dataframe", "pandas", "column", "row", "sql", "database",
    ],
    "text": [
        "text", "document", "string", "log", "email",
        "chat", "message", "markdown", "pdf",
    ],
    "images": [
        "image", "photo", "picture", "video", "media",
        "upload", "audio", "png", "jpg", "jpeg",
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

    # Keyword scoring — highest match count wins
    scores: dict[str, int] = {}
    for dtype, keywords in _DATA_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[dtype] = score

    if scores:
        return max(scores, key=scores.get)

    return None


# ── Usage Pattern ──

_USAGE_PATTERN_CHOICES: dict[str, str] = {
    "1": "sustained",
    "2": "burst",
    "3": "periodic",
    "4": "growing",
}

_USAGE_PATTERN_KEYWORDS: dict[str, list[str]] = {
    "sustained": [
        "steady", "constant", "continuous", "all day",
        "always", "throughout", "nonstop", "24/7",
    ],
    "burst": [
        "burst", "peak", "rush", "spike", "end of month",
        "deadline", "morning", "evening",
    ],
    "periodic": [
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
    """Extract usage pattern from a numbered choice or keyword match."""
    text_lower = text.lower().strip()

    if _is_skip(text_lower):
        return None

    stripped = text_lower.rstrip(".").strip()
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
        "burst": "business_hours",
        "periodic": "occasional",
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
