"""Shared test fixtures for myCode tests."""

import sys
from pathlib import Path

# Add src to path so tests can import mycode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
