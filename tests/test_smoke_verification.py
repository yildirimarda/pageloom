"""Verification of the live smoke-test claim for Milestone 7 audit."""

import os
import subprocess
import sys

import pytest


def test_smoke_vision_reports_rate_limit_reached():
    """Live smoke test (smoke_vision.py) confirms free-tier endpoint reachable
    and reports the known rate-limit message.

    This verifies the claim in PLAN.md line 270 that OpenRouter's free tier
    (`openrouter/free`) reaches the endpoint successfully but hits the
    `free-models-per-day-high-balance` rate limit.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("No OPENROUTER_API_KEY set for smoke test")

    result = subprocess.run(
        [sys.executable, "smoke_vision.py"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        timeout=60,
        check=False,
    )
    # The script exits 0 for an expected rate-limit outcome.
    assert result.returncode == 0, f"Unexpected exit: {result.returncode}"
    combined = (result.stdout + result.stderr).lower()
    # Confirm free-tier endpoint was reached.
    assert "openrouter/free" in (result.stdout + result.stderr)
    assert "image encoding: ok" in (result.stdout + result.stderr).lower()
    # Confirm the rate-limit message is reported.
    assert "rate limit" in combined
    assert "free tier" in combined or "rate limit" in combined
