"""CLAUDE.md invariant 1: no LLM calls anywhere in ``null/``.

Not for explanations, not for report prose, not "just this once" behind a flag.
If an LLM can influence a verdict, the verdict is worthless (BUILD.md section 0).

This is a source-text grep, deliberately. An import-graph check would miss a
subprocess shell-out to a CLI, an HTTP call built from a string, or a lazily
imported module. Grep catches the string wherever it hides.

Every scan here is paired with a negative control that plants the violation in a
temporary directory and asserts the scanner catches it. A guard test that has
never been seen to fail is not a guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "null"

# Vendor SDKs, hosted-inference clients, local runtimes, and the call shapes and
# model-id prefixes they are reached through.
FORBIDDEN_LLM_PATTERNS: tuple[str, ...] = (
    r"\banthropic\b",
    r"\bopenai\b",
    r"\blangchain\b",
    r"\bllama_index\b",
    r"\bllama_cpp\b",
    r"\blitellm\b",
    r"\bollama\b",
    r"\bcohere\b",
    r"\bmistralai\b",
    r"\bhuggingface\b",
    r"\btransformers\b",
    r"\bgenerativeai\b",
    r"\bvertexai\b",
    r"\bbedrock\b",
    # Narrowed from r"\breplicate\b". "replicate"/"replicates" is standard
    # statistics vocabulary -- bootstrap replicates -- and the broad form fired on
    # legitimate code in null/stats/. A guard that cries wolf gets disabled, which
    # is worse than a guard scoped to the thing it actually cares about.
    r"\bimport\s+replicate\b",
    r"\breplicate\.(?:com|run)\b",
    r"chat\.completions",
    r"completions\.create",
    r"messages\.create",
    r"generate_content",
    r"\bgpt-\d",
    r"\bclaude-\w",
    r"\bgemini-\w",
    r"\bllama-?\d",
)

# Each control must be caught by at least one pattern above. These strings live in
# tests/, which is never scanned, so they cannot trip the scan on themselves.
LLM_NEGATIVE_CONTROLS: tuple[str, ...] = (
    "from anthropic import Anthropic\n",
    "import openai\n",
    "resp = client.chat.completions.create(model='x')\n",
    "MODEL = 'gpt-4o'\n",
    "def explain(): return llm.generate_content(prompt)\n",
)


def scan(root: Path, pattern: str) -> list[str]:
    """Every ``pattern`` hit under ``root``, as ``path:lineno: line`` strings."""
    compiled = re.compile(pattern, re.IGNORECASE)
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if compiled.search(line):
                hits.append(f"{path.name}:{lineno}: {line.strip()}")
    return hits


def test_package_root_exists_and_has_sources() -> None:
    """A vacuous pass would make this whole module theatre."""
    assert PACKAGE_ROOT.is_dir(), f"{PACKAGE_ROOT} missing"
    assert sorted(PACKAGE_ROOT.rglob("*.py")), "no sources under null/ -- scan is empty"


@pytest.mark.parametrize("pattern", FORBIDDEN_LLM_PATTERNS)
def test_no_llm_references_in_null_package(pattern: str) -> None:
    """No file under ``null/`` may mention an LLM vendor, client, or model id."""
    hits = scan(PACKAGE_ROOT, pattern)
    assert not hits, (
        f"LLM reference /{pattern}/ found inside null/. NULL's verdict must be "
        "bit-for-bit reproducible; an LLM in the audit path makes it worthless.\n"
        + "\n".join(hits)
    )


@pytest.mark.parametrize("planted", LLM_NEGATIVE_CONTROLS)
def test_scanner_catches_a_planted_llm_call(planted: str, tmp_path: Path) -> None:
    """Negative control: prove the scan fails when a violation is really there."""
    (tmp_path / "sneaky.py").write_text(planted, encoding="utf-8")
    caught = [p for p in FORBIDDEN_LLM_PATTERNS if scan(tmp_path, p)]
    assert caught, f"no forbidden pattern matched {planted!r} -- the scan has a hole"


def test_scanner_is_quiet_on_clean_source(tmp_path: Path) -> None:
    """The other half of the control: no pattern fires on innocent code."""
    (tmp_path / "clean.py").write_text(
        "import math\n\n\ndef sharpe(mu: float, sigma: float) -> float:\n"
        "    return mu / sigma * math.sqrt(252)\n",
        encoding="utf-8",
    )
    assert not [p for p in FORBIDDEN_LLM_PATTERNS if scan(tmp_path, p)]
