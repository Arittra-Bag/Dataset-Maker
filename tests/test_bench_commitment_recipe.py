"""The commitment recipe published in docs/BENCHMARK.md must match the build.

Runs the stdlib snippet from the card, as written there, against the
test_commitment of a real build, so the documented recipe cannot drift.
"""
import os
import re

import pytest

from src.bench import build, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = "00ff" * 8                                   # throwaway, tests only
GIVEN = f"  {SECRET.upper()}\n"                      # as a user might paste it


def _card_snippet() -> dict:
    with open(os.path.join(ROOT, "docs", "BENCHMARK.md")) as fh:
        card = fh.read()
    code = re.search(r"```python\n(import hashlib, json\n.*?)```", card, re.S).group(1)
    ns: dict = {}
    exec(code, ns)                                    # noqa: S102 (our own docs)
    return ns


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("rel"))
    # Built from the padded, uppercased form, so build.py's own
    # normalisation is exercised, not only the snippet's.
    spec = build.write_release(out, list(build.TIER_SPECS), dict(build.DEFAULT_DOCS), ["test"],
                               GIVEN, limit_docs=1)
    return out, spec["test_commitment"]


def test_card_recipe_reproduces_answers_sha256(release):
    out, commitment = release
    recipe = _card_snippet()
    records = [r for tier in build.TIER_SPECS
               for r in evaluate.split_answers(out, tier, "test", SECRET).values()]
    assert recipe["answers_sha256"](records) == commitment["answers_sha256"]


@pytest.mark.parametrize("given", [SECRET, SECRET.upper(), GIVEN])
def test_card_recipe_reproduces_secret_sha256(release, given):
    _, commitment = release
    assert _card_snippet()["secret_sha256"](given) == commitment["secret_sha256"]
