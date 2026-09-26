"""Reproducibility bookkeeping: seed rule must never drift silently."""
import hashlib

from src.provenance import (
    MANIFEST_SCHEMA_VERSION,
    PAGE_SEED_RULE,
    environment,
    page_seed,
    sha256_file,
)


def test_page_seed_golden_values():
    # Same values the original inline formula in pipeline.py produced.
    assert page_seed(0, 0) == 0
    assert page_seed(0, 5) == 5
    assert page_seed(1, 0) == 1_000_003
    assert page_seed(1, 2) == 1_000_005
    assert page_seed(-1, 0) == 2_146_483_645


def test_page_seed_in_31_bit_range_and_varies_per_page():
    seeds = [page_seed(123456789, i) for i in range(50)]
    assert all(0 <= s <= 0x7FFFFFFF for s in seeds)
    assert len(set(seeds)) == 50


def test_rule_string_matches_function():
    for m, i in [(0, 0), (3, 7), (-5, 2), (2**40, 1)]:
        expected = eval(PAGE_SEED_RULE, {"master_seed": m, "page_index": i})
        assert page_seed(m, i) == expected


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"dataset-maker" * 1000)
    assert sha256_file(str(p)) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_environment_and_schema():
    env = environment()
    assert set(env) == {"python", "numpy", "scipy", "pillow", "pymupdf"}
    assert all(isinstance(v, str) and v for v in env.values())
    assert MANIFEST_SCHEMA_VERSION == "1.1"
