"""Benchmark builder: pure-function pages, held-out test, no leaks, integrity."""
import json
import os

import pytest

from src.bench import build, canon

SECRET = "00112233445566778899aabbccddeeff"      # throwaway, tests only
DOCS = {"train": 2, "val": 1, "test-dev": 1, "test": 1}


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("rel"))
    spec = build.write_release(out, ["easy", "hard"], DOCS, ["train", "val", "test"], SECRET)
    return out, spec


def _read(root, rel):
    with open(os.path.join(root, *rel.split("/")), "rb") as fh:
        return fh.read()


def test_page_is_pure_function_of_key():
    a = build.build_doc(build.PUBLIC_ENTROPY, "hard", "train", 1)
    b = build.build_doc(build.PUBLIC_ENTROPY, "hard", "train", 1)
    assert [p.page_id for p in a] == [p.page_id for p in b]
    assert [canon.dumps(p.answer) for p in a] == [canon.dumps(p.answer) for p in b]
    assert [p.pngs for p in a] == [p.pngs for p in b]


def test_slice_is_byte_subset_of_full_build(release, tmp_path):
    full, _ = release
    build.write_release(str(tmp_path), ["hard"], DOCS, ["train"], limit_docs=1)
    sums_full = set(_read(full, "SHA256SUMS").decode().splitlines())
    for line in _read(str(tmp_path), "SHA256SUMS").decode().splitlines():
        if not line.endswith("  benchmark.json"):
            assert line in sums_full


def test_test_split_is_held_out(release):
    root, spec = release
    assert not os.path.exists(os.path.join(root, "answers", "easy", "test"))
    assert os.listdir(os.path.join(root, "puzzles", "easy", "test"))
    commit = spec["test_commitment"]
    prefix = f"{build.BENCH_NAME}/{build.BENCH_VERSION}/test/"
    assert commit["secret_sha256"] == canon.sha256_bytes((prefix + SECRET).encode())
    with pytest.raises(ValueError, match="DM_BENCH_TEST_SECRET"):
        build.write_release(root + "-x", ["easy"], DOCS, ["test"], None)
    with pytest.raises(ValueError):
        build.secret_entropy("abc123")                         # too short to be a secret


def test_puzzles_leak_nothing(release):
    root, _ = release
    for dirpath, _, files in os.walk(os.path.join(root, "puzzles")):
        if "puzzle.json" not in files:
            continue
        puzzle = json.loads(_read(dirpath, "puzzle.json"))
        assert set(puzzle) == {"schema", "page_id", "tier", "canvas", "fragments"}
        ids = [f["id"] for f in puzzle["fragments"]]
        assert ids == sorted(ids)
        assert sorted(ids) == [f"f{k:03d}" for k in range(len(ids))]     # no gaps reveal missing
        for f in puzzle["fragments"]:
            assert set(f) == {"id", "file", "w", "h"}
        pngs = sorted(x for x in files if x.endswith(".png"))
        assert pngs == sorted(f["file"] for f in puzzle["fragments"])


def test_answers_consistent_with_puzzles(release):
    root, _ = release
    adir = os.path.join(root, "answers", "hard", "train")
    for name in os.listdir(adir):
        ans = json.loads(_read(adir, name))
        puzzle = json.loads(_read(root, f"puzzles/hard/train/{ans['page_id']}/puzzle.json"))
        ids = {f["id"] for f in puzzle["fragments"]}
        assert set(ans["fragments"]) == ids
        assert len(ans["missing"]) >= 1                          # hard tier drops pieces
        assert len(ids) + len(ans["missing"]) == ans["n_pieces"]
        for a, b, shared in ans["adjacency"]:
            assert a < b
            assert {a, b} <= ids
            assert shared > 0


def test_page_ids_are_opaque_across_tiers(release):
    root, _ = release
    easy = set(os.listdir(os.path.join(root, "puzzles", "easy", "train")))
    hard = set(os.listdir(os.path.join(root, "puzzles", "hard", "train")))
    assert easy.isdisjoint(hard)


def test_checksums_detect_tampering(release, tmp_path):
    root, _ = release
    assert build.verify_checksums(root) == []
    import shutil

    copy = str(tmp_path / "copy")
    shutil.copytree(root, copy)
    victim = next(line.split("  ", 1)[1] for line in _read(copy, "SHA256SUMS").decode().splitlines()
                  if line.endswith(".png"))
    with open(os.path.join(copy, *victim.split("/")), "ab") as fh:
        fh.write(b"\0")
    assert build.verify_checksums(copy) == [victim]


def test_bad_secret_is_never_echoed():
    bad = "00112233445566778899aabbccddeeXZ-real-secret-typo"
    with pytest.raises(ValueError) as err:
        build.secret_entropy(bad)
    assert bad not in str(err.value)
    assert err.value.__context__ is None or err.value.__suppress_context__


def test_refuses_non_empty_output_dir(tmp_path):
    (tmp_path / "stale.txt").write_text("left over")
    with pytest.raises(ValueError, match="not empty"):
        build.write_release(str(tmp_path), ["easy"], DOCS, ["val"])
