"""Determinism primitives for the benchmark.

Everything a release contains must be a pure function of its key and must
not depend on the host: no Python hash(), no call-order dependent RNG
spawning, no platform libm trig in stored numbers, no float formatting
drift in JSON. This module owns those rules.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from decimal import Decimal, localcontext

import numpy as np

# ---------------------------------------------------------------------------
# Seed streams
# ---------------------------------------------------------------------------
# Fixed integer codes. Never renumber: every stream in a released version is
# keyed by them.
TIERS = {"easy": 1, "medium": 2, "hard": 3}
PURPOSE = {
    "layout": 1,      # synthetic document content
    "npieces": 2,     # fragment count draw
    "tear": 3,        # tear_page seed
    "idperm": 4,      # opaque fragment ids
    "rotation": 5,
    "pad": 6,         # random margin after tight crop
    "blur": 7,
    "noise": 8,
    "jpeg": 9,
    "missing": 10,
    "erosion": 11,
    "pageid": 12,     # opaque page id
}


def stream(entropy: int, tier: int, doc: int, page: int, purpose: int,
           attempt: int = 0) -> np.random.Generator:
    """Independent RNG for one (tier, doc, page, purpose, attempt) key.

    Keyed SeedSequence, never .spawn(): a page can be rebuilt alone and adding
    a purpose later does not reshuffle existing streams.
    """
    ss = np.random.SeedSequence(entropy, spawn_key=(tier, doc, page, purpose, attempt))
    return np.random.default_rng(ss)


def stream_u64(entropy: int, tier: int, doc: int, page: int, purpose: int,
               attempt: int = 0) -> int:
    """64-bit integer seed for code that takes an int (e.g. tear_page)."""
    ss = np.random.SeedSequence(entropy, spawn_key=(tier, doc, page, purpose, attempt))
    return int(ss.generate_state(1, np.uint64)[0])


def opaque_id(entropy: int, tier: int, doc: int, page: int, nbytes: int = 6) -> str:
    """Opaque hex id that reveals nothing about tier/doc/page."""
    state = np.random.SeedSequence(
        entropy, spawn_key=(tier, doc, page, PURPOSE["pageid"], 0)
    ).generate_state(2, np.uint64)
    return state.tobytes().hex()[: 2 * nbytes]


# ---------------------------------------------------------------------------
# Exact trig
# ---------------------------------------------------------------------------
_TRIG_CACHE: dict[int, tuple[float, float]] = {}


def _decimal_pi() -> Decimal:
    # 50 digits is far beyond float64 needs; literal avoids computing it.
    return Decimal("3.14159265358979323846264338327950288419716939937511")


def cos_sin_mdeg(mdeg: int) -> tuple[float, float]:
    """Correctly rounded (cos, sin) of `mdeg` millidegrees.

    Host libm cos/sin are not correctly rounded everywhere (Apple libm misses
    on ~4% of angles), so stored affines would differ by platform. Taylor
    series in 40-digit Decimal, then a single correctly rounded float().
    """
    mdeg = int(mdeg) % 360000
    if mdeg in _TRIG_CACHE:
        return _TRIG_CACHE[mdeg]
    if mdeg % 90000 == 0:                       # exact at right angles
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[mdeg // 90000]
    with localcontext() as ctx:                 # never touch the thread's context
        ctx.prec = 40
        x = Decimal(mdeg) * _decimal_pi() / Decimal(180000)
        c = s = Decimal(0)
        term_c, term_s = Decimal(1), x
        for k in range(60):
            c += term_c
            s += term_s
            term_c *= -x * x / ((2 * k + 1) * (2 * k + 2))
            term_s *= -x * x / ((2 * k + 2) * (2 * k + 3))
    out = (float(c) + 0.0, float(s) + 0.0)      # + 0.0 normalises -0.0
    _TRIG_CACHE[mdeg] = out
    return out


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------
FLOAT_DECIMALS = 9


def _canon(obj):
    if isinstance(obj, float):
        v = round(obj, FLOAT_DECIMALS) + 0.0
        if not math.isfinite(v):
            raise ValueError("non-finite float in canonical JSON")
        return v
    if isinstance(obj, (np.floating,)):
        return _canon(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("canonical JSON keys must be str")
        return {k: _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    return obj


def dumps(obj) -> str:
    """One canonical serialisation: sorted keys, fixed separators, ASCII."""
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n"


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _finite_float(token: str) -> float:
    value = float(token)
    if value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite JSON number {token}")      # e.g. 1e400 overflow
    return value


def loads(text: str):
    """Strict JSON: NaN/Infinity tokens and overflowing numbers are rejected."""
    return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_digest_png(data: bytes) -> str:
    """Digest of decoded pixels, independent of PNG encoder/zlib version.

    RGB under alpha == 0 is invisible, so it is zeroed before hashing.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        rgba = np.array(im.convert("RGBA"))
    rgba[rgba[..., 3] == 0, :3] = 0
    h, w = rgba.shape[:2]
    header = f"RGBA:{w}x{h}:".encode()
    return sha256_bytes(header + rgba.tobytes())


def content_digest_json(data: bytes) -> str:
    return sha256_bytes(dumps(loads(data.decode("utf-8"))).encode("ascii"))
