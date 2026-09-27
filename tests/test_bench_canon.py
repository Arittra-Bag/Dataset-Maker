"""Determinism primitives: seed streams, exact trig, canonical JSON, digests."""
import io
import math
from decimal import Decimal, getcontext

import numpy as np
import pytest
from PIL import Image

from src.bench import canon


def test_streams_are_keyed_not_ordered():
    a = canon.stream(7, 1, 3, 0, canon.PURPOSE["tear"]).integers(0, 2**31, 5)
    canon.stream(7, 1, 2, 0, canon.PURPOSE["tear"]).integers(0, 2**31, 99)   # unrelated draw
    b = canon.stream(7, 1, 3, 0, canon.PURPOSE["tear"]).integers(0, 2**31, 5)
    assert np.array_equal(a, b)


def test_streams_differ_by_every_key_field():
    base = (7, 1, 3, 0, 3, 0)
    seen = {canon.stream_u64(*base)}
    for i in range(len(base)):
        key = list(base)
        key[i] += 1
        seen.add(canon.stream_u64(*key))
    assert len(seen) == len(base) + 1


def test_opaque_id_is_stable_hex():
    a = canon.opaque_id(7, 1, 3, 0)
    assert a == canon.opaque_id(7, 1, 3, 0)
    assert len(a) == 12
    assert int(a, 16) >= 0
    assert a != canon.opaque_id(7, 1, 3, 1)


def test_trig_is_correctly_rounded():
    getcontext().prec = 80
    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")
    rng = np.random.default_rng(0)
    for mdeg in [1, 45000, 123456, 359999, *rng.integers(0, 360000, 200).tolist()]:
        c, s = canon.cos_sin_mdeg(mdeg)
        x = Decimal(mdeg) * pi / Decimal(180000)
        # 80-digit reference via series
        rc = rs = Decimal(0)
        tc, ts = Decimal(1), x
        for k in range(80):
            rc += tc
            rs += ts
            tc *= -x * x / ((2 * k + 1) * (2 * k + 2))
            ts *= -x * x / ((2 * k + 2) * (2 * k + 3))
        assert c == float(rc) + 0.0
        assert s == float(rs) + 0.0
        assert abs(c - math.cos(float(x))) <= 2 * math.ulp(1.0)


def test_trig_exact_at_right_angles():
    assert canon.cos_sin_mdeg(0) == (1.0, 0.0)
    assert canon.cos_sin_mdeg(90000) == (0.0, 1.0)
    assert canon.cos_sin_mdeg(180000) == (-1.0, 0.0)
    assert canon.cos_sin_mdeg(-90000) == (0.0, -1.0)


def test_canonical_json():
    text = canon.dumps({"b": [1.0000000001, -0.0], "a": {"z": 1, "y": np.float64(0.5)}})
    assert text == '{"a":{"y":0.5,"z":1},"b":[1.0,0.0]}\n'
    with pytest.raises(ValueError):
        canon.dumps({"x": float("nan")})
    with pytest.raises(TypeError):
        canon.dumps({1: "int keys reorder after a round trip"})
    with pytest.raises(ValueError):
        canon.loads('{"a": NaN}')


def _png(rgba, level):
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=level)
    return buf.getvalue()


def test_png_content_digest_ignores_encoder_and_hidden_rgb():
    rgba = np.zeros((20, 30, 4), np.uint8)
    rgba[5:15, 5:25] = (200, 100, 50, 255)
    hidden = rgba.copy()
    hidden[0, 0, :3] = (9, 9, 9)                    # RGB under alpha == 0
    a, b = _png(rgba, 1), _png(hidden, 9)
    assert a != b
    assert canon.content_digest_png(a) == canon.content_digest_png(b)
    changed = rgba.copy()
    changed[10, 10, 0] = 201
    assert canon.content_digest_png(_png(changed, 1)) != canon.content_digest_png(a)


def test_json_content_digest_ignores_formatting():
    assert canon.content_digest_json(b'{"a": 1, "b": 2}') == canon.content_digest_json(b'{"b":2,"a":1}')


def test_numpy_bool_and_overflow():
    assert canon.dumps({"ok": np.bool_(True)}) == '{"ok":true}\n'
    with pytest.raises(ValueError):
        canon.loads('{"x": 1e400}')
