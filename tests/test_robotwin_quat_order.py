"""
Unit test for the RoboTwin quaternion-order handling (issue #93).

RoboTwin stores quaternions scalar-first (wxyz, transforms3d convention);
SciPy's Rotation uses scalar-last (xyzw). This test checks that

  * with ``scalar_first=True`` the rotation-6D built by the RoboTwin eval client
    matches the rotation matrix transforms3d would produce from the same wxyz
    quaternion, and that the rot6d -> quat round trip returns wxyz;
  * with the default ``scalar_first=False`` the legacy behaviour is unchanged;
  * ``datasets.utils.quat_to_rotate6d(..., scalar_first=True)`` agrees with the
    client (skipped if the training-side dependencies are not installed).

Only numpy / scipy are required.  Run with:  python -m pytest tests/
"""
import ast
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT = os.path.join(ROOT, "evaluation", "robotwin-2.0", "client.py")


def _load_client_funcs():
    """Load only the pure conversion helpers from client.py.

    Importing the module directly would chdir into a RoboTwin checkout and
    import simulator dependencies, so we exec just the two function bodies.
    """
    with open(CLIENT, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = {"quat_to_rotate6D", "rotate6D_to_quat"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    ns = {"np": np, "R": R}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), CLIENT, "exec"), ns)
    return ns["quat_to_rotate6D"], ns["rotate6D_to_quat"]


def quat2mat_wxyz(q):
    """Reference implementation of transforms3d.quaternions.quat2mat (wxyz)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )


def _random_wxyz(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def _mat_from_rot6d(v6):
    return v6.reshape(v6.shape[:-1] + (3, 2))


def test_scalar_first_matches_transforms3d_reference():
    quat_to_rotate6D, _ = _load_client_funcs()
    q_wxyz = _random_wxyz(64)
    r6 = quat_to_rotate6D(q_wxyz, scalar_first=True)
    assert r6.shape == (64, 6)
    ref = np.stack([quat2mat_wxyz(q)[:, :2] for q in q_wxyz])
    np.testing.assert_allclose(_mat_from_rot6d(r6), ref, atol=1e-12)


def test_legacy_default_is_unchanged():
    quat_to_rotate6D, rotate6D_to_quat = _load_client_funcs()
    q = _random_wxyz(32, seed=1)
    legacy = R.from_quat(q).as_matrix()[..., :, :2].reshape(32, 6)
    np.testing.assert_allclose(quat_to_rotate6D(q), legacy, atol=1e-12)
    # default output is SciPy xyzw, as before
    back = rotate6D_to_quat(legacy)
    np.testing.assert_allclose(np.abs(np.sum(back * q, axis=-1)), 1.0, atol=1e-10)


def test_round_trip_scalar_first():
    quat_to_rotate6D, rotate6D_to_quat = _load_client_funcs()
    q_wxyz = _random_wxyz(64, seed=2)
    back = rotate6D_to_quat(quat_to_rotate6D(q_wxyz, scalar_first=True), scalar_first=True)
    # q and -q are the same rotation; compare up to sign
    np.testing.assert_allclose(np.abs(np.sum(back * q_wxyz, axis=-1)), 1.0, atol=1e-10)
    # and the recovered quaternion must still be wxyz (scalar in slot 0)
    np.testing.assert_allclose(
        np.stack([quat2mat_wxyz(q) for q in back]),
        np.stack([quat2mat_wxyz(q) for q in q_wxyz]),
        atol=1e-10,
    )


def test_legacy_and_fixed_differ_for_generic_rotation():
    # Sanity check that the flag actually changes the representation.
    quat_to_rotate6D, _ = _load_client_funcs()
    q_wxyz = _random_wxyz(8, seed=3)
    assert not np.allclose(quat_to_rotate6D(q_wxyz), quat_to_rotate6D(q_wxyz, scalar_first=True))


def test_dataset_utils_agrees_with_client():
    try:
        sys.path.insert(0, ROOT)
        from datasets.utils import quat_to_rotate6d, rotate6d_to_quat
    except Exception as e:  # torch / mmengine / av etc. not installed
        import pytest
        pytest.skip(f"datasets.utils not importable here: {e!r}")
    quat_to_rotate6D, rotate6D_to_quat = _load_client_funcs()
    q_wxyz = _random_wxyz(32, seed=4)
    for sf in (False, True):
        np.testing.assert_allclose(
            quat_to_rotate6d(q_wxyz, scalar_first=sf), quat_to_rotate6D(q_wxyz, scalar_first=sf), atol=1e-12
        )
        r6 = quat_to_rotate6D(q_wxyz, scalar_first=sf)
        np.testing.assert_allclose(
            rotate6d_to_quat(r6, scalar_first=sf), rotate6D_to_quat(r6, scalar_first=sf), atol=1e-10
        )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
