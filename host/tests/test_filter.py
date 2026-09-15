"""Unit-test the plane-aware visibility filter on synthetic surfaces.

The whole design rests on one distinction: coplanar neighbours must be
averaged (that smooths a wall's penumbra) while neighbours that step ALONG
the normal must not (that preserves a sphere's per-voxel terraces). Both
cases look identical to a plain distance test, so they are checked directly.
"""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf

H = W = 64
FACE = 4                     # stepped +z, normal (0, 0, -1)
R = rf.SHADOW_FILTER_RADIUS
TOL = rf.SHADOW_FILTER_PLANE_TOL


def build(step_across_boundary):
    """A half-black/half-white visibility field. If step_across_boundary,
    the right half also sits one voxel further along the normal (a terrace)."""
    frac = np.zeros((H, W, 3))
    frac[:, W // 2:] = 1.0
    valid = np.ones((H, W), dtype=bool)
    face = np.full((H, W), FACE, dtype=np.uint8)
    pos = np.zeros((H, W, 3), dtype=np.int32)
    pos[:, :, 2] = 10
    if step_across_boundary:
        pos[:, W // 2:, 2] = 11
    return frac, valid, face, pos


def edge_profile(a):
    """Values either side of the boundary, along the middle row."""
    row = a[H // 2, :, 0]
    return row[W // 2 - 1], row[W // 2]


print(f"filter radius {R}px, plane tolerance {TOL} voxels\n")

# --- coplanar wall: the step MUST blur into a ramp ----------------------
frac, valid, face, pos = build(step_across_boundary=False)
out = rf.filter_visibility(frac, valid, face, pos, R, TOL)
lo, hi = edge_profile(out)
print(f"coplanar wall   : {lo:.3f} | {hi:.3f}  across the boundary")
assert 0.05 < lo < 0.95 and 0.05 < hi < 0.95, \
    "coplanar neighbours were not averaged -- the penumbra will stay blocky"
# and far from the boundary it must still reach the original values
assert out[H // 2, 0, 0] < 0.02 and out[H // 2, -1, 0] > 0.98, \
    "filter bled all the way across the image"
print("  -> smoothed, as a wall's penumbra should be")

# --- terrace step: the step MUST survive --------------------------------
frac, valid, face, pos = build(step_across_boundary=True)
out = rf.filter_visibility(frac, valid, face, pos, R, TOL)
lo, hi = edge_profile(out)
print(f"terrace step    : {lo:.3f} | {hi:.3f}  across the boundary")
assert lo < 0.02 and hi > 0.98, \
    "the filter averaged across a terrace -- per-voxel detail is lost"
print("  -> preserved, as a sphere's voxel detail should be")

# --- different faces must never mix -------------------------------------
frac, valid, face, pos = build(step_across_boundary=False)
face[:, W // 2:] = 0                      # a different surface orientation
out = rf.filter_visibility(frac, valid, face, pos, R, TOL)
lo, hi = edge_profile(out)
print(f"face change     : {lo:.3f} | {hi:.3f}  across the boundary")
assert lo < 0.02 and hi > 0.98, "the filter mixed across differing faces"
print("  -> preserved, so shadows cannot bleed across a silhouette")

# --- the filter must conserve the mean ----------------------------------
frac, valid, face, pos = build(step_across_boundary=False)
out = rf.filter_visibility(frac, valid, face, pos, R, TOL)
assert abs(out.mean() - frac.mean()) < 1e-6, "filter changed total visibility"
assert out.min() >= -1e-9 and out.max() <= 1 + 1e-9, "left the 0..1 range"
print(f"\nmean preserved  : {frac.mean():.6f} -> {out.mean():.6f}")
print("ALL FILTER UNIT TESTS PASSED")
