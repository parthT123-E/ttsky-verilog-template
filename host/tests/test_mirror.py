"""Check the mirror reflection maths and, crucially, that the bounce loop
TERMINATES -- including the pathological case of a ray trapped between two
parallel mirrors, which is the one way this can hang forever.
"""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf

print(f"mirror materials: { {m: r for m, r in rf.MIRROR.items()} }")
print(f"bounce cap {rf.MIRROR_BOUNCES}, "
      f"throughput floor {rf.MIRROR_MIN_THROUGHPUT:.5f}")

# --- the right wall really is the mirror -------------------------------
material = rf.make_scene()
G = rf.GRID
EW = rf.EDGE_WIDTH
# Exclude the matte trim along every edge the right wall shares with the
# front/back/floor/roof mirrors (see make_scene's `edge_zy`), not just the
# back-wall's own corner column.
right = material[EW:G - EW, EW:G - EW, G - 1]
assert np.all(right == rf.MAT_MIRROR), \
    f"right wall is not the mirror: {np.unique(right)}"
assert rf.REFLECTIVITY[rf.MAT_MIRROR] > 0, "mirror has no reflectivity"
print(f"\nright wall = material {rf.MAT_MIRROR} (mirror)")

# --- reflection law: angle in equals angle out, tangent preserved ------
for face, axis in ((0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)):
    d = np.array([[0.3, -0.5, 0.8]])
    d = d / np.linalg.norm(d)
    r = rf.reflect(d, np.array([face], dtype=np.uint8))[0]
    expect = d[0].copy()
    expect[axis] = -expect[axis]      # only the face-axis component flips
    assert np.allclose(r, expect, atol=1e-12), (face, r, expect)
    assert abs(np.linalg.norm(r) - 1.0) < 1e-12, "reflection changed length"
print("reflect(): flips only the face-axis component, preserves length")

# A ray straight at a wall must come straight back.
d = np.array([[1.0, 0.0, 0.0]])
r = rf.reflect(d, np.array([1], dtype=np.uint8))[0]   # face 1 -> +x normal
assert np.allclose(r, (-1, 0, 0)), r
print("head-on incidence reverses exactly")

# --- termination: throughput decay ------------------------------------
refl = rf.REFLECTIVITY[rf.MAT_MIRROR]
tint = np.array(rf.PALETTE[rf.MAT_MIRROR])
t = np.ones(3)
n = 0
while t.max() > rf.MIRROR_MIN_THROUGHPUT and n < 10000:
    t = t * refl * tint
    n += 1
print(f"\nthroughput falls below one 8-bit step after {n} bounces "
      f"(reflectivity {refl}, tint {tuple(tint)})")
assert n < 10000, "throughput never decays -- reflectivity must be < 1"
print(f"the cap of {rf.MIRROR_BOUNCES} bites first, so the loop is bounded "
      f"either way")

# --- termination: two parallel mirrors ---------------------------------
# Worst case for this scene shape: a ray bouncing between the side walls.
# It still carries +z from the camera, so it advances toward the back wall
# instead of looping -- but the cap is what actually guarantees an end.
d = np.array([[-0.6, 0.0, 0.8]])
d = d / np.linalg.norm(d)
z_gain = []
cur = d.copy()
for b in range(rf.MIRROR_BOUNCES):
    cur = rf.reflect(cur, np.array([1 if cur[0, 0] > 0 else 0],
                                   dtype=np.uint8))
    z_gain.append(float(cur[0, 2]))
assert all(z > 0 for z in z_gain), \
    "a side-wall bounce lost its forward component -- it could loop forever"
print(f"\nside-to-side bounces keep z = {z_gain[0]:.2f} each time, so the ray "
      f"still advances toward the back wall")

# A ray with NO forward component is the true trap; only the cap saves it.
flat = np.array([[1.0, 0.0, 0.0]])
cur = flat.copy()
seen = set()
looped = False
for b in range(rf.MIRROR_BOUNCES):
    cur = rf.reflect(cur, np.array([1 if cur[0, 0] > 0 else 0],
                                   dtype=np.uint8))
    key = tuple(np.round(cur[0], 6))
    if key in seen:
        looped = True
    seen.add(key)
assert looped, "expected a purely axial ray to cycle"
print("a purely axial ray does cycle -- bounded only by MIRROR_BOUNCES, "
      "which is why the cap is not optional")

print("\nALL MIRROR CHECKS PASSED")
