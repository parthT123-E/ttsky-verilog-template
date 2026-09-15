"""Is the seam an RTL/fixed-point artifact, or true geometry?

For each seam pixel, compute the entry face ANALYTICALLY from the ray and the
hit voxel (exact floats, no DDA, no fixed point) and compare against what the
bit-exact RTL model reports. Also measure how close the competing slab-entry
times are -- a near-tie means Q16.16 rounding decides the outcome.

Also locates the face_id == 6 population found by diag_seams.py.
"""
import math
import statistics
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 512
# Window around the big red sphere where diag_seams.py found the seams.
X0, X1, Y0, Y1 = 180, 330, 140, 270


def analytic_entry_face(eye, d, vox):
    """Face of `vox` the ray enters through, from exact ray geometry.
    Returns (face_id, entry_t, gap_to_runner_up) using the hardware's
    face encoding: axis*2 + (0 if stepping positive else 1)."""
    lo = vox
    hi = (vox[0] + 1, vox[1] + 1, vox[2] + 1)
    ts = []
    for a in range(3):
        if abs(d[a]) < 1e-12:
            continue
        # Entering: cross the low face when moving +, the high face when -.
        t = ((lo[a] if d[a] > 0 else hi[a]) - eye[a]) / d[a]
        ts.append((t, a))
    ts.sort(reverse=True)          # entry t = max over axes
    t_best, axis = ts[0]
    gap = t_best - ts[1][0] if len(ts) > 1 else float("inf")
    return axis * 2 + (0 if d[axis] > 0 else 1), t_best, gap


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))

    rows = range(Y0, Y1)
    cols = range(X0, X1)
    face, vox, raw, hit = {}, {}, {}, {}
    for py in rows:
        for px in cols:
            eye, dd = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, dd)
            if dda is None:
                continue
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            if not r["hit"]:
                continue
            hit[(px, py)] = True
            face[(px, py)] = r["face"]
            vox[(px, py)] = (r["x"], r["y"], r["z"])
            raw[(px, py)] = rf.raw_diffuse(r)

    # Seam detection, same rule as diag_seams.py
    seams = []
    for py in rows:
        for px in cols:
            if (px, py) not in hit:
                continue
            nb = [raw[(px + dx, py + dy)]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dx or dy) and (px + dx, py + dy) in hit]
            if len(nb) < 6:
                continue
            med = statistics.median(nb)
            if med > 0.05 and raw[(px, py)] < 0.6 * med:
                seams.append((px, py))

    print(f"seam pixels in window: {len(seams)}\n")

    agree = disagree = 0
    gaps = []
    print("px,py   model_face  analytic_face  entry-t gap (voxel units)  N.L")
    for px, py in seams:
        eye, dd = rf.camera_ray(px, py, W, H)
        d = unit(dd)
        af, t, gap = analytic_entry_face(eye, d, vox[(px, py)])
        mf = face[(px, py)]
        gaps.append(gap)
        if af == mf:
            agree += 1
        else:
            disagree += 1
        n = rf.FACE_NORMAL[mf]
        hp = [vox[(px, py)][i] + 0.5 for i in range(3)]
        tl = unit([rf.LIGHT_POS[i] - hp[i] for i in range(3)])
        ndl = sum(n[i] * tl[i] for i in range(3))
        if len(gaps) <= 10:
            print(f"{px:4d},{py:3d}   f{mf}          f{af}"
                  f"             {gap:.6f}            {ndl:+.3f}")

    print(f"\nanalytic face AGREES with model on {agree}/{len(seams)} seams, "
          f"disagrees on {disagree}")
    if gaps:
        print(f"entry-t gap to runner-up axis: min={min(gaps):.2e} "
              f"median={statistics.median(gaps):.2e} max={max(gaps):.2e}")
        print("(Q16.16 timer resolution is 1/65536 = 1.53e-05 voxel units)")

    # Where are the face==6 pixels? Sample the whole image cheaply by rows.
    print("\n--- face_id == 6 (hit own starting voxel) population ---")
    f6 = []
    for py in range(0, H, 4):
        for px in range(0, W, 4):
            eye, dd = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, dd)
            if dda is None:
                continue
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            if r["hit"] and r["face"] == 6:
                f6.append((px, py, (r["x"], r["y"], r["z"]),
                           material[r["z"]][r["y"]][r["x"]]))
    if f6:
        xs = [p[0] for p in f6]
        ys = [p[1] for p in f6]
        mats = {}
        for p in f6:
            mats[p[3]] = mats.get(p[3], 0) + 1
        print(f"count (1/16 sample): {len(f6)}")
        print(f"px range {min(xs)}..{max(xs)}, py range {min(ys)}..{max(ys)}")
        print(f"materials: {mats}")
        print(f"sample voxels: {[p[2] for p in f6[:6]]}")
    else:
        print("none in sample")


if __name__ == "__main__":
    main()
