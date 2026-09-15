"""Why is the shadow noise blotchy rather than fine grain?

Hypothesis: the shadow ray ORIGIN is quantised to voxel centres, so every
pixel landing on the same voxel fires from an identical point. At 1024x1024
over a 128^3 grid a voxel is ~8 pixels wide, so that produces voxel-sized
BLOCKS of correlated error -- which a 5x5 filter cannot touch, because the
structure is larger than the kernel and is systematic, not random.

Measures the spatial autocorrelation of the visibility term to see how far
the correlation actually extends.
"""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 128


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))
    emissive = np.zeros(len(rf.PALETTE), bool)
    emissive[list(rf.EMISSIVE)] = True

    eye, dirs = rf.camera_rays(W, H)
    rays = rf.dda_init_batch(eye, dirs)
    live = np.nonzero(rays["valid"])[0]
    sub = {k: rays[k][live] for k in ("voxel", "signs", "next", "inc")}

    total = W * H
    r_hit = np.zeros(total, bool)
    r_face = np.full(total, 7, np.uint8)
    r_pos = np.zeros((total, 3), np.int32)
    for i, pixel in enumerate(live):
        res = hw_model_trace(occupied, rf.dda_row(sub, i), rf.MAX_STEPS)
        r_hit[pixel] = res["hit"]
        r_face[pixel] = res["face"]
        r_pos[pixel] = (res["x"], res["y"], res["z"])

    mats = np.zeros(total, np.uint8)
    mats[r_hit] = material[r_pos[r_hit, 2], r_pos[r_hit, 1], r_pos[r_hit, 0]]
    shaded = r_hit & ~emissive[mats]
    sh = np.nonzero(shaded)[0]

    px_per_voxel = W / (2 * 0.22 * rf.GRID)   # rough: sphere width in pixels
    print(f"{W}x{H} render of a {rf.GRID}^3 grid")

    # How many distinct shadow-ray ORIGINS do neighbouring pixels share?
    origins = r_pos[sh]
    uniq = np.unique(origins, axis=0).shape[0]
    print(f"shaded pixels {sh.size}, distinct hit voxels {uniq} "
        f"-> {sh.size / uniq:.1f} pixels share each origin")

    # Visibility for one light stratum, per pixel.
    rng = np.random.default_rng(rf.JITTER_SEED)
    stratum = rf.LIGHT_STRATA[0]
    lpos = rf.jitter_strata(stratum, sh.size, rng)
    normals = rf.FACE_NORMAL_ARRAY[r_face[sh]]
    pts = r_pos[sh].astype(float) + 0.5
    d = lpos - pts
    dist = np.sqrt((d * d).sum(axis=1))
    cos = (normals * d).sum(axis=1) / np.maximum(dist, 1e-9)
    face_on = cos > 1e-9
    rows = sh[face_on]

    sdda, ssteps = rf.shadow_jobs(r_pos[rows], r_face[rows], lpos[face_on])
    vis = np.zeros(rows.size)
    for k in range(rows.size):
        res = hw_model_trace(occupied, rf.dda_row(sdda, k), int(ssteps[k]))
        if not res["hit"]:
            vis[k] = 1.0
        else:
            m = int(material[res["z"]][res["y"]][res["x"]])
            vis[k] = 1.0 if emissive[m] else 0.0

    grid = np.full(total, np.nan)
    grid[rows] = vis
    g = grid.reshape(H, W)

    # Autocorrelation of the visibility residual along a row: pure per-pixel
    # noise decorrelates after 1 pixel; blocky error stays correlated.
    print("\nlag :  correlation of visibility between pixels that distance apart")
    valid = ~np.isnan(g)
    for lag in (1, 2, 3, 4, 6, 8, 12, 16):
        a, b = g[:, :-lag], g[:, lag:]
        m = valid[:, :-lag] & valid[:, lag:]
        if m.sum() < 100:
            continue
        av, bv = a[m], b[m]
        c = np.corrcoef(av, bv)[0, 1]
        print(f"{lag:3d} : {c:+.3f}")
    print("\n(a filter of radius R can only remove structure smaller than ~R)")


if __name__ == "__main__":
    main()
