"""End-to-end render with mirrors, traced entirely through the RTL model.

Runs the same bounce loop main() does, so the reflection bookkeeping (per-ray
origins, throughput) is exercised before any of it reaches hardware. Shadows
are left on for the terminal hits.
"""
import os
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def trace_model(occupied, dda, steps, n):
    hit = np.zeros(n, bool)
    face = np.full(n, 7, np.uint8)
    pos = np.zeros((n, 3), np.int32)
    for k in range(n):
        s = int(steps[k]) if np.ndim(steps) else int(steps)
        r = hw_model_trace(occupied, rf.dda_row(dda, k), s)
        hit[k] = r["hit"]
        face[k] = r["face"]
        pos[k] = (r["x"], r["y"], r["z"])
    return hit, face, pos


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))
    palette = np.array(rf.PALETTE)
    emissive = np.zeros(len(rf.PALETTE), bool)
    emissive[list(rf.EMISSIVE)] = True

    total = W * H
    eye, dirs = rf.camera_rays(W, H)
    rays = rf.dda_init_batch(eye, dirs)
    live = np.nonzero(rays["valid"])[0]
    sub = {k: rays[k][live] for k in ("voxel", "signs", "next", "inc")}

    r_hit = np.zeros(total, bool)
    r_face = np.full(total, 7, np.uint8)
    r_pos = np.zeros((total, 3), np.int32)
    h, f, p = trace_model(occupied, sub, rf.MAX_STEPS, live.size)
    r_hit[live], r_face[live], r_pos[live] = h, f, p
    print(f"primary: {live.size} rays, {r_hit.sum()} hits")

    # ---- bounce loop, mirroring main() -----------------------------------
    cur_org = np.broadcast_to(np.asarray(eye, float), (total, 3)).copy()
    cur_dir = dirs.copy()
    throughput = np.ones((total, 3))
    counts = []
    for b in range(rf.MIRROR_BOUNCES):
        mats_now = np.zeros(total, np.uint8)
        mats_now[r_hit] = material[r_pos[r_hit, 2], r_pos[r_hit, 1],
                                   r_pos[r_hit, 0]]
        alive = (r_hit & (rf.REFLECTIVITY[mats_now] > 0)
                 & (throughput.max(axis=1) > rf.MIRROR_MIN_THROUGHPUT))
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break
        counts.append(idx.size)
        face_i = r_face[idx]
        hp = rf.ray_hit_points(cur_org[idx], cur_dir[idx], r_pos[idx], face_i)
        refl = rf.reflect(cur_dir[idx], face_i)
        org = rf.shadow_origins(hp, face_i)
        bdda = rf.dda_from_points(org, refl)
        bh, bf, bp = trace_model(occupied, bdda, rf.MAX_STEPS, idx.size)
        throughput[idx] *= (rf.REFLECTIVITY[mats_now[idx]][:, None]
                            * palette[mats_now[idx]])
        cur_org[idx], cur_dir[idx] = org, refl
        r_hit[idx], r_face[idx], r_pos[idx] = bh, bf, bp
    print(f"bounces: {counts}")
    assert counts, "no ray ever hit the mirror"
    assert len(counts) < rf.MIRROR_BOUNCES or counts[-1] < counts[0], \
        "the reflected set never shrinks -- rays may be trapped"

    # Reflected pixels must be dimmed, and never brightened.
    refl_px = throughput.max(axis=1) < 1.0
    print(f"reflected pixels: {refl_px.sum()} "
          f"({100 * refl_px.mean():.1f}% of frame)")
    assert throughput.max() <= 1.0 + 1e-9, "a bounce ADDED energy"
    assert refl_px.sum() > 0

    # ---- shade the terminal hits ----------------------------------------
    mats = np.zeros(total, np.uint8)
    mats[r_hit] = material[r_pos[r_hit, 2], r_pos[r_hit, 1], r_pos[r_hit, 0]]
    lamp = r_hit & emissive[mats]
    shaded = r_hit & ~lamp
    sh_idx = np.nonzero(shaded)[0]

    normals = rf.FACE_NORMAL_ARRAY[r_face[sh_idx]]
    hit_pt = rf.ray_hit_points(cur_org[sh_idx], cur_dir[sh_idx],
                               r_pos[sh_idx], r_face[sh_idx])
    origins = rf.shadow_origins(hit_pt, r_face[sh_idx])

    illum = np.zeros((total, 3))
    rng = np.random.default_rng(rf.JITTER_SEED)
    for stratum in rf.LIGHT_STRATA:
        lpos = rf.jitter_strata(stratum, sh_idx.size, rng)
        cr, cg, cb = stratum["colour"]
        d = lpos - hit_pt
        dist = np.sqrt((d * d).sum(axis=1))
        cos = (normals * d).sum(axis=1) / np.maximum(dist, 1e-9)
        m = cos > 1e-9
        if not m.any():
            continue
        rows = sh_idx[m]
        w = cos[m]
        sdda, ssteps = rf.shadow_jobs(origins[m], lpos[m])
        sh, _, sp = trace_model(occupied, sdda, ssteps, rows.size)
        smat = np.zeros(rows.size, np.uint8)
        smat[sh] = material[sp[sh, 2], sp[sh, 1], sp[sh, 0]]
        w = w * rf.shadow_visible(sh, smat, emissive)
        illum[rows, 0] += cr * w
        illum[rows, 1] += cg * w
        illum[rows, 2] += cb * w

    illum *= throughput
    image = np.zeros((total, 3), np.uint8)
    image[lamp] = np.clip(np.rint(palette[mats[lamp]] * throughput[lamp] * 255),
                          0, 255).astype(np.uint8)
    albedo = np.zeros((total, 3))
    albedo[shaded] = palette[mats[shaded]]
    rf.normalize_image(illum, albedo, shaded, image)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "render_mirror.png")
    rf.write_png(out, image, W, H)
    print(f"\nwrote {out}")
    print("ALL MIRROR RENDER CHECKS PASSED")


if __name__ == "__main__":
    main()
