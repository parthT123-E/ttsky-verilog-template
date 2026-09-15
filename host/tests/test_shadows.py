"""Verify the shadow pass offline, tracing every shadow ray through the
bit-exact RTL model instead of hardware.

Checks the mechanism itself (no self-shadowing, lamps reachable, occluders
actually block) and then renders the scene with shadows so the result can be
eyeballed before any bitstream is involved.
"""
import os
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = int(sys.argv[1]) if len(sys.argv) > 1 else 96


def model_trace(occupied, dda_row_dict, max_steps):
    return hw_model_trace(occupied, dda_row_dict, max_steps)


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))
    emissive = np.zeros(len(rf.PALETTE), dtype=bool)
    emissive[list(rf.EMISSIVE)] = True

    # ---- primary pass through the model ---------------------------------
    eye, dirs = rf.camera_rays(W, H)
    rays = rf.dda_init_batch(eye, dirs)
    live = np.nonzero(rays["valid"])[0]
    sub = {k: rays[k][live] for k in ("voxel", "signs", "next", "inc")}

    total = W * H
    r_hit = np.zeros(total, bool)
    r_face = np.full(total, 7, np.uint8)
    r_pos = np.zeros((total, 3), np.int32)
    for i, pixel in enumerate(live):
        res = model_trace(occupied, rf.dda_row(sub, i), rf.MAX_STEPS)
        r_hit[pixel] = res["hit"]
        r_face[pixel] = res["face"]
        r_pos[pixel] = (res["x"], res["y"], res["z"])

    mats = np.zeros(total, np.uint8)
    mats[r_hit] = material[r_pos[r_hit, 2], r_pos[r_hit, 1], r_pos[r_hit, 0]]
    lamp = r_hit & emissive[mats]
    shaded = r_hit & ~lamp
    sh_idx = np.nonzero(shaded)[0]
    print(f"{live.size} primary rays -> {shaded.sum()} shaded surfaces")

    # ---- shadow pass -----------------------------------------------------
    normals = rf.FACE_NORMAL_ARRAY[r_face[sh_idx]]
    hit_pt = rf.ray_hit_points(eye, dirs[sh_idx], r_pos[sh_idx],
                               r_face[sh_idx])
    origins = rf.shadow_origins(hit_pt, r_face[sh_idx])
    pts = hit_pt

    # The exact hit point must lie inside (or on the face of) its own voxel.
    off = hit_pt - r_pos[sh_idx]
    assert off.min() >= -1e-6 and off.max() <= 1.0 + 1e-6, \
        f"hit points escaped their voxel: {off.min():.3f}..{off.max():.3f}"
    # And each pixel must now get its OWN origin, not its voxel's centre.
    uniq = np.unique(np.round(origins, 4), axis=0).shape[0]
    print(f"distinct shadow origins: {uniq} for {sh_idx.size} pixels "
          f"({uniq / sh_idx.size:.1%})")
    assert uniq > 0.9 * sh_idx.size, \
        "pixels are still sharing shadow-ray origins"
    illum = np.zeros((total, 3))
    illum_noshadow = np.zeros((total, 3))

    self_shadow = 0
    reached_lamp = 0
    blocked = 0
    timed_out = 0
    n_rays = 0

    rng = np.random.default_rng(rf.JITTER_SEED)
    for stratum in rf.LIGHT_STRATA:
        lpos = rf.jitter_strata(stratum, sh_idx.size, rng)
        cr, cg, cb = stratum["colour"]
        dx = lpos[:, 0] - pts[:, 0]
        dy = lpos[:, 1] - pts[:, 1]
        dz = lpos[:, 2] - pts[:, 2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        cos_theta = ((normals[:, 0] * dx + normals[:, 1] * dy
                      + normals[:, 2] * dz) / np.maximum(dist, 1e-9))
        face_on = cos_theta > 1e-9
        if not face_on.any():
            continue
        rows = sh_idx[face_on]
        weight = cos_theta[face_on]

        illum_noshadow[rows, 0] += cr * weight
        illum_noshadow[rows, 1] += cg * weight
        illum_noshadow[rows, 2] += cb * weight

        sdda, ssteps = rf.shadow_jobs(origins[face_on], lpos[face_on])
        vis = np.zeros(rows.size, bool)
        for k in range(rows.size):
            res = model_trace(occupied, rf.dda_row(sdda, k), int(ssteps[k]))
            n_rays += 1
            if not res["hit"]:
                vis[k] = True
                timed_out += 1
                continue
            m = int(material[res["z"]][res["y"]][res["x"]])
            if emissive[m]:
                vis[k] = True
                reached_lamp += 1
            else:
                blocked += 1
                # A shadow ray that terminates on its OWN surface voxel means
                # the one-voxel normal offset failed.
                if (res["x"], res["y"], res["z"]) == tuple(r_pos[rows[k]]):
                    self_shadow += 1
        w = weight * vis
        illum[rows, 0] += cr * w
        illum[rows, 1] += cg * w
        illum[rows, 2] += cb * w

    print(f"shadow rays: {n_rays}")
    print(f"  reached a lamp's geometry : {reached_lamp}")
    print(f"  ran the full distance     : {timed_out}  (invisible light)")
    print(f"  blocked by something else : {blocked}")
    print(f"  self-shadowed (must be 0) : {self_shadow}")
    assert self_shadow == 0, "rays are hitting their own surface voxel"
    assert reached_lamp > 0, "no ray ever reached a visible lamp"
    assert timed_out > 0, "no ray ever reached the invisible fill light"
    assert blocked > 0, "nothing is ever occluded -- shadows are doing nothing"

    # ---- spatial filter on the visibility term ---------------------------
    if rf.SHADOW_FILTER_RADIUS > 0:
        lit_mask = illum_noshadow.max(axis=1) > 1e-9
        frac = np.divide(illum, illum_noshadow, out=np.ones_like(illum),
                         where=illum_noshadow > 1e-9)
        rough = frac.copy()
        frac = rf.filter_visibility(
            frac.reshape(H, W, 3), (shaded & lit_mask).reshape(H, W),
            r_face.reshape(H, W), r_pos.reshape(H, W, 3),
            rf.SHADOW_FILTER_RADIUS,
            rf.SHADOW_FILTER_PLANE_TOL).reshape(total, 3)

        # The filter must smooth without inventing light or changing the mean.
        sel = shaded & lit_mask
        assert frac[sel].min() >= -1e-9 and frac[sel].max() <= 1.0 + 1e-9, \
            "filtered visibility left the 0..1 range"
        before = float(rough[sel].mean())
        after = float(frac[sel].mean())
        assert abs(after - before) < 0.02, \
            f"filter shifted mean visibility {before:.3f} -> {after:.3f}"

        # Roughness = mean absolute difference between horizontal neighbours,
        # measured only within a surface. Filtering must reduce it.
        def roughness(a):
            g = a.reshape(H, W, 3).mean(axis=2)
            m = sel.reshape(H, W)
            pair = m[:, 1:] & m[:, :-1]
            return float(np.abs(g[:, 1:] - g[:, :-1])[pair].mean())

        r0, r1 = roughness(rough), roughness(frac)
        print(f"\nvisibility roughness: {r0:.4f} -> {r1:.4f} "
              f"({100 * (1 - r1 / r0):.0f}% smoother)")
        assert r1 < r0, "filter did not smooth the visibility term"
        illum = illum_noshadow * frac

    lit_frac = illum.max(axis=1)[shaded] / np.maximum(
        illum_noshadow.max(axis=1)[shaded], 1e-9)
    print(f"\nlight retained after shadowing: min {lit_frac.min():.2f} "
          f"median {np.median(lit_frac):.2f} max {lit_frac.max():.2f}")
    assert lit_frac.max() <= 1.0 + 1e-9, "shadowing added light"
    partial = np.count_nonzero((lit_frac > 0.02) & (lit_frac < 0.98))
    print(f"pixels in penumbra (partly shadowed): {partial} "
          f"({100 * partial / lit_frac.size:.1f}%)")
    assert partial > 0, "no soft shadows -- every pixel is fully lit or dark"

    # ---- render ----------------------------------------------------------
    illum *= rf.LIGHT_INTENSITY
    palette = np.array(rf.PALETTE)
    image = np.zeros((total, 3), np.uint8)
    image[lamp] = np.rint(palette[mats[lamp]] * 255).astype(np.uint8)
    albedo = np.zeros((total, 3))
    albedo[shaded] = palette[mats[shaded]]
    rf.normalize_image(illum, albedo, shaded, image)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "render_shadows.png")
    rf.write_png(out, image, W, H)
    print(f"\nwrote {out}")
    print("ALL SHADOW CHECKS PASSED")


if __name__ == "__main__":
    main()
