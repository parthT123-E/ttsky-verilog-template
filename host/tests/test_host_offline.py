"""Offline check of host/render_fpga.py against a bit-exact model of the RTL.

Models step_control_multi + voxel_raytracer_core semantics:
  - test CURRENT voxel occupancy (including the starting voxel)
  - terminate on hit / out-of-bounds (stepped coords) / steps >= max_steps
  - axis_choose: unsigned min of next_x/y/z (x wins ties, then y)
  - step_update: voxel +/-1 on chosen axis, next += inc (mod 2^32)
  - face: entry face of the hit voxel (post entry-face fix), sentinel 6 for
    a hit on the ray's own starting voxel

Grid-size agnostic: derives everything from rf.GRID.
"""
import os
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf

MASK32 = 0xFFFFFFFF
BITS = rf.GRID.bit_length() - 1   # address bits per axis (GRID is a power of 2)
MAXV = rf.GRID - 1                # MAX_VAL in bounds_check.sv
# Coordinate registers are one bit WIDER than the grid (COORD_REG_BITS in
# xem7310_raytracer_top.sv). That headroom bit is what lets a step off either
# face land outside 0..MAXV so bounds_check can detect it.
CMASK = (1 << (BITS + 1)) - 1


def unpack_scene(data):
    """Invert pack_scene: byte/bit -> occupancy lookup by (x, y, z)."""
    def occupied(x, y, z):
        addr = (z << (2 * BITS)) | (y << BITS) | x
        return bool(data[addr >> 3] & (1 << (addr & 7)))
    return occupied


def hw_model_trace(occupied, dda, max_steps):
    """Bit-exact model of the RTL DDA loop."""
    ix, iy, iz = dda["voxel"]
    nx, ny, nz = dda["next"]
    sx, sy, sz = dda["signs"]
    steps = 0
    entry_face = 6  # sentinel: no voxel entered yet
    while True:
        # Test CURRENT voxel (bounds guaranteed on entry; RTL checks bounds
        # on the STEPPED coords)
        solid = occupied(ix, iy, iz)
        # axis_choose on current timers
        if nx <= ny and nx <= nz:
            axis = 0
        elif ny <= nz:
            axis = 1
        else:
            axis = 2
        # step_update
        jx, jy, jz = ix, iy, iz
        if axis == 0:
            jx = (ix + (1 if sx else -1)) & CMASK
            nx = (nx + dda["inc"][0]) & MASK32
        elif axis == 1:
            jy = (iy + (1 if sy else -1)) & CMASK
            ny = (ny + dda["inc"][1]) & MASK32
        else:
            jz = (iz + (1 if sz else -1)) & CMASK
            nz = (nz + dda["inc"][2]) & MASK32
        oob = jx > MAXV or jy > MAXV or jz > MAXV
        step_face = {0: (0 if sx else 1), 1: (2 if sy else 3),
                     2: (4 if sz else 5)}[axis]
        steps += 1
        if solid or oob or steps >= max_steps:
            return {
                "hit": solid,
                "timeout": (not solid) and (not oob),
                "x": ix, "y": iy, "z": iz,
                "face": entry_face, "steps": steps, "pixel_id": 0,
            }
        ix, iy, iz = jx, jy, jz
        entry_face = step_face  # this step's face = next voxel's entry face


def main():
    print(f"GRID={rf.GRID}  SCENE_BYTES={rf.SCENE_BYTES}  "
          f"MAX_STEPS={rf.MAX_STEPS}")
    material = rf.make_scene()
    packed = rf.pack_scene(material)
    assert len(packed) == rf.SCENE_BYTES, len(packed)

    # pack/unpack round-trip: packed occupancy must equal (material != 0)
    occupied = unpack_scene(packed)
    for z in range(rf.GRID):
        for y in range(rf.GRID):
            for x in range(rf.GRID):
                assert occupied(x, y, z) == bool(material[z][y][x]), (x, y, z)
    print(f"pack_scene round-trip OK ({rf.SCENE_BYTES} bytes)")

    import numpy as np
    W = H = 96
    total = W * H
    image = np.zeros((total, 3), dtype=np.uint8)
    raw = np.zeros((total, 3), dtype=np.float64)
    albedo = np.zeros((total, 3), dtype=np.float64)
    hit_mask = np.zeros(total, dtype=bool)
    hits = misses = off = timeouts = 0
    max_steps_seen = 0
    mats_hit = set()
    for py in range(H):
        for px in range(W):
            idx = py * W + px
            eye, d = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, d)
            if dda is None:
                off += 1
                continue
            # field range checks -- these must fit the hardware wire fields
            for v in dda["voxel"]:
                assert 0 <= v <= MAXV, dda
            for v in dda["next"] + dda["inc"]:
                assert 0 <= v <= MASK32, dda
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            r["pixel_id"] = py * W + px
            max_steps_seen = max(max_steps_seen, r["steps"])
            if r["hit"]:
                hits += 1
                mats_hit.add(int(material[r["z"]][r["y"]][r["x"]]))
                assert occupied(r["x"], r["y"], r["z"]), r
                if rf.is_emissive(material, r):
                    image[idx] = rf.emissive_color(material, r)
                    continue
                if rf.REFLECTIVITY[rf.hit_material(material, r)] > 0.0:
                    # This reference path traces primary rays only, with no
                    # bounce loop. A mirror hit is left black rather than
                    # diffusely shaded with its own near-white albedo, same
                    # as how the real renderer treats an exhausted
                    # reflection chain (see `trapped` in main()).
                    continue
                raw[idx] = rf.raw_illumination(r)
                albedo[idx] = rf.hit_albedo(material, r)
                hit_mask[idx] = True
            else:
                image[idx] = rf.to_rgb8(rf.background_color(r))
                misses += 1
                if r["timeout"]:
                    timeouts += 1
    rf.normalize_image(raw, albedo, hit_mask, image)

    print(f"rays: {W*H}  hits: {hits}  misses: {misses}  "
          f"timeouts: {timeouts}  off-grid: {off}")
    print(f"max steps taken by any ray: {max_steps_seen} "
          f"(budget {rf.MAX_STEPS})")
    print(f"materials hit: {sorted(mats_hit)}")
    assert hits > 0
    assert timeouts == 0, "unexpected timeouts -- MAX_STEPS too low?"
    assert rf.MAT_EMPTY not in mats_hit, "a hit resolved to empty material"
    # All-mirror box: both spheres and the mirrored enclosure. This path is
    # primary rays only (no bounce tracing -- see hw_model_trace), and with
    # the camera now sitting close inside a mostly-mirrored box, a lamp
    # panel embedded in a side wall is only reliably reached via a bounce,
    # not guaranteed on a primary ray at this resolution.
    for expect in (rf.MAT_SPHERE_BIG, rf.MAT_SPHERE_SMALL, rf.MAT_MIRROR):
        assert expect in mats_hit, f"material {expect} not visible"
    # Whenever the emissive panel IS directly visible, it must render at full
    # brightness and be excluded from the normalisation, i.e. be the
    # brightest thing in the frame.
    if rf.MAT_LIGHT in mats_hit:
        lamp = rf.to_rgb8(rf.PALETTE[rf.MAT_LIGHT])
        brightest = int(image.astype(int).sum(axis=1).max())
        assert sum(lamp) >= brightest, (
            f"lamp {lamp} is not the brightest pixel ({brightest}) -- "
            f"emissive handling or normalisation is wrong")
    # The box encloses the view, so essentially every ray should land on a
    # surface -- stray escapes would show as black bands at the frame edge.
    assert misses < 0.01 * (W * H), f"{misses} rays escaped the enclosure"
    assert image[(H // 2) * W + W // 2].sum() > 0, "centre pixel should be lit"

    # Colour sanity: the two spheres must render as distinguishable hues,
    # which only holds if the per-voxel material lookup actually works.
    hues = set()
    for idx in np.nonzero(hit_mask)[0]:
        r8, g8, b8 = (int(c) for c in image[idx])
        if max(r8, g8, b8) > 40:
            hues.add((r8 > b8 + 20, b8 > r8 + 20))  # reddish / bluish
    assert (True, False) in hues, "no reddish pixels -- big sphere miscoloured"
    assert (False, True) in hues, "no bluish pixels -- small sphere miscoloured"

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "render_offline.png")
    rf.write_png(out, image, W, H)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
