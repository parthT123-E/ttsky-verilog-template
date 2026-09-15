"""A -z-facing surface at depth z is only lit by light samples SHALLOWER than
it. The panels sample at z = 27 and 37, so a sphere front must sit past ~27 to
catch any light and past ~37 to catch all of it. Sweep how far back the
spheres can go before they clip the back wall, and measure the result."""
import statistics
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 128
BIG_R, SMALL_R = 0.22, 0.12

# (big z fraction, small z fraction)
CANDIDATES = [
    (0.55, 0.34),   # current
    (0.62, 0.45),
    (0.68, 0.54),
    (0.72, 0.60),
    (0.74, 0.64),
]


def build(big_z, small_z):
    """Rebuild make_scene() with the spheres at the given depths."""
    material = [[[rf.MAT_EMPTY] * rf.GRID for _ in range(rf.GRID)]
                for _ in range(rf.GRID)]
    G = rf.GRID
    check = max(1, rf.CHECKER_SIZE)
    for z in range(G):
        for x in range(G):
            pale = (((x // check) + (z // check)) & 1) == 0
            tile = rf.MAT_CHECK_LIGHT if pale else rf.MAT_CHECK_DARK
            material[z][0][x] = tile
            material[z][G - 1][x] = tile
    for z in range(G):
        for y in range(G):
            material[z][y][0] = rf.MAT_WALL
            material[z][y][G - 1] = rf.MAT_WALL
    for y in range(G):
        for x in range(G):
            material[G - 1][y][x] = rf.MAT_WALL
    h = rf.LIGHT_PANEL_HALF
    for panel_x, _ in rf.LIGHT_PANELS:
        for y in range(rf.LIGHT_PANEL_CY - h, rf.LIGHT_PANEL_CY + h):
            for z in range(rf.LIGHT_PANEL_CZ - h, rf.LIGHT_PANEL_CZ + h):
                material[z][y][panel_x] = rf.MAT_LIGHT
    rf.add_sphere(material, 0.38, 0.52, big_z, BIG_R, rf.MAT_SPHERE_BIG)
    rf.add_sphere(material, 0.70, 0.34, small_z, SMALL_R, rf.MAT_SPHERE_SMALL)
    return material


def evaluate(material, occupied):
    buckets = {}
    for py in range(H):
        for px in range(W):
            eye, d = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, d)
            if dda is None:
                continue
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            if not r["hit"] or rf.is_emissive(material, r):
                continue
            mat = material[r["z"]][r["y"]][r["x"]]
            key = {rf.MAT_SPHERE_BIG: "red",
                   rf.MAT_SPHERE_SMALL: "blue"}.get(mat, "room")
            buckets.setdefault(key, []).append(rf.raw_diffuse(r))
    return {k: statistics.median(v) for k, v in buckets.items()}


def main():
    G = rf.GRID
    print(f"panels sample at z = "
          f"{sorted({round(p[2]) for p in rf.LIGHT_POINTS})}; "
          f"back wall at z={G - 1}")
    print(f"{'big z':>7s}{'small z':>9s}{'big front':>11s}{'big back':>10s}"
          f"{'red':>8s}{'blue':>8s}{'room':>8s}")
    for bz, sz in CANDIDATES:
        front = bz * G - BIG_R * G
        back = bz * G + BIG_R * G
        material = build(bz, sz)
        occupied = unpack_scene(rf.pack_scene(material))
        med = evaluate(material, occupied)
        clip = "  CLIPS!" if back > G - 2 else ""
        print(f"{bz:7.2f}{sz:9.2f}{front:11.1f}{back:10.1f}"
              f"{med.get('red', 0):8.3f}{med.get('blue', 0):8.3f}"
              f"{med.get('room', 0):8.3f}{clip}")


if __name__ == "__main__":
    main()
