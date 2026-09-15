"""Sweep candidate light positions and report median illumination per
surface, so the choice is measured rather than guessed. Roof and floor are
separated (both are checker material) since they are the two faces most at
risk of going dark when the light moves."""
import statistics
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 128

CANDIDATES = [
    (0.32, 0.80, 0.18),   # current
    (0.50, 0.62, 0.10),
    (0.55, 0.55, 0.12),
    (0.50, 0.50, 0.08),
    (0.45, 0.68, 0.15),
    (0.55, 0.45, 0.10),
    (0.50, 0.40, 0.12),
]


def evaluate(material, occupied):
    buckets = {}
    for py in range(H):
        for px in range(W):
            eye, d = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, d)
            if dda is None:
                continue
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            if not r["hit"]:
                continue
            mat = material[r["z"]][r["y"]][r["x"]]
            if mat in (rf.MAT_CHECK_LIGHT, rf.MAT_CHECK_DARK):
                key = "roof" if r["y"] == rf.GRID - 1 else "floor"
            elif mat == rf.MAT_SPHERE_BIG:
                key = "red"
            elif mat == rf.MAT_SPHERE_SMALL:
                key = "blue"
            else:
                key = "walls"
            buckets.setdefault(key, []).append(rf.raw_diffuse(r))
    return {k: statistics.median(v) for k, v in buckets.items()}


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))
    keys = ("red", "blue", "walls", "floor", "roof")
    print(f"{'light (fractions of GRID)':28s}" +
          "".join(f"{k:>8s}" for k in keys) + f"{'min':>8s}")
    best = None
    for frac in CANDIDATES:
        rf.LIGHT_POS = tuple(f * rf.GRID for f in frac)
        med = evaluate(material, occupied)
        row = [med.get(k, 0.0) for k in keys]
        worst = min(row)
        print(f"{str(frac):28s}" + "".join(f"{v:8.3f}" for v in row) +
              f"{worst:8.3f}")
        if best is None or worst > best[1]:
            best = (frac, worst)
    print(f"\nbest worst-case surface: {best[0]} at {best[1]:.3f}")


if __name__ == "__main__":
    main()
