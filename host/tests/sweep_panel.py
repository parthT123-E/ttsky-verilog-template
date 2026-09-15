"""The spheres' camera-facing surfaces face -z, so a panel deep in the box
lights their sides but not their fronts. Sweep the panel's z position (and a
two-panel option) to quantify the trade-off."""
import statistics
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 128


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
    keys = ("red", "blue", "walls", "floor", "roof")
    print(f"{'panel centre z':>15s}" + "".join(f"{k:>8s}" for k in keys)
          + f"{'min':>8s}")
    for cz_frac in (0.50, 0.35, 0.25, 0.15):
        rf.LIGHT_PANEL_CZ = int(rf.GRID * cz_frac)
        rf.LIGHT_POINTS = rf.light_sample_points()
        material = rf.make_scene()
        occupied = unpack_scene(rf.pack_scene(material))
        med = evaluate(material, occupied)
        row = [med.get(k, 0.0) for k in keys]
        print(f"{cz_frac:15.2f}" + "".join(f"{v:8.3f}" for v in row)
              + f"{min(row):8.3f}")


if __name__ == "__main__":
    main()
