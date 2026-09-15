"""Higher-resolution offline preview of the current scene, plus a per-material
illumination report so lighting balance can be judged numerically."""
import os
import statistics
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 256
NAMES = {rf.MAT_SPHERE_BIG: "red sphere", rf.MAT_SPHERE_SMALL: "blue sphere",
         rf.MAT_WALL: "walls", rf.MAT_CHECK_LIGHT: "checker light",
         rf.MAT_CHECK_DARK: "checker dark", rf.MAT_LIGHT: "warm lamp",
         rf.MAT_LIGHT_COOL: "cool lamp"}


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))

    black = (0, 0, 0)
    image = [[black] * W for _ in range(H)]
    raw = [[0.0] * W for _ in range(H)]
    albedo = [[black] * W for _ in range(H)]
    hit_mask = [[False] * W for _ in range(H)]
    per_mat = {}

    for py in range(H):
        for px in range(W):
            eye, d = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, d)
            if dda is None:
                continue
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            if not r["hit"]:
                image[py][px] = rf.to_rgb8(rf.background_color(r))
                continue
            mat = material[r["z"]][r["y"]][r["x"]]
            if rf.is_emissive(material, r):
                image[py][px] = rf.emissive_color(material, r)
                per_mat.setdefault(mat, []).append((1.0, 1.0, 1.0))
                continue
            raw[py][px] = rf.raw_illumination(r)
            albedo[py][px] = rf.hit_albedo(material, r)
            hit_mask[py][px] = True
            per_mat.setdefault(mat, []).append(raw[py][px])
    rf.normalize_image(raw, albedo, hit_mask, image)

    for p in rf.LIGHT_PANELS:
        print(f"  panel axis={p['axis']} plane={p['plane']} "
              f"centre=({p['cu']},{p['cv']}) span={2 * p['half']} "
              f"{'visible' if p.get('visible', True) else 'INVISIBLE'}")
    print(f"  {len(rf.LIGHT_POINTS)} light samples total")
    # Report illumination magnitude plus the blue:red ratio -- >1 means the
    # surface is receiving a net blue tint from the cool lamp.
    print(f"{'material':16s} {'pixels':>7s} {'illum min':>10s} "
          f"{'median':>8s} {'max':>8s} {'blue:red':>9s}")
    for mat in sorted(per_mat):
        v = per_mat[mat]
        mags = sorted(max(c) for c in v)
        reds = sum(c[0] for c in v)
        blues = sum(c[2] for c in v)
        ratio = blues / reds if reds > 1e-9 else float("inf")
        print(f"{NAMES.get(mat, mat):16s} {len(v):7d} {mags[0]:10.3f} "
              f"{statistics.median(mags):8.3f} {mags[-1]:8.3f} {ratio:9.2f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "preview.png")
    rf.write_png(out, image, W, H)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
