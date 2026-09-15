"""Does the left wall receive light from its own (warm) panel?

Walks left-wall voxels outward from the panel and reports the illumination
they receive, split into the warm contribution (left panel) and the cool one
(right panel). If the wall near the panel is NOT warm-biased, the panel is
failing to light its own wall.
"""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf


def illum_at(x, y, z, normal, points):
    hx, hy, hz = x + 0.5, y + 0.5, z + 0.5
    nx, ny, nz = normal
    tr = tg = tb = 0.0
    for lx, ly, lz, cr, cg, cb in points:
        dx, dy, dz = lx - hx, ly - hy, lz - hz
        length = (dx * dx + dy * dy + dz * dz) ** 0.5
        if length < 1e-9:
            continue
        c = (nx * dx + ny * dy + nz * dz) / length
        if c > 0.0:
            tr += cr * c
            tg += cg * c
            tb += cb * c
    k = rf.LIGHT_INTENSITY / len(rf.LIGHT_POINTS)
    return (tr * k, tg * k, tb * k)


def main():
    warm_pts = [p for p in rf.LIGHT_POINTS if p[0] < rf.GRID / 2]
    cool_pts = [p for p in rf.LIGHT_POINTS if rf.GRID / 2 < p[0] < rf.GRID]
    fill_pts = [p for p in rf.LIGHT_POINTS
                if p not in warm_pts and p not in cool_pts]
    print(f"left-panel (warm) samples : {[tuple(round(c,1) for c in p[:3]) for p in warm_pts]}")
    print(f"right-panel (cool) samples: {[tuple(round(c,1) for c in p[:3]) for p in cool_pts]}")
    print(f"fill samples              : {[tuple(round(c,1) for c in p[:3]) for p in fill_pts]}")

    panel = rf.LIGHT_PANELS[0]
    cy, cz = panel["cu"], panel["cv"]
    half = panel["half"]
    normal = (1.0, 0.0, 0.0)   # left wall faces +x, into the room
    print(f"\nleft panel occupies voxel x={panel['plane']}, "
          f"y/z {cy-half}..{cy+half-1}; wall is at x=0")
    print(f"\n{'wall voxel':>14s}{'gap':>5s}{'illum RGB':>26s}"
          f"{'mag':>7s}{'warm share':>12s}")
    for dz in (0, 4, 8, 11, 14, 20, 28):
        z = cz + half - 1 + dz     # step away from the panel edge in z
        if z >= rf.GRID:
            break
        total = illum_at(0, cy, z, normal, rf.LIGHT_POINTS)
        warm = illum_at(0, cy, z, normal, warm_pts)
        share = (max(warm) / max(total)) if max(total) > 1e-9 else 0.0
        print(f"   (0,{cy:2d},{z:2d})  {dz:4d}"
              f"   ({total[0]:.3f},{total[1]:.3f},{total[2]:.3f})"
              f"{max(total):7.3f}{share:11.0%}")


if __name__ == "__main__":
    main()
