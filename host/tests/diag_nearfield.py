"""Near-field check: how much of its OWN panel's light does the wall get, and
does finer area sampling change it?

A wall point sees the opposite wall's lamp almost head-on (N.L ~ 0.95) but its
own lamp nearly edge-on, so the near-field warm contribution depends entirely
on having sample points close to the wall. Sweep LIGHT_SAMPLES to separate
'physically small' from 'under-sampled'.
"""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf


def contrib(x, y, z, normal, points):
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
    return (tr, tg, tb)


NORMAL = (1.0, 0.0, 0.0)


def main():
    panel = rf.LIGHT_PANELS[0]
    cy, cz = panel["cu"], panel["cv"]
    half = panel["half"]
    edge_z = cz + half          # first voxel beyond the panel in z

    print(f"left panel: voxel x={panel['plane']}, y/z "
          f"{cy - half}..{cy + half - 1}; wall at x=0, normal +x")
    print("warm share of total illumination, for wall voxels just past the "
          "panel edge\n")
    header = "".join(f"z={edge_z + d:<6d}" for d in (0, 1, 2, 4, 8))
    print(f"{'LIGHT_SAMPLES':>14s}{'samples':>9s}   {header}")

    for n in (2, 3, 4, 6, 8):
        rf.LIGHT_SAMPLES = n
        rf.LIGHT_POINTS = rf.light_sample_points()
        warm = [p for p in rf.LIGHT_POINTS if p[0] < rf.GRID / 2]
        row = ""
        for d in (0, 1, 2, 4, 8):
            z = edge_z + d
            tot = contrib(0, cy, z, NORMAL, rf.LIGHT_POINTS)
            w = contrib(0, cy, z, NORMAL, warm)
            share = max(w) / max(tot) if max(tot) > 1e-9 else 0.0
            row += f"{share:7.0%} "
        print(f"{n:>14d}{len(rf.LIGHT_POINTS):>9d}   {row}")


if __name__ == "__main__":
    main()
