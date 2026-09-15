"""Diagnose the thin dark seams seen on the spheres in the 512x512 render.

Reproduces the render through the bit-exact RTL model, finds hit pixels that
are markedly darker than their neighbours, and reports what distinguishes
them (entry face, hit voxel, step count).
"""
import os
import statistics
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import render_fpga as rf
from test_host_offline import hw_model_trace, unpack_scene

W = H = 512


def main():
    material = rf.make_scene()
    occupied = unpack_scene(rf.pack_scene(material))

    face = [[None] * W for _ in range(H)]
    vox = [[None] * W for _ in range(H)]
    steps = [[0] * W for _ in range(H)]
    raw = [[0.0] * W for _ in range(H)]
    hit = [[False] * W for _ in range(H)]
    mat = [[0] * W for _ in range(H)]

    print(f"tracing {W}x{H} through the RTL model ...")
    for py in range(H):
        for px in range(W):
            eye, d = rf.camera_ray(px, py, W, H)
            dda = rf.dda_init(eye, d)
            if dda is None:
                continue
            r = hw_model_trace(occupied, dda, rf.MAX_STEPS)
            if not r["hit"]:
                continue
            hit[py][px] = True
            face[py][px] = r["face"]
            vox[py][px] = (r["x"], r["y"], r["z"])
            steps[py][px] = r["steps"]
            raw[py][px] = rf.raw_diffuse(r)
            mat[py][px] = material[r["z"]][r["y"]][r["x"]]

    # Global face histogram
    hist = {}
    for py in range(H):
        for px in range(W):
            if hit[py][px]:
                hist[face[py][px]] = hist.get(face[py][px], 0) + 1
    print(f"\nentry-face histogram over all hits: "
          f"{ {k: hist[k] for k in sorted(hist)} }")

    # Find "seam" pixels: a hit whose illumination is well below the median
    # of its hit neighbours (i.e. a local dark line, not a broad terrace).
    seams = []
    for py in range(1, H - 1):
        for px in range(1, W - 1):
            if not hit[py][px]:
                continue
            nb = [raw[py + dy][px + dx]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy or dx) and hit[py + dy][px + dx]]
            if len(nb) < 6:
                continue
            med = statistics.median(nb)
            if med > 0.05 and raw[py][px] < 0.6 * med:
                seams.append((px, py, med))

    print(f"\nseam pixels (much darker than neighbours): {len(seams)}")
    if not seams:
        print("none found -- artifact does not reproduce in the model")
        return

    # What faces do seam pixels have vs their neighbours?
    seam_faces = {}
    same_voxel_as_neighbour = 0
    step_deltas = []
    for px, py, med in seams:
        f = face[py][px]
        seam_faces[f] = seam_faces.get(f, 0) + 1
        nb_v = [vox[py + dy][px + dx]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and hit[py + dy][px + dx]]
        if vox[py][px] in nb_v:
            same_voxel_as_neighbour += 1
        nb_s = [steps[py + dy][px + dx]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and hit[py + dy][px + dx]]
        step_deltas.append(steps[py][px] - statistics.median(nb_s))

    print(f"seam entry-face histogram: "
          f"{ {k: seam_faces[k] for k in sorted(seam_faces)} }")
    print(f"seam pixels whose hit voxel also appears among neighbours: "
          f"{same_voxel_as_neighbour}/{len(seams)}")
    print(f"step-count delta vs neighbours: "
          f"min={min(step_deltas)} median={statistics.median(step_deltas)} "
          f"max={max(step_deltas)}")

    # Material breakdown
    seam_mats = {}
    for px, py, _ in seams:
        seam_mats[mat[py][px]] = seam_mats.get(mat[py][px], 0) + 1
    print(f"seam material histogram: "
          f"{ {k: seam_mats[k] for k in sorted(seam_mats)} }")

    # Detailed look at a handful, with their horizontal neighbours
    print("\nsample seam pixels (px,py): face/voxel/steps, with L and R "
          "neighbours")
    for px, py, med in seams[:12]:
        def desc(x, y):
            if not hit[y][x]:
                return "miss"
            return f"f{face[y][x]} v{vox[y][x]} s{steps[y][x]}"
        print(f"  ({px:3d},{py:3d})  L[{desc(px-1, py)}]  "
              f"*[{desc(px, py)}]*  R[{desc(px+1, py)}]")


if __name__ == "__main__":
    main()
