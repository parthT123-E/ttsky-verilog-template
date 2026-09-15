"""A light's contribution must depend only on its colour and geometry, never
on how finely it is sampled. Verify by changing LIGHT_SAMPLES and checking the
per-light contributions at a fixed surface point stay put."""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")
import render_fpga as rf

# A point on the big sphere's camera-facing side, normal (0,0,-1).
PX, PY, PZ = 24, 33, 32
NORMAL = (0.0, 0.0, -1.0)


def contrib(points):
    hx, hy, hz = PX + 0.5, PY + 0.5, PZ + 0.5
    nx, ny, nz = NORMAL
    t = [0.0, 0.0, 0.0]
    for lx, ly, lz, cr, cg, cb in points:
        dx, dy, dz = lx - hx, ly - hy, lz - hz
        L = (dx * dx + dy * dy + dz * dz) ** 0.5
        if L < 1e-9:
            continue
        c = (nx * dx + ny * dy + nz * dz) / L
        if c > 0.0:
            t[0] += cr * c
            t[1] += cg * c
            t[2] += cb * c
    return max(t)


def split(points):
    """Contribution of each panel, identified by its sample plane."""
    out = []
    idx = 0
    n = max(1, rf.LIGHT_SAMPLES)
    for panel in rf.LIGHT_PANELS:
        cnt = n * n + (4 * n if panel.get("visible", True) else 0)
        out.append(contrib(points[idx:idx + cnt]))
        idx += cnt
    return out


print(f"{'LIGHT_SAMPLES':>14s}{'total pts':>11s}"
      f"{'warm wall':>11s}{'cool wall':>11s}{'fill':>9s}")
base = None
for n in (2, 3, 4, 6):
    rf.LIGHT_SAMPLES = n
    # Strata are built once at import, so changing the sample count means
    # rebuilding them before the derived point list is meaningful.
    rf.LIGHT_STRATA = rf.light_strata()
    rf.LIGHT_POINTS = rf.light_sample_points()
    parts = split(rf.LIGHT_POINTS)
    print(f"{n:>14d}{len(rf.LIGHT_POINTS):>11d}"
          + "".join(f"{v:11.4f}" for v in parts))
    if base is None:
        base = parts
    else:
        for a, b in zip(base, parts):
            assert abs(a - b) < 0.05 * max(b, 1e-6) + 0.01, \
                f"contribution moved with sample count: {a:.4f} -> {b:.4f}"

print("\nPASS: each light's contribution is stable across sample densities")

# And the fill must still be the dominant light on a camera-facing surface.
rf.LIGHT_SAMPLES = 2
rf.LIGHT_STRATA = rf.light_strata()
rf.LIGHT_POINTS = rf.light_sample_points()
parts = split(rf.LIGHT_POINTS)
share = parts[2] / sum(parts)
print(f"fill light share of a -z facing surface: {share:.0%}")
assert share > 0.5, "fill should dominate surfaces facing the camera"
print("PASS: fill light dominates the spheres' camera-facing sides again")
