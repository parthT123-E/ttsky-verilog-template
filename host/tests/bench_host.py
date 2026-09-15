"""Time the host-side work at full resolution, excluding all FPGA I/O, and
compare against the old scalar cost extrapolated from a small sample."""
import sys
import time

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf

W, H = rf.WIDTH, rf.HEIGHT
N = W * H
print(f"{W}x{H} = {N:,} rays, GRID={rf.GRID}\n")


def timed(label, fn):
    t = time.monotonic()
    out = fn()
    dt = time.monotonic() - t
    print(f"  {label:26s} {dt:7.2f}s")
    return out, dt


total = 0.0
material, dt = timed("make_scene", rf.make_scene);            total += dt
packed, dt = timed("pack_scene", lambda: rf.pack_scene(material)); total += dt
(eye, dirs), dt = timed("camera_rays", lambda: rf.camera_rays(W, H)); total += dt
rays, dt = timed("dda_init_batch", lambda: rf.dda_init_batch(eye, dirs))
total += dt

live = np.nonzero(rays["valid"])[0]
sub = {k: rays[k][live] for k in ("voxel", "signs", "next", "inc")}
words, dt = timed("job_words_batch",
                  lambda: rf.job_words_batch(sub, live.astype(np.uint32),
                                             rf.MAX_STEPS))
total += dt

# Stand-in results so shading can be timed without hardware.
rng = np.random.default_rng(0)
face = rng.integers(0, 6, N).astype(np.uint8)
pos = rng.integers(0, rf.GRID, (N, 3)).astype(np.int32)
illum, dt = timed("raw_illumination_batch",
                  lambda: rf.raw_illumination_batch(face, pos[:, 0],
                                                    pos[:, 1], pos[:, 2]))
total += dt

albedo = np.tile(np.array(rf.PALETTE[rf.MAT_WALL]), (N, 1))
image = np.zeros((N, 3), dtype=np.uint8)
mask = np.ones(N, dtype=bool)
_, dt = timed("normalize_image",
              lambda: rf.normalize_image(illum, albedo, mask, image))
total += dt
_, dt = timed("write_png", lambda: rf.write_png(
    r"C:\Users\eshaa\AppData\Local\Temp\claude\c--GitHub-Projects-ASIC-ttsky-ASIC"
    r"\8a89ece7-1893-47bb-ac22-fe124ae44ee3\scratchpad\bench.png",
    image, W, H))
total += dt

print(f"  {'-' * 34}\n  {'TOTAL host work':26s} {total:7.2f}s")

# --- what the scalar path cost, extrapolated from a sample -----------------
print("\nscalar reference cost, sampled and extrapolated:")
SAMPLE = 3000
t = time.monotonic()
for i in range(SAMPLE):
    py, px = divmod(i * 7919 % N, W)
    e, d = rf.camera_ray(px, py, W, H)
    rf.dda_init(e, d)
scalar_dda = (time.monotonic() - t) / SAMPLE * N

t = time.monotonic()
for i in range(SAMPLE):
    rf.raw_illumination({"face": int(face[i]), "x": int(pos[i, 0]),
                         "y": int(pos[i, 1]), "z": int(pos[i, 2])})
scalar_shade = (time.monotonic() - t) / SAMPLE * N

print(f"  {'camera_ray + dda_init':26s} {scalar_dda:7.2f}s")
print(f"  {'raw_illumination':26s} {scalar_shade:7.2f}s")
print(f"  {'-' * 34}\n  {'those two alone':26s} "
      f"{scalar_dda + scalar_shade:7.2f}s")
print(f"\nspeedup on that pair: "
      f"{(scalar_dda + scalar_shade) / max(total, 1e-9):.0f}x "
      f"(and pack_scene/write_png were loops too)")
