"""The vectorised host math must agree with the scalar reference exactly --
otherwise the FPGA gets different ray parameters than the model predicts.

Checks dda_init_batch vs dda_init, job_words_batch vs job_words,
raw_illumination_batch vs raw_illumination, and pack_scene's packbits form
against a hand-written bit loop.
"""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")

import numpy as np
import render_fpga as rf

W = H = 61          # deliberately not a power of two or a multiple of anything

print(f"GRID={rf.GRID}  comparing {W*H} rays")

eye, dirs = rf.camera_rays(W, H)
batch = rf.dda_init_batch(eye, dirs)

# --- camera_rays vs camera_ray ------------------------------------------
for i in (0, 1, W // 2, W * H // 3, W * H - 1):
    py, px = divmod(i, W)
    s_eye, s_dir = rf.camera_ray(px, py, W, H)
    assert s_eye == eye, (s_eye, eye)
    v = dirs[i] / np.linalg.norm(dirs[i])
    s = np.array(s_dir) / np.linalg.norm(s_dir)
    assert np.allclose(v, s, atol=1e-12), (i, v, s)
print("camera_rays matches camera_ray")

# --- dda_init_batch vs dda_init -----------------------------------------
mismatch = 0
checked = 0
for i in range(W * H):
    py, px = divmod(i, W)
    s_eye, s_dir = rf.camera_ray(px, py, W, H)
    scalar = rf.dda_init(s_eye, s_dir)
    if scalar is None:
        assert not batch["valid"][i], f"ray {i}: scalar rejects, batch accepts"
        continue
    assert batch["valid"][i], f"ray {i}: scalar accepts, batch rejects"
    checked += 1
    row = rf.dda_row(batch, i)
    if row["voxel"] != scalar["voxel"] or row["signs"] != scalar["signs"]:
        mismatch += 1
        print(f"  ray {i}: voxel/signs {row} vs {scalar}")
        continue
    # Fixed-point timers must match to the LSB -- the hardware compares them
    # directly, so an off-by-one changes which axis steps first.
    for k in ("next", "inc"):
        for a in range(3):
            if row[k][a] != scalar[k][a]:
                mismatch += 1
                print(f"  ray {i} {k}[{a}]: {row[k][a]} vs {scalar[k][a]}")
assert mismatch == 0, f"{mismatch} dda mismatches"
print(f"dda_init_batch matches dda_init exactly on all {checked} valid rays")

# --- job_words_batch vs job_words ---------------------------------------
live = np.nonzero(batch["valid"])[0]
sub = {k: batch[k][live] for k in ("voxel", "signs", "next", "inc")}
words = rf.job_words_batch(sub, live.astype(np.uint32), rf.MAX_STEPS)
for n, i in enumerate(live):
    if n % 97:
        continue
    scalar = rf.job_words(rf.dda_row(batch, int(i)), int(i), rf.MAX_STEPS)
    got = tuple(int(v) for v in words[n])
    assert got == scalar, (int(i), got, scalar)
print(f"job_words_batch matches job_words ({words.shape[0]} jobs)")

# --- raw_illumination_batch vs raw_illumination -------------------------
rng = np.random.default_rng(7)
faces = rng.integers(0, 7, 400).astype(np.uint8)
pos = rng.integers(0, rf.GRID, (400, 3)).astype(np.int32)
vec = rf.raw_illumination_batch(faces, pos[:, 0], pos[:, 1], pos[:, 2])
for i in range(400):
    scalar = rf.raw_illumination(
        {"face": int(faces[i]), "x": int(pos[i, 0]),
         "y": int(pos[i, 1]), "z": int(pos[i, 2])})
    assert np.allclose(vec[i], scalar, atol=1e-12), (i, vec[i], scalar)
print("raw_illumination_batch matches raw_illumination")

# --- pack_scene packbits form vs an explicit bit loop -------------------
material = rf.make_scene()
packed = rf.pack_scene(material)
assert len(packed) == rf.SCENE_BYTES
ref = bytearray(rf.SCENE_BYTES)
G, CB = rf.GRID, rf.COORD_BITS
occ = np.asarray(material) != 0
for z in range(G):
    for y in range(G):
        base = (z << (2 * CB)) | (y << CB)
        row = occ[z, y]
        for x in np.nonzero(row)[0]:
            addr = base | int(x)
            ref[addr >> 3] |= 1 << (addr & 7)
assert packed == bytes(ref), "packbits form differs from the bit-loop form"
print(f"pack_scene matches an explicit bit loop ({rf.SCENE_BYTES} bytes)")

print("\nALL VECTORISATION CHECKS PASSED")
