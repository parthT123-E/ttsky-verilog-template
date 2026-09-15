# Handoff: FPGA voxel raytracer (context for an assistant continuing this work)

You are continuing work on an FPGA voxel raytracer. This document is written
for you, not for a human. It records the architecture, the invariants that are
easy to violate, and the failures already hit and fixed — several were subtle
and cost multiple hardware rebuild cycles to find. **Read the "Invariants"
section before changing RTL or the host datapath.**

---

## 1. What this is

A voxel DDA raytracer running on an Opal Kelly **XEM7310-A75** (Artix-7
XC7A75T) driven over FrontPanel USB by a Python host. Originally a TinyTapeout
ASIC design; the QSPI interface was replaced with a FrontPanel wrapper for the
FPGA port.

**Critical architectural fact: the FPGA solves *visibility only*.** For each
ray job it returns the first solid voxel hit, the face entered through, and a
step count. *All* shading — colour, lighting, shadows, specular, reflections,
tone mapping — happens host-side in numpy. This is why most feature work is
host-only and needs no rebuild.

| | |
|---|---|
| Board | XEM7310-A75, serial 24090019ZE |
| Grid | 128³ voxels, 1 bit/voxel occupancy = 256 KB scene |
| Render | 1024×1024 |
| Bitstream ID | **RTF5** (`0x52544635`) |
| Host venv | `C:\GitHub_Projects\ASIC\ttsky-ASIC\fpga-venv` (has `ok` 6.0.0 + numpy) |
| Vivado project | **NOT in this repo** — user has it elsewhere |
| Bitstream | `host/xem7310_raytracer_top.bit` (path is script-relative) |

---

## 2. Layout

```
src/*.sv                      RTL
  xem7310_raytracer_top.sv    FrontPanel wrapper — MOST changes happen here
  raytracer_top.sv            tracer top (params threaded through)
  step_control_multi.sv       5-context scheduler + result FIFO
  voxel_raytracer_core.sv     5-stage DDA pipeline
  voxel_ram.sv                voxel store; wraps blk_mem_gen_0 IP
  scene_loader_if.sv          scene-load write path
  axis_choose / step_update / bounds_check / voxel_addr_map   DDA primitives
  qspi_stream_raytracer.sv, raytracer_host_regs.sv,
  tt_um_ttsky_asic_raytracer.sv    LEGACY ASIC path, not used by the FPGA build

host/render_fpga.py           the entire host renderer (single file)
host/tests/*.py               verification suite — run these after any change
host/xem7310_raytracer_top.bit
```

Run anything with the venv python:
`C:\GitHub_Projects\ASIC\ttsky-ASIC\fpga-venv\Scripts\python.exe`

---

## 3. Tooling available to you

- **ModelSim** at `/c/intelFPGA_lite/18.1/modelsim_ase/win32aloem/` (`vlog.exe`,
  `vsim.exe`, `vlib.exe`). **Use it.** `vlog -sv <files>` compiles the RTL and
  catches width/syntax/port errors. `vsim -c -do "run -all; quit -f" work.tb`
  runs testbenches. Opal Kelly IP (`okHost`, `okBTPipeIn`, …) and
  `blk_mem_gen_0` are black boxes and cannot be simulated, but everything else can.
- **No Vivado, no Verilator, no iverilog.** You cannot build a bitstream or run
  synthesis. The user does that.
- **Offline RTL model**: `host/tests/test_host_offline.py` contains
  `hw_model_trace()`, a bit-exact Python model of the DDA loop. Use it to verify
  any rendering change *before* the user burns a rebuild cycle. It is slow
  (pure Python per ray) — keep test resolutions ≤ 224.

---

## 4. Working style that has worked here

- **Measure, don't guess.** Several "obvious" diagnoses were wrong (see §7).
  Write a throwaway diagnostic into `host/tests/`, run it, then decide.
- **Verify before the user rebuilds.** A Vivado cycle is expensive. Compile with
  vlog, simulate what you can, and run the offline model.
- **Distinguish host-only vs rebuild-required changes** and say which it is.
- Tests here assert *mechanisms*, not just "it ran" — e.g. that shadow rays never
  self-shadow, that filtering preserves the mean, that a light's contribution is
  independent of its sample count. Keep that standard.

---

## 5. Hardware interface (FrontPanel endpoints)

WireIn `0x00` xyz (x@0, y@8, z@16, each `COORD_REG_BITS`=8 wide)
WireIn `0x01` `[9:0]` max_steps, `[10..12]` sx/sy/sz, `[26:13]` pixel_id
WireIn `0x02-07` next_x/y/z, inc_x/y/z · WireIn `0x08` `[0]` route results to PipeOut
TriggerIn `0x40`: 0 reset, 1 begin scene load, 2 submit job, 3 pop result, 4 clear errors, 5 abort
BTPipeIn `0x80` scene (256 KB) · BTPipeIn `0x81` bulk jobs · PipeOut `0xA0` bulk results
WireOut `0x20` status · `0x21` voxels written · `0x22-24` result · `0x25` build ID · `0x26` FIFO levels
TriggerOut `0x60`: 0 result ready, 1 scene loaded, 2 error

**Job record** = 8 × u32 `{xyz, meta, next_x, next_y, next_z, inc_x, inc_y, inc_z}`
**Result record** = 4 × u32 `{info, xy, z_steps, tag}` where `tag = {0xA5, seq[23:0]}`

---

## 6. INVARIANTS — violating these causes silent corruption or deadlock

1. **Coordinate registers need one bit MORE than the grid.**
   `COORD_REG_BITS = ADDR_COORD_BITS + 1`. `bounds_check` detects out-of-bounds
   by `coord > MAX_VAL`; with exactly-sized registers a coordinate can never
   exceed MAX_VAL, so OOB never fires and rays wrap around the grid until
   timeout. This bit is the *entire* mechanism.

2. **`okBTPipeIn` commits a whole block once `ep_ready` is high.** It cannot be
   throttled mid-block. Any input FIFO must have room for a full
   `BLOCK_WORDS` (256) at the moment it asserts ready, or words are silently
   dropped.

3. **`okPipeOut` samples `ep_datain` the cycle AFTER asserting `ep_read`.** The
   result FIFO therefore uses a **registered** read, not first-word-fall-through.
   FWFT skips the first word of every stream. (This cost a rebuild to find.)

4. **A whole host batch must fit BOTH FIFOs.** The host writes every job before
   reading any result, so worst case all its results queue mid-write. If the
   result FIFO overflows → tracer stalls → jobs stop draining → the blocking job
   write never returns → **deadlock**, not a wrong pixel.
   `MAX_BATCH_RAYS = min(JOB_FIFO_DEPTH/8, RES_FIFO_DEPTH/4)`.

5. **Bump the build ID on ANY host-visible change** (endpoint map, payload
   format, FIFO depth, grid size). The host checks it at startup. Without this,
   a stale bitstream fails deep in a render with a confusing error — or hangs.
   History: RTF1 wires → RTF2 bulk pipes → RTF3 tagged results + PipeOut fix →
   RTF4 128³ → RTF5 deeper FIFOs.

6. **`blk_mem_gen_0` IP config** (regenerate in Vivado whenever geometry changes):
   Simple Dual Port, common clock, **32 bits wide × 65536 deep**,
   **Enable Port Type = "Always Enabled" on BOTH ports** (anything else exposes
   `ena`/`enb`, which are not connected → Opt Design fails with a confusing
   "LUT5 missing input" inside the IP), **byte write enable OFF**,
   **Port B read latency = 2** (the bit-select delay is hardwired to 2).
   BMG caps depth at 2^20, which is why the RAM is wide rather than 1-bit.

7. **Each light must be normalised by ITS OWN sample count**, not the global
   count. Otherwise adding samples to one light silently re-weights every other
   light. (This dimmed the fill light 2.3× when edge sampling was added.)

8. **`pixel_id` is 14 bits and wraps.** Results return out of order within a
   batch. Jobs carry their row index as pixel_id and results are matched via a
   scratch lookup array. Safe because a batch (1024) « the wrap period (16384).

9. **Grid must stay a power of two per axis.** `voxel_addr_map` builds the
   address by bit-concatenating `{z,y,x}`.

10. **`face_id` is the face the ray ENTERED the hit voxel through.** The RTL was
    fixed for this (it originally reported the *exit* face of the terminating
    step). `face_id == 6` is a sentinel meaning "hit its own starting voxel, no
    entry face".

---

## 7. Rendering pipeline (host, `render_fpga.py`)

All vectorised over the full image with numpy. Scalar reference functions
(`dda_init`, `raw_illumination`, `job_words`) are retained as the readable
reference and are checked against the vectorised path by `test_vectorised.py`
**to the LSB** — the hardware compares fixed-point timers directly, so an
off-by-one changes which axis steps first.

```
camera_rays -> dda_init_batch -> job_words_batch -> trace_all   (primary)
  -> mirror bounce loop (reflect, re-trace, accumulate throughput)
  -> per-light-sample loop: prune by N·L, shadow ray, diffuse + Blinn-Phong
  -> isolate visibility, spatially filter it, re-apply
  -> normalize_image -> write_png
```

### Hard-won details in the shading path

- **Shadow ray origin must be the EXACT sub-voxel hit point**
  (`ray_hit_points()`), not the voxel centre. At 1024²/128³ a voxel is ~8 px, so
  voxel-centre origins make ~64 pixels share one origin → blocky error that no
  filter can remove. The host recovers the true crossing analytically from the
  ray and the reported face.
- **Origin is offset along the normal** by `SHADOW_ORIGIN_BIAS` so `floor()`
  lands in the neighbouring empty voxel. The tracer tests its own starting
  voxel; starting on the surface self-shadows everything.
- **`max_steps` = exact voxel distance to the light.** "Ran out of steps" then
  means "reached the light unobstructed" — the *only* way to detect the
  **invisible** fill light, which has no geometry to land on. Lights that do
  have geometry are detected by landing on their emissive material.
- **Jitter light samples per pixel** (`jitter_strata`). Without it every pixel
  samples the same points, so each sample's hard shadow edge is a shared
  contour → N concentric rings per panel. Jitter costs zero extra rays and
  converts rings into noise.
- **Filter the VISIBILITY term, not the final image.** `illum_open` (unshadowed,
  free) is accumulated alongside `illum`; the ratio is the visibility, which is
  filtered and re-applied. Blurring the product would soften the shading
  terminator and bleed colour across material edges.
- **The filter guard is COPLANARITY, not distance.** Same face + offset along
  that face's normal < `SHADOW_FILTER_PLANE_TOL`. A flat wall's voxels are
  coplanar however far apart (so the penumbra smooths freely); a sphere's
  terraces step along the normal (so per-voxel detail survives). A distance test
  cannot tell them apart. Filter is separable: 2×(2R+1) taps.
- **Order matters**: jitter *then* filter. Filtering the banded version just
  produces blurry rings.
- **Specular is accumulated separately** from diffuse and passed to
  `normalize_image` as its own term, so a highlight keeps the *lamp's* colour,
  not the material's albedo.
- **Mirror throughput is applied BEFORE normalisation**, so the brightness
  stretch sees the dimming and deep reflections stay correctly darker.
- **After a mirror bounce the "view" vector is the incoming ray reversed**, not
  the camera ray, or specular highlights appear in the wrong place in the mirror.
- **Mirror termination**: reflectivity < 1 makes throughput decay geometrically
  (`MIRROR_MIN_THROUGHPUT` = 1/255 ends it adaptively); `MIRROR_BOUNCES` is the
  hard backstop for a ray trapped between parallel mirrors. Camera rays always
  carry +z so a side-wall corridor still advances to the back wall — but a
  purely axial ray *would* cycle, so the cap is not optional.

---

## 8. Current scene and config

Enclosed box filling the whole grid, open toward −z (camera looks down +z).
Camera at `(GRID/2, GRID/2, −GRID·0.875)`.

| material | id | notes |
|---|---|---|
| big sphere | 1 | warm red, at `(0.38, 0.52, 0.72)·GRID`, r `0.22·GRID` |
| small sphere | 2 | blue, at `(0.70, 0.34, 0.60)·GRID`, r `0.12·GRID` |
| wall (back) | 3 | warm neutral grey |
| checker light/dark | 4 / 5 | floor + roof, `CHECKER_SIZE` = GRID/8 |
| warm lamp | 6 | emissive, left wall panel |
| cool lamp | 7 | emissive, right wall panel |
| neutral lamp | 8 | emissive; **defined but unused** — swap into a light to use |
| shiny wall | 9 | left wall, Blinn-Phong `(0.9, 32)` |
| mirror | 10 | right wall, reflectivity 0.9 |

Spheres sit *behind* the light panels' front edge deliberately: a −z-facing
surface is only lit by samples shallower than it, so spheres in front of the
panels present an entirely unlit face to the camera.

Three lights (`LIGHT_PANELS`), 28 strata total: left wall warm (visible), right
wall cool blue (visible), plus an **invisible** warm fill near the open face —
the only light in front of the spheres and thus the only one lighting their
camera-facing sides. Visible panels stand `LIGHT_PANEL_STANDOFF` voxels proud
and also emit from their four side edges.

Key config (all in the CONFIG block at the top of `render_fpga.py`):
`GRID=128 WIDTH=HEIGHT=1024 MAX_STEPS=400 BATCH_RAYS=1024 LIGHT_SAMPLES=2
ENABLE_SHADOWS=True SHADOW_JITTER=True SHADOW_FILTER_RADIUS=12
SHADOW_FILTER_PLANE_TOL=0.5 SHADOW_ORIGIN_BIAS=0.05 MIRROR_BOUNCES=6
CONTRAST=1.4 SHADE_FLOOR=0.08 USE_BULK_PIPES=True`

`USE_BULK_PIPES=False` falls back to the original per-ray wire path — slow, but
useful to isolate whether a problem is in the bulk datapath.

---

## 9. Performance

Measured history: 5 min (per-ray wires) → **6.9 s** (bulk pipes, 512², no
shadows) → ~37 s (1024², 128³) → ~400 s (with shadows, before batch 1024).

Host work at 1024² after numpy vectorisation: **~2.6 s** total
(`dda_init_batch` 0.64 s, shading 1.31 s, normalise 0.27 s, PNG 0.12 s;
`make_scene`/`pack_scene` ≈ 0). `pack_scene` is a single `np.packbits` —
the grid's C-order flattening *is* the hardware's linear address.

**Shadows dominate**: ~26 shadow rays per pixel (pruning only removes ~3% in an
enclosed box) ≈ 27 M rays at 1024². Cost is per-batch USB round trips, not
tracing. BRAM ≈ 81/105 blocks.

Remaining levers, in order of value:
1. **Decouple shadow sample count from shading sample count** — shading needs no
   rays, so keep 28 samples for `N·L` and use ~4–6 jittered samples for
   visibility. Cuts rays ~4× at the source. *This is the biggest remaining win.*
2. Overlap host prep with USB (thread double-buffering the batches).
3. Raise `SHADOW_FILTER_RADIUS` (cost is linear now, filter is separable).

---

## 10. Verification suite — run after every change

```
C:\GitHub_Projects\ASIC\ttsky-ASIC\fpga-venv\Scripts\python.exe host\tests\<name>.py
```

| test | asserts |
|---|---|
| `test_vectorised.py` | vectorised == scalar reference exactly, incl. fixed-point timers to the LSB; `pack_scene` vs an explicit bit loop |
| `test_host_offline.py` | full render through the RTL model; pack round-trip; all materials visible; no timeouts; lamp is brightest |
| `check_bulk.py` | job/result byte alignment, pipe block multiples, FIFO capacity vs `BATCH_RAYS` (deadlock margin) |
| `test_framing.py` | result-record framing recovery: aligned / dropped word / inserted word / unframeable |
| `test_light_weight.py` | each light's contribution is independent of its sample count |
| `test_filter.py` | coplanar surfaces smooth, terrace steps and face changes do not; mean preserved |
| `test_specular.py` | Blinn-Phong peaks at the mirror direction and falls off; highlight keeps the lamp's colour; `spec=None` is bit-identical to the old path |
| `test_mirror.py` | reflection flips only the face axis; throughput decay and bounce cap both bound the loop |
| `test_shadows.py [res]` | every shadow ray through the RTL model: no self-shadowing, both light-reaching paths, occlusion happens, penumbra exists, filter smooths |
| `test_render_mirror.py [res]` | end-to-end render with bounces; throughput never exceeds 1 |

Also present: `bench_host.py` (timing breakdown), `preview.py` (per-material
illumination stats), `sweep_*.py` / `diag_*.py` (one-off investigations, kept
because their measurement approach is often reusable).

`test_shadows.py` and `test_render_mirror.py` take a resolution argument;
they trace every ray in pure Python, so keep it ≤ 224.

---

## 11. State / next steps

Everything above is implemented and passing. The board needs the **RTF5**
bitstream (deeper FIFOs); the user rebuilds in Vivado. Host-side work needs no
rebuild.

Open items:
- Shadow render is ~400 s; item 1 in §9 is the fix.
- Faint dark seams on sphere silhouettes: single-voxel side facets whose true
  normal points away from the light. Geometrically correct, softened by
  `SHADE_FLOOR`. Would need smoothed normals (estimated from neighbouring
  voxels) to remove properly — that would also fix faceted specular highlights
  on curved surfaces.
- 256³ is **not** possible on this board (needs ~5× the device's BRAM) and would
  also break the byte-aligned WireIn coordinate slots (`COORD_REG_BITS` 9 > 8).
  Would need the A200 variant or a sparse scene representation.
