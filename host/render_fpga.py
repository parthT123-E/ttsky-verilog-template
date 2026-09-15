#!/usr/bin/env python3
"""Host-side renderer for the XEM7310 voxel raytracer (xem7310_raytracer_top).

Flow:
  1. Open the first attached Opal Kelly device and configure it with the
     .bit file given on the command line.
  2. Build a 32x32x32 one-bit-per-voxel scene (a sphere) and stream it
     through BTPipeIn 0x80 as exactly 4096 bytes.
  3. Fire one ray job per pixel through the WireIns + TriggerIn 0x40 bit 2,
     wait for each result, and pop it with TriggerIn 0x40 bit 3.
  4. Shade each hit (Lambert lighting x per-material albedo) and write the
     image to a truecolour PNG.

Colour is entirely host-side: the hardware returns the hit voxel's
coordinates, so the host looks the material up in a parallel material grid.
The FPGA stores only 1-bit occupancy and needs no changes to support this.

Usage: edit the CONFIG block below, then run:
  python render_fpga.py

Requires the Opal Kelly FrontPanel Python API ("import ok") on sys.path.
No other third-party dependencies (PNG is written with stdlib zlib/struct).
"""

import datetime
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

# =============================================================================
# CONFIG — edit these directly
# =============================================================================
# Paths are resolved relative to this script's directory, not the shell's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BITFILE   = os.path.join(_HERE, "xem7310_raytracer_top.bit")
WIDTH     = 1024
HEIGHT    = 1024
OUT_PNG   = os.path.join(_HERE, f"sphere_{_TIMESTAMP}.png")

# Voxel grid size. Must match ADDR_COORD_BITS in xem7310_raytracer_top.sv
# (GRID == 2**ADDR_COORD_BITS) and the regenerated blk_mem_gen_0 IP depth.
GRID = 128                # voxels per side

# Per-ray step budget (10-bit hardware field, so 1..1023). Worst-case
# traversal scales with the grid diagonal (up to ~3*GRID steps for an
# axis-aligned DDA), so derive it from GRID and clamp to the field limit.
# At GRID=128 this saturates at 1023 -- acceptable, since such a ray has
# effectively crossed the whole scene already.
MAX_STEPS = min(1023, 3 * GRID + 16)

# Stream ray jobs/results over the bulk pipes instead of one ray per set of
# USB round trips. The wire path costs ~4 USB transactions per ray (~1.15 ms
# measured) against ~5 us of actual tracing, so this is the difference between
# a latency-bound and a bandwidth-bound render. Set False to fall back to the
# original per-ray wire path (much slower, but simpler to debug).
USE_BULK_PIPES = True
# Rays per bulk transfer. Bigger batches amortise the ~4 USB round trips each
# one costs, which dominate once shadows push the ray count into the tens of
# millions. Must keep job bytes a multiple of PIPE_BLOCK_SIZE
# (BATCH_RAYS % 32 == 0) and stay within MAX_BATCH_RAYS below.
BATCH_RAYS = 1024

# Light source: a square panel of EMISSIVE voxels set flush into the middle of
# the left wall (x = 0). "Emissive" means the panel is drawn at its own colour
# and never shaded -- a lamp glows regardless of where anything else is, and
# shading it diffusely would actually render it black, since its outward
# normal points away from the light it contains.
#
# Because the panel is real geometry in the voxel grid, the FPGA sees it as
# solid and rays terminate on it, so it is genuinely visible in the render
# rather than being an overlay drawn afterwards.
#
# It is an AREA light: illumination is averaged over LIGHT_SAMPLES^2 points
# spread across the panel face, so nearby surfaces fall off smoothly instead
# of pivoting about one point. This is host-side only, but it multiplies the
# per-pixel shading work: 1 degenerates to a point light, 2 (4 samples) is a
# good balance, 4 (16 samples) is noticeably slower.
LIGHT_SAMPLES    = 2                   # N x N sample points across each panel
LIGHT_INTENSITY  = 1.0                 # diffuse term scale, 0..1 typical

# Cast shadows by tracing a second ray from each hit toward each light sample.
# Cost is one extra ray per (hit, sample) pair that actually faces the light --
# samples behind the surface are pruned first, since they contribute nothing
# whether occluded or not. Soft shadows fall out for free: a point that sees 9
# of a panel's 12 samples receives 9/12 of its light, which is the penumbra.
ENABLE_SHADOWS = True

# Jitter each pixel's light sample positions within their stratum instead of
# using one shared set of points. Costs no extra rays. Without it, every pixel
# samples the SAME spots, so each sample's hard shadow boundary is a clean
# shared contour and the panel's N samples show up as N concentric rings.
# Jittering decorrelates them, turning those rings into fine noise.
SHADOW_JITTER = True
JITTER_SEED = 12345      # fixed so renders are reproducible

# Radius, in PIXELS, of the spatial filter applied to the VISIBILITY term
# (0 disables). Jitter trades banding for noise; this removes the noise.
# Because only visibility is filtered, albedo, normals and the geometric N.L
# term stay perfectly sharp. Filtering before jitter would have been useless:
# it would just have blurred the rings.
#
# The residual error is VOXEL-scale, not pixel-scale: shadow rays leaving
# different points of one voxel usually traverse the same voxel sequence, so
# a whole voxel shares an error. The radius therefore has to span several
# voxels to average that out -- at 1024x1024 over a 128^3 grid a voxel is ~8
# pixels, so ~12 covers three of them. The filter is separable, so cost grows
# as 2*(2R+1) rather than (2R+1)^2.
SHADOW_FILTER_RADIUS = 12
# Neighbours are mixed only when they lie on the SAME PLANE: identical entry
# face, and offset along that face's normal within this many voxels.
#
# Coplanarity is the right test, not proximity. On a wall, neighbouring voxels
# sit in one plane and should be averaged freely however far apart they are --
# that is what smooths the penumbra. On a sphere, adjacent voxels step ALONG
# the normal onto a different terrace, and averaging those would destroy the
# per-voxel detail. A plain distance test cannot tell the two apart.
SHADOW_FILTER_PLANE_TOL = 0.5

# How far along the surface normal a shadow ray starts, in voxels. It only has
# to be enough to land in the neighbouring (empty) voxel, so a small value
# keeps the ray's origin on the true surface. Using a whole voxel instead
# would put every pixel covering a given voxel at the SAME start point -- at
# 1024x1024 over a 128^3 grid that is ~64 pixels sharing one origin, which
# shows up as blocky shadow error no filter of sane radius can remove.
SHADOW_ORIGIN_BIAS = 0.05

# The raw Lambertian brightness of the solid (hit) pixels is linearly
# stretched so the shaded region always uses the full available range,
# regardless of how bright/dim the lighting geometry happens to make the
# scene. CONTRAST > 1 then spreads that normalized range further away from
# mid-gray for extra punch; 1.0 leaves the plain range-stretch alone.
# Emissive surfaces are excluded from this stretch (see normalize_image).
CONTRAST        = 1.4

# Brightness the *darkest* shaded pixel maps to, as a fraction of its own
# albedo (the range becomes [SHADE_FLOOR, 1.0] instead of [0.0, 1.0]).
#
# A voxel sphere has occasional single-voxel side facets whose true normal
# points away from the light -- geometrically correct, but they clamp to
# zero illumination and punch pure-black specks through a lit surface. A
# small floor keeps those faces as a dark TINT of the surface colour rather
# than holes, without lifting them enough to read as lit. Keep this low;
# much above ~0.15 the unlit faces start competing with genuine shading.
#
# Note this cannot be done with a plain additive ambient term: a uniform
# offset applied before the min/max stretch is simply normalized back out.
SHADE_FLOOR     = 0.08

# Materials. Index 0 is empty space; any nonzero index is solid, so the 1-bit
# occupancy bitmap the FPGA needs is simply (material != 0). Storing colour as
# a per-voxel INDEX rather than an RGB triple keeps the grid compact and
# mirrors how a hardware palette would work if material ever moves into the
# voxel RAM.
MAT_EMPTY        = 0
MAT_SPHERE_BIG   = 1
MAT_SPHERE_SMALL = 2
MAT_WALL         = 3
MAT_CHECK_LIGHT  = 4
MAT_CHECK_DARK   = 5
MAT_LIGHT        = 6
MAT_LIGHT_COOL   = 7
MAT_LIGHT_PURE   = 8
MAT_MIRROR       = 9

# Per-material albedo (linear RGB, each channel 0..1), indexed by material.
PALETTE = [
    (0.00, 0.00, 0.00),   # 0 empty space   -- never shaded
    (0.92, 0.30, 0.24),   # 1 large sphere  -- warm red
    (0.28, 0.58, 0.95),   # 2 small sphere  -- blue
    (0.82, 0.80, 0.74),   # 3 walls         -- warm neutral grey
    (0.93, 0.93, 0.93),   # 4 checker light -- white
    (0.05, 0.05, 0.05),   # 5 checker dark  -- black (not pure 0, so the
                          #                    checker still shows shading)
    (1.00, 0.96, 0.88),   # 6 warm lamp     -- warm white, emissive
    (0.85, 0.92, 1.00),   # 7 cool lamp     -- barely-blue white, emissive.
                          #   This is only how the panel LOOKS; the colour it
                          #   casts is LIGHT_COOL, set independently below.
    (1.00, 1.00, 1.00),   # 8 neutral lamp  -- pure white, emissive
    (0.97, 0.98, 1.00),   # 9 mirror        -- near-white; tints each bounce
]

# Fully reflective materials: material -> reflectivity (0..1). A ray landing
# on one is REFLECTED rather than shaded, and the pixel keeps following it.
#
# Reflectivity below 1 is what makes the recursion terminate on its own: each
# bounce multiplies the accumulated throughput, so the contribution decays
# geometrically and eventually cannot change an 8-bit pixel. MIRROR_BOUNCES is
# the hard backstop for the pathological case (a ray trapped between two
# parallel mirrors); MIRROR_MIN_THROUGHPUT is the adaptive one that usually
# fires first.
MIRROR = {
    MAT_MIRROR: 0.90,
}
MIRROR_BOUNCES = 64
MIRROR_MIN_THROUGHPUT = 1.0 / 255.0     # below one 8-bit step: invisible

# An unresolved mirror ray (open-face escape, step timeout, or trapped at
# the bounce cap) is shown as its own accumulated throughput tint rather
# than flat black -- see `ghost`/`trapped` below. This extra factor just
# knocks that tint down a bit so a bare mirror surface reads as a dim
# silvery void rather than competing with the objects it reflects.
MIRROR_VOID_DARKEN = 0.4
REFLECTIVITY = np.zeros(len(PALETTE), dtype=np.float64)
for _m, _r in MIRROR.items():
    REFLECTIVITY[_m] = _r

# Materials drawn at full brightness with no lighting calculation, and kept
# out of the illumination normalisation so a bright lamp cannot compress the
# rest of the image.
EMISSIVE = frozenset({MAT_LIGHT, MAT_LIGHT_COOL, MAT_LIGHT_PURE})

# Checker square size in voxels, for the floor and roof planes. Derived from
# GRID so the board always shows 8 squares per side whatever the grid size.
CHECKER_SIZE = max(1, GRID // 8)

# Width in voxels of the matte trim painted along each mirror-to-mirror
# edge of the box (see make_scene). Derived from GRID like CHECKER_SIZE so
# it scales with the grid instead of becoming vanishingly thin or
# overwhelming at a different resolution.
EDGE_WIDTH = max(1, GRID // 64)

# Camera position and aim point, as fractions of GRID so both scale with
# grid size. The box is fully enclosed now (including the front face at
# z=0), so the camera sits inside it rather than outside looking in through
# what used to be an open face -- here, tucked in the top-right-front
# corner. It is aimed at the sphere cluster rather than the box centre, and
# the cluster itself is off-axis (the two spheres are not symmetric about
# any diagonal), so the resulting view angle is a genuinely skewed one, not
# a tidy 45 degrees on any axis.
CAMERA_EYE = (GRID * 0.90, GRID * 0.90, EDGE_WIDTH + 6.0)
CAMERA_TARGET = (GRID * 0.47, GRID * 0.42, GRID * 0.66)


def _camera_basis():
    """Orthonormal (forward, right, up) for CAMERA_EYE looking at
    CAMERA_TARGET, world up = +y. Computed once at import time so the
    scalar and vectorised ray generators below share the exact same
    numbers (see test_vectorised.py)."""
    forward = np.asarray(CAMERA_TARGET, dtype=np.float64) - np.asarray(
        CAMERA_EYE, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


CAMERA_FORWARD, CAMERA_RIGHT, CAMERA_UP = _camera_basis()

# Vertical field of view. Wider spreads more of the scene across the same
# frame, so each object (spheres included) covers a smaller fraction of the
# image -- the lever for making things read as farther away without
# physically moving them, at the cost of more wide-angle edge distortion
# given how close the camera now sits to the walls.
CAMERA_FOV_Y_DEG = 80.0

# Light panel geometry, in voxel coordinates. Derived once and used by BOTH
# the scene builder (which paints the emissive voxels) and the shading (which
# samples points across the same faces), so the visible lamps and the light
# they cast can never disagree about where they are.
#
# Each light is a flat square:
#   axis    0 -> lies in an x plane, spanning y/z (the side walls)
#           2 -> lies in a z plane, spanning x/y (facing the camera)
#   plane   voxel index of that plane
#   inward  +1 if it radiates toward increasing axis, -1 toward decreasing
#   cu, cv  centre on the two in-plane axes (y,z for axis 0; x,y for axis 2)
#   half    half-extent; the square spans 2*half voxels -- an EVEN count,
#           which is what centres exactly on an even-sized wall
#   visible whether it is painted into the voxel grid as an emissive material
#   colour  linear RGB the light emits; it MULTIPLIES the diffuse term, so a
#           tinted light casts that tint onto every surface it reaches
#   mat     which emissive material to paint when visible
#
# A light's GEOMETRY and its ILLUMINATION are independent. visible=True makes
# it emissive voxels the FPGA can hit -- a lamp you can see, which necessarily
# occludes whatever is behind it. visible=False contributes exactly the same
# light while existing only as sample points on the host, like a classic point
# light: nothing to hit, nothing to block the view.
#
# Adding lights costs nothing on the FPGA -- the hardware only solves
# visibility; lighting is host-side arithmetic over this list.
LIGHT_WARM    = (1.00, 0.94, 0.82)   # slightly warm white
LIGHT_COOL    = (0.45, 0.68, 1.00)   # cool blue
# Equal in all three channels, so it scales every surface's own albedo without
# shifting its hue -- pure brightness, no colour cast.
LIGHT_NEUTRAL = (1.00, 1.00, 1.00)

# How far the wall panels sit proud of their wall, in voxels. 0 mounts them
# flush; 1 stands them one voxel into the room, so their side edges catch the
# view and they read as fittings on the wall rather than holes in it. The
# emitting face moves out with them, which also lifts the wall they are
# mounted on out of its own shadow line.
LIGHT_PANEL_STANDOFF = max(1, GRID // 32)

LIGHT_PANELS = (
    dict(axis=0, plane=LIGHT_PANEL_STANDOFF,                 # left wall: warm
         inward=+1, visible=True,
         cu=GRID // 2, cv=GRID // 2, half=GRID // 6,
         colour=LIGHT_WARM, mat=MAT_LIGHT),
    dict(axis=0, plane=GRID - 1 - LIGHT_PANEL_STANDOFF,      # right wall: cool
         inward=-1, visible=True,
         cu=GRID // 2, cv=GRID // 2, half=GRID // 6,
         colour=LIGHT_COOL, mat=MAT_LIGHT_COOL),
    # Invisible fill light near the open face, where the original point light
    # sat. It is the only light IN FRONT of the spheres, so it is the one that
    # illuminates their camera-facing sides at all -- the wall panels sit
    # deeper than the spheres and can only rim-light them. Left invisible so
    # it lights the scene without hanging a bright square in the middle of it.
    # Swap to colour=LIGHT_NEUTRAL, mat=MAT_LIGHT_PURE for an untinted fill.
    dict(axis=2, plane=int(GRID * 0.10), inward=+1, visible=False,
         cu=int(GRID * 0.55), cv=int(GRID * 0.45),
         half=max(2, GRID // 16),
         colour=LIGHT_WARM, mat=MAT_LIGHT),
)


def panel_voxel(panel, u, v):
    """(x, y, z) of the in-plane cell (u, v) of a panel."""
    axis = panel["axis"]
    if axis == 0:
        return (panel["plane"], u, v)     # u = y, v = z
    if axis == 1:
        return (u, panel["plane"], v)     # u = x, v = z
    return (u, v, panel["plane"])         # u = x, v = y


def light_strata():
    """Every panel's emitting surface, divided into STRATA.

    A stratum is a patch of the panel: a centre, the light's colour, and the
    two in-plane half-extents it spans. Sampling at the centre reproduces the
    old fixed-point behaviour; sampling at a random position inside the patch
    (see jitter_strata) is what breaks up shadow banding.

    Voxel i spans [i, i+1), so a panel radiating toward increasing axis emits
    from the plane i+1 and one radiating the other way from i. Using the face
    rather than the voxel centre keeps the light just inside the room instead
    of buried in the surface.

    A panel mounted proud of its wall is a BOX, not a plane: it glows from its
    four side edges as well as its front. Those edges sit right against the
    wall behind it, so they are what lets a lamp light its own wall -- sampling
    only the front face leaves that wall lit almost entirely by the other
    lamps. Edges are sampled at mid-thickness, and only for panels that are
    real geometry; an invisible light has no edges to glow.

    Every sample is an omnidirectional point source -- there is no orientation
    term -- so light does leave a panel in all directions, including back
    toward its own wall."""
    strata = []
    n = max(1, LIGHT_SAMPLES)
    for panel in LIGHT_PANELS:
        first = len(strata)
        axis = panel["axis"]
        plane = float(panel["plane"]) + (1.0 if panel["inward"] > 0 else 0.0)
        half = panel["half"]
        cu, cv = panel["cu"], panel["cv"]
        colour = np.array(panel["colour"], dtype=np.float64)
        # Unit vectors of the panel's two in-plane axes, in world terms.
        if axis == 0:      # x plane: u = y, v = z
            eu, ev = np.array((0., 1., 0.)), np.array((0., 0., 1.))
        elif axis == 1:    # y plane: u = x, v = z
            eu, ev = np.array((1., 0., 0.)), np.array((0., 0., 1.))
        else:              # z plane: u = x, v = y
            eu, ev = np.array((1., 0., 0.)), np.array((0., 1., 0.))
        step = half / n    # half-extent of one stratum along an in-plane axis

        def add(a, u, v, ju, jv):
            if axis == 0:
                pos = (a, u, v)
            elif axis == 1:
                pos = (u, a, v)
            else:
                pos = (u, v, a)
            strata.append({"pos": np.array(pos, dtype=np.float64),
                           "colour": colour.copy(), "ju": ju, "jv": jv})

        # Front face, split into n x n patches.
        for i in range(n):
            fu = (i + 0.5) / n * 2.0 - 1.0
            for j in range(n):
                fv = (j + 0.5) / n * 2.0 - 1.0
                add(plane, cu + fu * half, cv + fv * half, eu * step, ev * step)

        # Four side edges, at the panel's mid-thickness. Each edge patch is a
        # line, so it can only be jittered ALONG the edge.
        if panel.get("visible", True):
            mid = float(panel["plane"]) + 0.5
            zero = np.zeros(3)
            for i in range(n):
                t = (i + 0.5) / n * 2.0 - 1.0
                add(mid, cu - half, cv + t * half, zero, ev * step)
                add(mid, cu + half, cv + t * half, zero, ev * step)
                add(mid, cu + t * half, cv - half, eu * step, zero)
                add(mid, cu + t * half, cv + half, eu * step, zero)

        # Normalise this panel's samples by ITS OWN count, so a light's
        # brightness depends only on its colour and geometry -- never on how
        # finely it happens to be sampled. Dividing by the global sample count
        # instead would silently re-weight every light whenever one of them
        # gains or loses samples.
        count = len(strata) - first
        if count:
            for s in strata[first:]:
                s["colour"] /= count
    return strata


def jitter_strata(stratum, count, rng):
    """`count` sample positions inside one stratum, one per surface point.

    Stratified jitter: the panel is already divided into patches, and each
    point picks a random spot within its patch. Sample count and ray count are
    unchanged -- only the correlation between pixels is broken. That is what
    turns banding (every pixel sampling the SAME points, so each shadow
    boundary is a clean shared contour) into fine noise."""
    pos = stratum["pos"][None, :]
    if not SHADOW_JITTER:
        return np.repeat(pos, count, axis=0)
    ru = rng.uniform(-1.0, 1.0, count)[:, None]
    rv = rng.uniform(-1.0, 1.0, count)[:, None]
    return pos + ru * stratum["ju"][None, :] + rv * stratum["jv"][None, :]


def light_sample_points():
    """Stratum centres in the flat (x, y, z, r, g, b) form used by the
    unjittered shading path and the RTL-model tests."""
    return [tuple(s["pos"]) + tuple(s["colour"]) for s in LIGHT_STRATA]


LIGHT_STRATA = light_strata()
LIGHT_POINTS = light_sample_points()

# Rays that hit nothing. TIMEOUT is lifted slightly off black so an exhausted
# step budget stays visually distinguishable from empty background.
COLOR_BACKGROUND = (0.00, 0.00, 0.00)
COLOR_TIMEOUT    = (0.05, 0.05, 0.05)

# =============================================================================
# FrontPanel endpoint map (must match xem7310_raytracer_top.sv)
# =============================================================================
WI_JOB_XYZ   = 0x00  # [5:0] x, [13:8] y, [21:16] z (starting voxel, integer)
WI_JOB_META  = 0x01  # [9:0] max_steps, [10] sx, [11] sy, [12] sz, [26:13] pixel_id
WI_NEXT_X    = 0x02
WI_NEXT_Y    = 0x03
WI_NEXT_Z    = 0x04
WI_INC_X     = 0x05
WI_INC_Y     = 0x06
WI_INC_Z     = 0x07
WI_PIPE_CTRL = 0x08  # [0] route results to PipeOut 0xA0

TI_CONTROL   = 0x40
TI_RESET        = 0
TI_SCENE_BEGIN  = 1
TI_JOB_SUBMIT   = 2
TI_RESULT_POP   = 3
TI_CLEAR_ERRORS = 4
TI_SCENE_ABORT  = 5

PIPE_SCENE   = 0x80
PIPE_JOBS    = 0x81  # BTPipeIn: stream of JOB_WORDS-word ray jobs
PIPE_RESULTS = 0xA0  # PipeOut:  stream of RES_WORDS-word results

# Bulk job/result record layout (must match xem7310_raytracer_top.sv).
JOB_WORDS = 8   # xyz, meta, next_x/y/z, inc_x/y/z
RES_WORDS = 4   # info, xy, z_steps, tag  -- 16 bytes keeps pipe alignment legal

# Mirrors of the RTL FIFO depths. A batch's jobs AND its results must both fit
# whole: the host writes every job before reading any result, so if the result
# FIFO overflowed the tracer would stall, jobs would stop draining, and the
# blocking job write would never return -- a deadlock, not a wrong pixel.
JOB_FIFO_DEPTH = 8192
RES_FIFO_DEPTH = 8192
MAX_BATCH_RAYS = min(JOB_FIFO_DEPTH // JOB_WORDS, RES_FIFO_DEPTH // RES_WORDS)

# The tag word is {RES_TAG_MAGIC, 24-bit sequence}. The magic byte makes
# record boundaries self-identifying: no other result word can set bits
# [31:24], so framing can be established without guessing.
RES_TAG_MAGIC = 0xA5
RES_SEQ_MASK  = 0xFFFFFF


def is_tag(word):
    return (word >> 24) == RES_TAG_MAGIC


def tag_seq(word):
    return word & RES_SEQ_MASK

WO_STATUS    = 0x20
WO_PROGRESS  = 0x21
WO_RESULT_INFO = 0x22  # [13:0] pixel_id, [14] hit, [15] timeout, [18:16] face_id
WO_RESULT_XY   = 0x23  # [15:0] hit_x, [31:16] hit_y
WO_RESULT_ZS   = 0x24  # [15:0] hit_z, [31:16] steps
WO_BUILD_ID    = 0x25
WO_PIPE_STATUS = 0x26  # [15:0] result words ready, [31:16] job words queued

# Bumped whenever the endpoint map OR a payload format changes; see
# wo_build_id in the RTL. RTF1 = wires + scene pipe, RTF2 = adds the bulk
# job/result pipes, RTF3 = tagged result records + fixed PipeOut read timing,
# RTF4 = 128^3 grid, RTF5 = deeper job/result FIFOs for 1024-ray batches.
EXPECTED_BUILD_ID = 0x52544635  # ASCII "RTF5"

# Width of the hardware pixel_id field (PIXEL_ID_WIDTH in the RTL). Renders
# larger than 2**PIXEL_ID_BITS pixels simply wrap this counter -- see the note
# in trace_ray() for why that is safe.
PIXEL_ID_BITS = 14
PIXEL_ID_MASK = (1 << PIXEL_ID_BITS) - 1

# wo_status bit positions
ST_RST_N          = 1 << 0
ST_SCENE_LOADING  = 1 << 1
ST_SCENE_LOADED   = 1 << 2
ST_JOB_PENDING    = 1 << 4
ST_JOB_READY      = 1 << 5
ST_RESULT_VALID   = 1 << 6
ST_TRACER_IDLE    = 1 << 7
ST_PIPE_ERROR     = 1 << 11
ST_JOB_OVERFLOW   = 1 << 12
ST_RESULT_UNDERFLOW = 1 << 13
ST_JOB_PIPE_ERROR = 1 << 14
ST_RES_PIPE_ERROR = 1 << 15

# Any of these latching means the datapath dropped or mis-sequenced data.
ST_ANY_ERROR = (ST_PIPE_ERROR | ST_JOB_OVERFLOW | ST_RESULT_UNDERFLOW
                | ST_JOB_PIPE_ERROR | ST_RES_PIPE_ERROR)

STATUS_BITS = {
    0: "rst_n", 1: "scene_loading", 2: "scene_loaded", 3: "fifo_busy",
    4: "job_pending", 5: "job_ready", 6: "result_valid", 7: "tracer_idle",
    8: "load_ready", 9: "pipe_ready", 10: "fifo_full",
    11: "pipe_error", 12: "job_overflow", 13: "result_underflow",
    14: "job_pipe_error", 15: "res_pipe_error",
}

SCENE_BYTES = GRID ** 3 // 8   # 32768 at 64^3
COORD_BITS = GRID.bit_length() - 1  # address bits per axis (GRID is a power of 2)
COORD_MASK = GRID - 1               # per-axis coordinate mask
# BTPipeIn block size in bytes. MUST match BLOCK_WORDS in
# xem7310_raytracer_top.sv (PIPE_BLOCK_SIZE == BLOCK_WORDS * 4): okBTPipeIn
# commits a whole block once the gateware asserts ep_ready, so the RTL sizes
# its streaming FIFO to guarantee room for exactly this much data. Raising
# this without raising BLOCK_WORDS causes silently dropped voxels.
PIPE_BLOCK_SIZE = 1024    # = BLOCK_WORDS(256) * 4

# =============================================================================
# Fixed-point contract for the DDA timers
# -----------------------------------------------------------------------------
# From the RTL (axis_choose.sv, step_update.sv):
#   * next_x/y/z and inc_x/y/z are UNSIGNED 32-bit values.
#   * Each step the core picks the axis with the smallest next_* (unsigned
#     compare), moves the voxel index by +/-1 on that axis (sx/sy/sz: 1 = +1),
#     and does next_* += inc_* on that axis. No saturation, no scaling.
#
# The hardware therefore has NO intrinsic fixed-point format: only relative
# magnitudes matter, and the host is free to choose the scale. We use Q16.16
# on the ray parameter t measured in voxel units (t advances by 1.0 when the
# ray travels one voxel along its own direction):
#   inc_axis  = round((1 / |dir_axis|) * 2**16)
#   next_axis = round(t_to_first_boundary_on_axis * 2**16)
#
# Overflow check: an axis can step at most ~GRID+1 times before the ray
# leaves the grid (out-of-bounds terminates the ray), and we treat any
# |dir_axis| < MIN_AXIS_DIR as zero, so
#   next_max ~= (GRID+1) * FP_ONE / MIN_AXIS_DIR ~= 33 * 2^16 * 2^10 < 2^32.
# An axis with (near-)zero direction gets next = 0xFFFFFFFF and inc = 0 so
# it is never selected.
#
# If you change the RTL timer semantics, this block is the only thing that
# needs to change on the host.
# =============================================================================
FP_ONE = 1 << 16
TIMER_MAX = 0xFFFFFFFF
MIN_AXIS_DIR = 1.0 / 1024.0


def dda_init(origin, direction):
    """Set up an Amanatides-Woo DDA traversal for one ray.

    origin/direction are 3-tuples of floats in voxel units (grid spans
    [0, GRID) on each axis). direction need not be normalized.

    Returns None if the ray misses the grid, else a dict with the exact
    field values to write to the WireIns.
    """
    length = math.sqrt(sum(c * c for c in direction))
    d = [c / length for c in direction]

    # Slab intersection with the grid AABB [0, GRID]^3.
    t_enter, t_exit = 0.0, float("inf")
    for a in range(3):
        if abs(d[a]) < 1e-12:
            if not (0.0 <= origin[a] <= GRID):
                return None
            continue
        t0 = (0.0 - origin[a]) / d[a]
        t1 = (GRID - origin[a]) / d[a]
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
    if t_enter >= t_exit:
        return None

    # Entry point, nudged slightly inside so floor() lands in a valid voxel.
    t_start = t_enter + 1e-6
    p = [origin[a] + d[a] * t_start for a in range(3)]
    ivox = [min(GRID - 1, max(0, int(math.floor(p[a])))) for a in range(3)]

    next_fp = [TIMER_MAX] * 3
    inc_fp = [0] * 3
    signs = [1, 1, 1]  # 1 = +1 step, 0 = -1 step (don't-care on dead axes)
    for a in range(3):
        if abs(d[a]) < MIN_AXIS_DIR:
            continue  # axis never steps: next stays at TIMER_MAX, inc = 0
        if d[a] > 0:
            signs[a] = 1
            t_boundary = (ivox[a] + 1 - p[a]) / d[a]
        else:
            signs[a] = 0
            t_boundary = (ivox[a] - p[a]) / d[a]
        inc_fp[a] = min(TIMER_MAX, int(round(FP_ONE / abs(d[a]))))
        next_fp[a] = min(TIMER_MAX, max(0, int(round(t_boundary * FP_ONE))))

    return {
        "voxel": ivox,
        "signs": signs,
        "next": next_fp,
        "inc": inc_fp,
    }


# =============================================================================
# Scene construction and packing
# =============================================================================
def add_sphere(material, fx, fy, fz, fr, mat_id):
    """Fill a solid sphere of material mat_id. Centre and radius are given as
    fractions of GRID so the scene keeps its proportions at any grid size.
    Only the sphere's bounding box is evaluated, not the whole volume."""
    cx, cy, cz = fx * GRID, fy * GRID, fz * GRID
    r = fr * GRID
    z0, z1 = max(0, int(cz - r)), min(GRID, int(cz + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(GRID, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(GRID, int(cx + r) + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    dz = np.arange(z0, z1, dtype=np.float64)[:, None, None] - cz
    dy = np.arange(y0, y1, dtype=np.float64)[None, :, None] - cy
    dx = np.arange(x0, x1, dtype=np.float64)[None, None, :] - cx
    inside = (dz * dz + dy * dy + dx * dx) <= r * r
    material[z0:z1, y0:y1, x0:x1][inside] = mat_id


def make_scene():
    """GRID^3 material grid: material[z][y][x] -> material index (0 = empty).

    Two spheres inside an open-fronted box: checkerboard floor and roof,
    plain left/right walls and a plain back wall. The -z face is left open
    so the camera can see in. Camera looks down +z, so x = right, y = up,
    z = away from the viewer.

    The enclosure spans the full grid rather than being inset, so every ray
    entering the open face terminates on a surface -- an inset box would let
    rays near the frame edge slip through the gap between it and the grid
    boundary and escape, leaving black bands top and bottom.

    This single grid is the source of truth for BOTH the occupancy bitmap
    sent to the FPGA and the host-side colour lookup, so the two can never
    disagree about which voxels are solid.

    Returned as a numpy array indexed [z, y, x] -- the same order as the
    hardware's linear voxel address, which lets pack_scene() below be a single
    packbits call."""
    material = np.zeros((GRID, GRID, GRID), dtype=np.uint8)

    floor_y, roof_y = 0, GRID - 1
    back_z = GRID - 1

    # Floor and roof: mirrors, over the x/z plane.
    material[:, floor_y, :] = MAT_MIRROR
    material[:, roof_y, :] = MAT_MIRROR

    # Side, front and back walls, drawn after the floor/roof so the box gets
    # a clean edge everywhere the planes meet. The front face (z=0) is where
    # the camera used to sit outside looking in; the camera now sits just
    # inside it instead (see camera_ray/camera_rays), so this can close
    # without blocking the view.
    material[:, :, 0] = MAT_MIRROR            # left wall: mirror
    material[:, :, GRID - 1] = MAT_MIRROR     # right wall: mirror
    material[0, :, :] = MAT_MIRROR            # front wall: mirror
    material[back_z, :, :] = MAT_MIRROR       # back wall: mirror

    # Matte trim along every edge where two mirror planes meet, so the box
    # reads as a room with defined corners instead of one unbroken
    # reflective shell.
    near0 = np.arange(GRID) < EDGE_WIDTH
    near1 = np.arange(GRID) >= GRID - EDGE_WIDTH
    edge_z = near0[:, None] | near1[:, None]
    edge_zx = near0[None, :] | near1[None, :] | edge_z
    material[:, floor_y, :][edge_zx] = MAT_WALL
    material[:, roof_y, :][edge_zx] = MAT_WALL
    edge_zy = near0[None, :] | near1[None, :] | edge_z
    material[:, :, 0][edge_zy] = MAT_WALL
    material[:, :, GRID - 1][edge_zy] = MAT_WALL
    edge_yx = near0[None, :] | near1[None, :] | near0[:, None] | near1[:, None]
    material[0, :, :][edge_yx] = MAT_WALL
    material[back_z, :, :][edge_yx] = MAT_WALL

    # Emissive light panels. The span is 2*half voxels (not 2*half+1): an EVEN
    # count is what centres exactly on an even-sized wall. An odd span would
    # always land half a voxel off, leaving one more row on one side.
    for panel in LIGHT_PANELS:
        if not panel.get("visible", True):
            continue        # illuminates, but has no geometry to hit
        half = panel["half"]
        u0, u1 = max(0, panel["cu"] - half), min(GRID, panel["cu"] + half)
        v0, v1 = max(0, panel["cv"] - half), min(GRID, panel["cv"] + half)
        plane, mat = panel["plane"], panel["mat"]
        if panel["axis"] == 0:      # (x,y,z) = (plane, u, v)
            material[v0:v1, u0:u1, plane] = mat
        elif panel["axis"] == 1:    # (x,y,z) = (u, plane, v)
            material[v0:v1, plane, u0:u1] = mat
        else:                       # (x,y,z) = (u, v, plane)
            material[plane, v0:v1, u0:u1] = mat

    # Spheres last, so the enclosure can never clip them.
    #
    # Both sit BEHIND the light panels' front edge. A -z-facing surface is
    # only lit by light samples shallower than itself, so a sphere in front of
    # the panels presents an entirely unlit face to the camera. These depths
    # push each sphere's near surface past the panels' leading sample plane
    # while leaving a few voxels of clearance from the back wall.
    # Large sphere: left of centre, higher, farthest from the camera.
    add_sphere(material, 0.38, 0.52, 0.72, 0.22, MAT_SPHERE_BIG)
    # Small sphere: right, lower, nearer the camera -- the size and depth
    # difference gives the render an obvious sense of scale.
    add_sphere(material, 0.70, 0.34, 0.60, 0.12, MAT_SPHERE_SMALL)
    return material


def pack_scene(material):
    """Pack the material grid's occupancy (material != 0) into SCENE_BYTES
    bytes for BTPipeIn 0x80.

    Word N (little-endian 32-bit) holds voxel addresses 32N..32N+31, LSB
    first, with linear address {z, y, x} where each axis occupies
    COORD_BITS bits. Little-endian words with LSB-first bits collapse to a
    plain LSB-first bitstream over bytes.

    Because GRID is a power of two, that linear address is exactly
    z*GRID^2 + y*GRID + x -- which is the C-order flattening of the
    [z, y, x] array. So the whole pack is one packbits call, with
    bitorder='little' supplying the LSB-first bit placement.
    """
    occupied = (material != MAT_EMPTY).reshape(-1)
    data = np.packbits(occupied, bitorder="little").tobytes()
    assert len(data) == SCENE_BYTES, (len(data), SCENE_BYTES)
    return data


# =============================================================================
# Ray job / result record encoding
# =============================================================================
# Both transports carry identical field layouts -- the WireIn path writes
# these words to 0x00..0x07, the bulk path streams them through PipeIn 0x81 --
# so they are built and parsed in one place.
def job_words(dda, pixel_id, max_steps):
    """The JOB_WORDS words describing one ray job."""
    ivox, signs = dda["voxel"], dda["signs"]
    m = COORD_MASK
    xyz = (ivox[0] & m) | ((ivox[1] & m) << 8) | ((ivox[2] & m) << 16)
    meta = ((max_steps & 0x3FF)
            | (signs[0] << 10) | (signs[1] << 11) | (signs[2] << 12)
            | ((pixel_id & PIXEL_ID_MASK) << 13))
    return (xyz, meta,
            dda["next"][0], dda["next"][1], dda["next"][2],
            dda["inc"][0], dda["inc"][1], dda["inc"][2])


def job_words_batch(batch, pixel_ids, max_steps):
    """Vectorised job_words(): an (N, JOB_WORDS) little-endian uint32 array,
    ready to slice and .tobytes() straight into the job pipe."""
    ivox = batch["voxel"].astype(np.uint32) & np.uint32(COORD_MASK)
    signs = batch["signs"].astype(np.uint32)
    steps = np.asarray(max_steps, dtype=np.uint32) & np.uint32(0x3FF)
    xyz = ivox[:, 0] | (ivox[:, 1] << 8) | (ivox[:, 2] << 16)
    meta = (steps
            | (signs[:, 0] << 10) | (signs[:, 1] << 11) | (signs[:, 2] << 12)
            | ((pixel_ids.astype(np.uint32) & np.uint32(PIXEL_ID_MASK)) << 13))
    return np.stack((xyz, meta,
                     batch["next"][:, 0], batch["next"][:, 1],
                     batch["next"][:, 2],
                     batch["inc"][:, 0], batch["inc"][:, 1],
                     batch["inc"][:, 2]), axis=1).astype("<u4")


def parse_result(info, xy, z_steps):
    """Decode the three result words shared by both transports."""
    return {
        "pixel_id": info & PIXEL_ID_MASK,
        "hit": bool(info & (1 << 14)),
        "timeout": bool(info & (1 << 15)),
        "face": (info >> 16) & 0x7,
        "x": xy & 0xFFFF,
        "y": (xy >> 16) & 0xFFFF,
        "z": z_steps & 0xFFFF,
        "steps": (z_steps >> 16) & 0xFFFF,
    }


# =============================================================================
# FrontPanel device wrapper
# =============================================================================
class RaytracerDevice:
    """Uses the FrontPanel 6.x ('FrontPanel-Platform') Python API: devices are
    opened through okCFrontPanelDevices, and all wire/trigger/pipe traffic
    goes through the classic FPGA data port object."""

    def __init__(self, bitfile):
        try:
            import ok
        except ImportError:
            sys.exit(
                "Could not import the Opal Kelly FrontPanel Python API ('ok').\n"
                "Install the wheel into this venv, e.g.\n"
                "  pip install \"C:\\Program Files\\Opal Kelly\\FrontPanel-Platform"
                "\\API\\Python\\x64\\ok-6.0.0-cp39-abi3-win_amd64.whl\""
            )
        self.ok = ok
        devices = ok.okCFrontPanelDevices()
        self.dev = devices.Open("")  # empty serial = first available device
        if self.dev is None:
            sys.exit(f"No Opal Kelly device found "
                     f"(devices detected: {devices.GetCount()}).")

        info = ok.okTDeviceInfo()
        if self.dev.GetDeviceInfo(info) == ok.ErrorCode.NoError:
            print(f"Opened {info.productName} (serial {info.serialNumber})")

        err = self.dev.ConfigureFPGA(bitfile)
        if err != ok.ErrorCode.NoError:
            sys.exit(f"ConfigureFPGA('{bitfile}') failed: "
                     f"{ok.okCFrontPanel.GetErrorMessage(err)}")
        if not self.dev.IsFrontPanelEnabled():
            sys.exit("FrontPanel is not enabled in this bitfile.")

        # All endpoint access (wires, triggers, pipes) lives on the data port.
        self.fp = self.dev.GetFPGADataPortClassic()
        # Mirrors the RTL's free-running result sequence counter; any gap means
        # results were dropped between the tracer and the host.
        self._res_seq = 0
        # Words read from the result pipe but not yet consumed as whole
        # records, plus the one-time record-framing state (see _frame_results).
        self._res_words = []
        self._res_framed = False
        self.lost_results = 0   # results the transport never delivered

        build_id = self.read_wire(WO_BUILD_ID)
        if build_id != EXPECTED_BUILD_ID:
            def tag(v):
                b = struct.pack(">I", v)
                return b.decode("ascii") if all(32 <= c < 127 for c in b) else "?"
            sys.exit(
                f"Build ID mismatch: bitstream reports 0x{build_id:08x} "
                f"('{tag(build_id)}') but this host expects "
                f"0x{EXPECTED_BUILD_ID:08x} ('{tag(EXPECTED_BUILD_ID)}').\n"
                f"The loaded bitstream is older than this script -- "
                f"re-run synthesis/implementation and reprogram, or check "
                f"BITFILE points at the newest .bit.")
        print(f"Build ID OK: {struct.pack('>I', build_id).decode('ascii')}")

    # --- low-level helpers ---------------------------------------------------
    def ok_error(self, code):
        """Describe a negative FrontPanel return code by its ErrorCode name.

        Pipe calls return a byte count on success and a negative ErrorCode on
        failure, so a bare number like -2 is otherwise easy to misread as a
        length."""
        for name in dir(self.ok.ErrorCode):
            if not name.startswith("_") and \
                    getattr(self.ok.ErrorCode, name, None) == code:
                return f"{name} ({code})"
        return f"unknown error {code}"

    def read_wire(self, addr):
        self.fp.UpdateWireOuts()
        return self.fp.GetWireOutValue(addr) & 0xFFFFFFFF

    def status(self):
        return self.read_wire(WO_STATUS)

    def decode_status(self, status):
        return [name for bit, name in STATUS_BITS.items() if status & (1 << bit)]

    def trigger(self, bit):
        self.fp.ActivateTriggerIn(TI_CONTROL, bit)

    def wait_status(self, mask, value, timeout_s, what):
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.status()
            if (status & mask) == value:
                return status
            if status & ST_ANY_ERROR:
                raise RuntimeError(
                    f"Wrapper error while waiting for {what}: "
                    f"status=0x{status:08x} {self.decode_status(status)}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {what}: "
                    f"status=0x{status:08x} {self.decode_status(status)}")

    # --- protocol steps ------------------------------------------------------
    def reset(self):
        self.trigger(TI_RESET)
        self.wait_status(ST_RST_N | ST_TRACER_IDLE, ST_RST_N | ST_TRACER_IDLE,
                         1.0, "reset release")
        self.trigger(TI_CLEAR_ERRORS)
        self._res_seq = 0   # the RTL counter resets with the design
        self._res_words = []
        self._res_framed = False

    def load_scene(self, scene_bytes):
        assert len(scene_bytes) == SCENE_BYTES
        self.trigger(TI_SCENE_BEGIN)
        self.wait_status(ST_SCENE_LOADING, ST_SCENE_LOADING, 1.0,
                         "scene load to start")

        buf = bytearray(scene_bytes)
        sent = self.fp.WriteToBlockPipeIn(PIPE_SCENE, PIPE_BLOCK_SIZE, buf)
        if sent != SCENE_BYTES:
            raise RuntimeError(f"BTPipeIn transferred {sent} bytes, "
                               f"expected {SCENE_BYTES}")

        self.wait_status(ST_SCENE_LOADED | ST_SCENE_LOADING, ST_SCENE_LOADED,
                         5.0, "scene load to complete")
        voxels = self.read_wire(WO_PROGRESS)  # full 32-bit voxel write count
        if voxels != GRID ** 3:
            raise RuntimeError(
                f"Scene load wrote {voxels} voxels, expected {GRID ** 3}. "
                f"The gateware's grid size does not match this host's GRID="
                f"{GRID} -- check ADDR_COORD_BITS in the RTL and that the "
                f"blk_mem_gen_0 IP was regenerated at depth {GRID ** 3}.")
        print(f"Scene loaded: {voxels} voxels written (expected {GRID ** 3})")

    def trace_ray(self, dda, pixel_id, max_steps, timeout_s=1.0):
        """Submit one ray job, wait for its result, pop it, return the result.

        pixel_id is truncated to the hardware's PIXEL_ID_BITS-wide field, so
        renders with more than 2**PIXEL_ID_BITS pixels wrap it. That is safe
        because the caller places results using its own pixel coordinates and
        never the returned id -- pixel_id only ever serves as a request/reply
        consistency check. Wrapping merely makes it a mod-2**PIXEL_ID_BITS
        check, which stays unambiguous as long as far fewer than that many
        rays are in flight at once (currently exactly one; even fully batching
        the 5 contexts plus the 8-deep result FIFO would only reach ~13)."""
        words = job_words(dda, pixel_id, max_steps)
        for addr, value in zip((WI_JOB_XYZ, WI_JOB_META,
                                WI_NEXT_X, WI_NEXT_Y, WI_NEXT_Z,
                                WI_INC_X, WI_INC_Y, WI_INC_Z), words):
            self.fp.SetWireInValue(addr, value)
        self.fp.UpdateWireIns()
        self.trigger(TI_JOB_SUBMIT)

        self.wait_status(ST_RESULT_VALID, ST_RESULT_VALID, timeout_s,
                         f"result of pixel {pixel_id}")
        info = self.fp.GetWireOutValue(WO_RESULT_INFO)
        xy = self.fp.GetWireOutValue(WO_RESULT_XY)
        zs = self.fp.GetWireOutValue(WO_RESULT_ZS)
        self.trigger(TI_RESULT_POP)

        result = parse_result(info, xy, zs)
        if result["pixel_id"] != (pixel_id & PIXEL_ID_MASK):
            raise RuntimeError(f"Result pixel_id {result['pixel_id']} does not "
                               f"match submitted {pixel_id}")
        return result

    # --- bulk pipe path ------------------------------------------------------
    def set_result_pipe(self, enabled):
        """Route results to PipeOut 0xA0 (True) or the result WireOuts."""
        self.fp.SetWireInValue(WI_PIPE_CTRL, 1 if enabled else 0)
        self.fp.UpdateWireIns()

    def trace_batch(self, words, timeout_s=10.0):
        """Stream one batch of ray jobs and collect their results.

        `words` is an (M, JOB_WORDS) little-endian uint32 array from
        job_words_batch(). Returns an (M', RES_WORDS) uint32 array of raw
        result records in hardware completion order -- which is NOT submission
        order, because up to 5 rays are in flight at once. Callers must place
        pixels using each record's pixel_id, not its position.

        Jobs are padded up to the pipe's block granularity by repeating the
        first job; duplicate results are harmless because they carry the same
        pixel_id and identical data.
        """
        jobs_per_block = PIPE_BLOCK_SIZE // (JOB_WORDS * 4)
        pad = (-len(words)) % jobs_per_block
        if pad:
            words = np.vstack((words, np.repeat(words[:1], pad, axis=0)))
        expected = len(words)

        buf = bytearray(words.tobytes())
        sent = self.fp.WriteToBlockPipeIn(PIPE_JOBS, PIPE_BLOCK_SIZE, buf)
        if sent != len(buf):
            status = self.status()
            raise RuntimeError(
                f"job PipeIn 0x{PIPE_JOBS:02x} write failed: "
                f"{self.ok_error(sent)}, expected {len(buf)} bytes. "
                f"status=0x{status:08x} {self.decode_status(status)}\n"
                f"A Timeout here usually means the gateware never asserted "
                f"ep_ready -- most often because the loaded bitstream lacks "
                f"this endpoint, or its job FIFO is full because results are "
                f"not being drained.")

        records = []      # (K, RES_WORDS) chunks, concatenated at the end
        got_count = 0
        deadline = time.monotonic() + timeout_s
        while got_count < expected:
            status = self.status()
            if status & ST_ANY_ERROR:
                raise RuntimeError(
                    f"Wrapper error draining results: status=0x{status:08x} "
                    f"{self.decode_status(status)}")
            ready_words = self.fp.GetWireOutValue(WO_PIPE_STATUS) & 0xFFFF
            # Pipe transfers must be a whole number of 16-byte (RES_WORDS)
            # units, so always move a multiple of one record's worth.
            n = ready_words // RES_WORDS
            if n:
                out = bytearray(n * RES_WORDS * 4)
                got = self.fp.ReadFromPipeOut(PIPE_RESULTS, out)
                if got != len(out):
                    raise RuntimeError(
                        f"result PipeOut 0x{PIPE_RESULTS:02x} read failed: "
                        f"{self.ok_error(got)}, expected {len(out)} bytes")
                self._res_words.extend(
                    np.frombuffer(out, dtype="<u4").tolist())
                deadline = time.monotonic() + timeout_s

            if not self._res_framed:
                self._frame_results()

            # Consume whole records in bulk and validate their tags as a
            # block: the magic must be present on every one and the sequence
            # numbers must run consecutively.
            avail = len(self._res_words) // RES_WORDS
            take = min(avail, expected - got_count)
            if self._res_framed and take:
                block = np.array(self._res_words[:take * RES_WORDS],
                                 dtype=np.uint32).reshape(take, RES_WORDS)
                del self._res_words[:take * RES_WORDS]
                tags = block[:, 3]
                want = (self._res_seq + np.arange(take)) & RES_SEQ_MASK
                bad = np.nonzero((tags >> 24 != RES_TAG_MAGIC)
                                 | ((tags & RES_SEQ_MASK) != want))[0]
                if bad.size:
                    i = int(bad[0])
                    tag = int(tags[i])
                    raise RuntimeError(
                        f"result framing lost mid-stream at record {i}: tag "
                        f"word 0x{tag:08x} (magic ok={is_tag(tag)}, "
                        f"seq={tag_seq(tag)}), expected seq {int(want[i])}."
                        f"\n{self._dump_words(block.reshape(-1).tolist())}")
                self._res_seq = int((self._res_seq + take) & RES_SEQ_MASK)
                records.append(block)
                got_count += take

            if got_count >= expected or n:
                continue
            # Nothing arrived. If the tracer is idle with no jobs queued and
            # the result FIFO empty, the missing results are never coming --
            # report and return short rather than hanging, so the render can
            # finish and the loss rate is visible.
            pipe = self.fp.GetWireOutValue(WO_PIPE_STATUS)
            drained = (status & ST_TRACER_IDLE) and (pipe & 0xFFFF) == 0 \
                and ((pipe >> 16) & 0xFFFF) == 0
            if drained and time.monotonic() > deadline:
                self.lost_results += expected - got_count
                break
            if time.monotonic() > deadline + timeout_s:
                raise TimeoutError(
                    f"Stalled with {got_count}/{expected} results, "
                    f"{len(self._res_words)} spare words; "
                    f"status=0x{status:08x} {self.decode_status(status)}"
                    f"\n{self._dump_words(self._res_words)}")

        # A batch produces exactly expected*RES_WORDS words and reads move
        # whole records, so nothing should be left over. Anything remaining
        # would be parsed during the NEXT batch and yield results whose
        # pixel_ids belong to this one -- fail loudly instead of leaking.
        if self._res_words:
            raise RuntimeError(
                f"{len(self._res_words)} word(s) left over after a batch -- "
                f"result records are not aligned to the read granularity, so "
                f"they would leak into the next batch."
                f"\n{self._dump_words(self._res_words)}")
        if not records:
            return np.zeros((0, RES_WORDS), dtype=np.uint32)
        return np.vstack(records)

    @staticmethod
    def _dump_words(words, count=12):
        """Hex dump with each word decoded every way a result word can be
        read, so a framing problem is obvious from the output alone."""
        lines = ["  raw result words (idx: hex -> plausible interpretations):"]
        for i, v in enumerate(words[:count]):
            kind = (f"TAG(seq={tag_seq(v)})" if is_tag(v)
                    else f"info(pid={v & PIXEL_ID_MASK},hit={(v >> 14) & 1},"
                         f"face={(v >> 16) & 7}) | "
                         f"pair({v & 0xFFFF},{(v >> 16) & 0xFFFF})")
            lines.append(f"    {i:2d}: {v:08x}  {kind}")
        return "\n".join(lines)

    def _frame_results(self):
        """Establish where result records begin within the word stream.

        Records should start at word 0, but a transport that drops or inserts
        a word at the head of the stream would shift every record by a
        constant. The sequence tag makes that detectable: at the correct
        offset the tag words are consecutive. Detect the offset once, discard
        the stray words, and report loudly -- a nonzero offset is a real
        datapath bug, not something to silently paper over.
        """
        w = self._res_words
        if len(w) < 2 * RES_WORDS:
            return  # need two records before the pattern is unambiguous
        for skew in range(RES_WORDS):
            tags = [w[skew + RES_WORDS * i + 3] for i in range(2)]
            # The magic byte identifies tag words outright, and requiring two
            # consecutive ones rules out a chance match. Anchoring on the
            # magic rather than on the expected value matters: if the
            # transport dropped the head word, record 0 is unrecoverable and
            # the stream legitimately resumes at a later sequence number.
            if not (is_tag(tags[0]) and is_tag(tags[1])):
                continue
            if tag_seq(tags[1]) != (tag_seq(tags[0]) + 1) & RES_SEQ_MASK:
                continue
            lost = (tag_seq(tags[0]) - self._res_seq) & RES_SEQ_MASK
            if skew or lost:
                print(f"\nWARNING: result stream is misframed -- records "
                      f"start {skew} word(s) in"
                      + (f" and {lost} result(s) were lost with the discarded "
                         f"words" if lost else "") +
                      f". Resynchronising to sequence {tag_seq(tags[0])}.\n"
                      f"{self._dump_words(w)}\n")
                del w[:skew]
                self._res_seq = tag_seq(tags[0])
                self.lost_results += lost
            self._res_framed = True
            return
        raise RuntimeError(
            f"Could not establish result record framing: no word offset puts "
            f"tag words (magic 0x{RES_TAG_MAGIC:02x}) on a "
            f"{RES_WORDS}-word grid.\n{self._dump_words(w)}")


# =============================================================================
# Camera and shading
# =============================================================================
def camera_rays(width, height):
    """Directions for EVERY pixel at once, row-major (py major, px minor), as
    an (N, 3) array. Same convention as camera_ray() below."""
    eye = CAMERA_EYE
    fov_y = math.radians(CAMERA_FOV_Y_DEG)
    aspect = width / height
    half_h = math.tan(fov_y / 2)
    px = np.arange(width, dtype=np.float64)
    py = np.arange(height, dtype=np.float64)
    ndc_x = (px + 0.5) / width * 2.0 - 1.0
    ndc_y = 1.0 - (py + 0.5) / height * 2.0   # image row 0 = +y (top)
    cx = np.repeat(ndc_x[None, :] * half_h * aspect, height, axis=0).ravel()
    cy = np.repeat(ndc_y[:, None] * half_h, width, axis=1).ravel()
    cz = np.ones(width * height, dtype=np.float64)
    # Camera-space (right, up, forward) -> world space, via the fixed basis
    # for CAMERA_EYE/CAMERA_TARGET (see _camera_basis).
    dirs = (cx[:, None] * CAMERA_RIGHT + cy[:, None] * CAMERA_UP
            + cz[:, None] * CAMERA_FORWARD)
    return eye, dirs


def dda_init_batch(eye, dirs):
    """Vectorised dda_init() over an (N, 3) array of directions.

    Returns a dict of arrays -- voxel (N,3) int, signs (N,3) bool,
    next/inc (N,3) uint32 -- plus a (N,) bool `valid` marking rays that
    actually enter the grid. Same arithmetic as the scalar version below,
    which stays the readable reference; dda_row() extracts one ray in the
    scalar layout so both paths share these numbers."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    eye_a = np.asarray(eye, dtype=np.float64)

    # Slab intersection with the grid AABB [0, GRID]^3.
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (0.0 - eye_a) / d
        t1 = (GRID - eye_a) / d
    lo = np.minimum(t0, t1)
    hi = np.maximum(t0, t1)
    # An axis with (near) zero direction imposes no bound -- unless the eye is
    # already outside the slab on that axis, in which case the ray never enters.
    flat = np.abs(d) < 1e-12
    eye_inside = (eye_a >= 0.0) & (eye_a <= GRID)
    lo = np.where(flat, -np.inf, lo)
    hi = np.where(flat, np.inf, hi)
    t_enter = np.maximum(lo.max(axis=1), 0.0)
    t_exit = hi.min(axis=1)
    valid = (t_enter < t_exit) & ~np.any(flat & ~eye_inside, axis=1)

    # Entry point, nudged slightly inside so floor() lands in a valid voxel.
    p = eye_a + d * (t_enter + 1e-6)[:, None]
    out = dda_from_points(p, d)
    out["valid"] = valid
    return out


def dda_from_points(p, d):
    """DDA registers for rays already AT point p (N,3) heading in direction d
    (N,3, need not be unit). This is the half of the setup that does not care
    how the ray got there, so primary rays (which enter through the grid
    boundary) and shadow rays (which start on a surface inside the grid) share
    it."""
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    ivox = np.clip(np.floor(p), 0, GRID - 1).astype(np.int64)

    alive = np.abs(d) >= MIN_AXIS_DIR
    positive = d > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary = np.where(positive, ivox + 1 - p, ivox - p) / d
        inc = np.round(FP_ONE / np.abs(d))
        nxt = np.round(boundary * FP_ONE)
    # Dead axes never step: timer pinned at max, increment zero.
    inc = np.where(alive, np.clip(inc, 0, TIMER_MAX), 0)
    nxt = np.where(alive, np.clip(nxt, 0, TIMER_MAX), TIMER_MAX)

    return {
        "voxel": ivox,
        "signs": np.where(alive, positive, True),
        "next": nxt.astype(np.uint32),
        "inc": inc.astype(np.uint32),
    }


def dda_row(batch, i):
    """One ray out of dda_init_batch(), in the scalar dda_init() layout."""
    return {
        "voxel": [int(v) for v in batch["voxel"][i]],
        "signs": [int(s) for s in batch["signs"][i]],
        "next": [int(v) for v in batch["next"][i]],
        "inc": [int(v) for v in batch["inc"][i]],
    }


def camera_ray(px, py, width, height):
    """Perspective camera at CAMERA_EYE, aimed at CAMERA_TARGET (see
    _camera_basis) -- currently the top-right-front corner of the now fully
    enclosed box, looking down at the sphere cluster. Positioned relative to
    GRID so framing is identical at any grid size."""
    eye = CAMERA_EYE
    fov_y = math.radians(CAMERA_FOV_Y_DEG)
    aspect = width / height
    half_h = math.tan(fov_y / 2)
    ndc_x = (px + 0.5) / width * 2.0 - 1.0
    ndc_y = 1.0 - (py + 0.5) / height * 2.0   # image row 0 = +y (top)
    cx, cy, cz = ndc_x * half_h * aspect, ndc_y * half_h, 1.0
    direction = tuple(
        cx * CAMERA_RIGHT[j] + cy * CAMERA_UP[j] + cz * CAMERA_FORWARD[j]
        for j in range(3))
    return eye, direction


# Outward surface normal for each hardware face_id: the face the ray entered
# the hit voxel through (see step_control_multi.sv). face_id == 6 is the
# sentinel for "ray hit its own starting voxel" -- no entry face/normal is
# known in that case.
FACE_NORMAL = {
    0: (-1.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (0.0, -1.0, 0.0),
    3: (0.0, 1.0, 0.0),
    4: (0.0, 0.0, -1.0),
    5: (0.0, 0.0, 1.0),
}


def raw_illumination(result):
    """Un-normalized RGB light arriving at a HIT surface: the sum over all
    light samples of (light colour * N.L), averaged over the sample count.

    Illumination is a COLOUR, not a scalar, so a tinted light tints every
    surface it reaches -- that is the whole mechanism behind the blue lamp.
    White lights leave the three channels equal, reducing exactly to the
    previous monochrome behaviour.

    face_id == 6 (ray hit its own starting voxel) has no known normal, so it
    is treated as fully unlit -- normalize_image() below then naturally pushes
    it toward the darkest end of the render."""
    normal = FACE_NORMAL.get(result["face"])
    if normal is None:
        return (0.0, 0.0, 0.0)

    hx = result["x"] + 0.5
    hy = result["y"] + 0.5
    hz = result["z"] + 0.5
    nx, ny, nz = normal
    tr = tg = tb = 0.0
    for lx, ly, lz, cr, cg, cb in LIGHT_POINTS:
        dx, dy, dz = lx - hx, ly - hy, lz - hz
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-9:
            continue
        cos_theta = (nx * dx + ny * dy + nz * dz) / length
        if cos_theta > 0.0:
            tr += cr * cos_theta
            tg += cg * cos_theta
            tb += cb * cos_theta
    # Sample colours already carry their panel's 1/count weight, so this is a
    # plain sum: each light contributes independently of its sample density.
    return (tr * LIGHT_INTENSITY, tg * LIGHT_INTENSITY, tb * LIGHT_INTENSITY)


FACE_NORMAL_ARRAY = np.array(
    [FACE_NORMAL.get(f, (0.0, 0.0, 0.0)) for f in range(8)], dtype=np.float64)


def raw_illumination_batch(face, hx, hy, hz):
    """Vectorised raw_illumination() over arrays of hits.

    Returns (N, 3) RGB illumination. Faces with no known normal (the
    face_id == 6 sentinel) get a zero normal from FACE_NORMAL_ARRAY, so they
    fall out as unlit without needing a separate branch."""
    normal = FACE_NORMAL_ARRAY[face]
    nx, ny, nz = normal[:, 0].copy(), normal[:, 1].copy(), normal[:, 2].copy()
    px = hx.astype(np.float64) + 0.5      # voxel centres
    py = hy.astype(np.float64) + 0.5
    pz = hz.astype(np.float64) + 0.5

    n = px.size
    tr = np.zeros(n)
    tg = np.zeros(n)
    tb = np.zeros(n)
    # Scratch buffers reused across lights: with ~1M hits, allocating fresh
    # (N,3) temporaries per light dominates the runtime, so everything below
    # works component-wise and writes through `out=`.
    dx = np.empty(n)
    dy = np.empty(n)
    dz = np.empty(n)
    acc = np.empty(n)
    cos_theta = np.empty(n)

    for lx, ly, lz, cr, cg, cb in LIGHT_POINTS:
        np.subtract(lx, px, out=dx)
        np.subtract(ly, py, out=dy)
        np.subtract(lz, pz, out=dz)

        # |d|^2, then the dot product, then cos = dot / |d|
        np.multiply(dx, dx, out=acc)
        np.multiply(dy, dy, out=cos_theta)
        np.add(acc, cos_theta, out=acc)
        np.multiply(dz, dz, out=cos_theta)
        np.add(acc, cos_theta, out=acc)
        np.sqrt(acc, out=acc)
        np.maximum(acc, 1e-9, out=acc)

        np.multiply(nx, dx, out=cos_theta)
        np.multiply(ny, dy, out=dy)
        np.add(cos_theta, dy, out=cos_theta)
        np.multiply(nz, dz, out=dz)
        np.add(cos_theta, dz, out=cos_theta)
        np.divide(cos_theta, acc, out=cos_theta)
        np.maximum(cos_theta, 0.0, out=cos_theta)

        tr += cr * cos_theta
        tg += cg * cos_theta
        tb += cb * cos_theta

    out = np.stack((tr, tg, tb), axis=1)
    out *= LIGHT_INTENSITY
    return out


def ray_hit_points(origins, dirs, hit_vox, face):
    """The exact sub-voxel point where a ray crossed into its hit voxel,
    recovered analytically from the ray and the reported face.

    The tracer only reports the hit VOXEL. Firing shadow rays from voxel
    centres makes every pixel covering that voxel share one origin, so their
    shadow results are identical up to jitter -- blocky error rather than
    noise. Recovering the true crossing gives each pixel its own origin, which
    is what makes the error fine-grained and therefore filterable.

    `origins` may be a single point (all rays from the eye) or one per ray,
    which is what reflected rays need."""
    o = np.asarray(origins, dtype=np.float64)
    if o.ndim == 1:
        o = np.broadcast_to(o, dirs.shape)
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    rows = np.arange(len(face))
    axis = np.clip(face >> 1, 0, 2).astype(np.int64)
    # face 0/2/4 stepped positively, so the ray entered through the LOW side.
    low = (face & 1) == 0
    plane = hit_vox[rows, axis] + np.where(low, 0.0, 1.0)
    da = d[rows, axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane - o[rows, axis]) / da
    t = np.where(np.abs(da) > 1e-12, t, 0.0)
    point = o + d * t[:, None]
    # face 6/7 carry no normal; those never reach a shadow ray (their N.L is
    # zero so they are pruned), but fall back to the voxel centre regardless.
    return np.where((face < 6)[:, None], point, hit_vox + 0.5)


def reflect(dirs, face):
    """Mirror a direction about a surface. With axis-aligned voxel normals
    R = D - 2(D.N)N reduces to negating D's component along the face axis."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    n = FACE_NORMAL_ARRAY[face]
    return d - 2.0 * (d * n).sum(axis=1)[:, None] * n


def shadow_origins(hit_pt, face):
    """Shadow-ray start: the surface point nudged along its normal so floor()
    lands in the neighbouring, empty voxel. The tracer tests its own starting
    voxel, so a ray starting ON the hit surface self-shadows every pixel."""
    normal = FACE_NORMAL_ARRAY[face]
    p = hit_pt + normal * SHADOW_ORIGIN_BIAS
    return np.clip(p, 1e-3, GRID - 1e-3)


def shadow_jobs(origin_pt, light_xyz):
    """Build shadow rays from surface points toward one light sample.

    Returns (dda, max_steps). max_steps is the exact number of voxel
    boundaries between the start and the light, which turns "ran out of steps"
    into "travelled the whole way without hitting anything" -- the only way to
    detect an INVISIBLE light, since it has no geometry to land on. Lights
    that do have geometry are detected by landing on their emissive material.
    """
    p = np.asarray(origin_pt, dtype=np.float64)
    dda = dda_from_points(p, np.asarray(light_xyz, dtype=np.float64) - p)
    target = np.clip(np.floor(light_xyz), 0, GRID - 1).astype(np.int64)
    steps = np.abs(target - dda["voxel"]).sum(axis=1)
    return dda, np.clip(steps, 1, 1023)


def _shift2d(a, dy, dx, fill):
    """a shifted so out[y, x] == a[y + dy, x + dx], padded with `fill`."""
    out = np.full_like(a, fill)
    h, w = a.shape[0], a.shape[1]
    yd0, yd1 = max(0, -dy), min(h, h - dy)
    xd0, xd1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    if yd0 < yd1 and xd0 < xd1:
        out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def _filter_1d(frac, valid, face, posf, normal, radius, plane_tol, vertical):
    """One separable pass of the plane-aware visibility filter."""
    acc = np.zeros_like(frac)
    wsum = np.zeros(frac.shape[:2], dtype=np.float64)
    for k in range(-radius, radius + 1):
        dy, dx = (k, 0) if vertical else (0, k)
        s_valid = _shift2d(valid, dy, dx, False)
        s_face = _shift2d(face, dy, dx, 255)
        s_pos = _shift2d(posf, dy, dx, 1e9)
        # Distance from the neighbour's hit point to THIS pixel's surface
        # plane. Zero for coplanar voxels however far apart they lie; one full
        # voxel for a step onto the next terrace.
        along = np.abs(((s_pos - posf) * normal).sum(axis=2))
        ok = valid & s_valid & (s_face == face) & (along <= plane_tol)
        acc += np.where(ok[:, :, None], _shift2d(frac, dy, dx, 0.0), 0.0)
        wsum += ok
    return np.divide(acc, wsum[:, :, None],
                     out=frac.copy(), where=wsum[:, :, None] > 0)


def filter_visibility(frac, valid, face, pos, radius, plane_tol):
    """Smooth the per-pixel visibility fraction across coplanar surfaces.

    frac is (H, W, 3), valid/face are (H, W), pos is (H, W, 3). Two pixels are
    averaged only if both are valid, share an entry face, and their hit points
    lie within `plane_tol` voxels of each other ALONG that face's normal.

    Testing coplanarity rather than proximity is what lets one filter serve
    both cases: a flat wall's voxels are coplanar no matter how far apart, so
    the penumbra smooths freely, while a sphere's terraces step along the
    normal and stay independent, preserving per-voxel detail.

    Filtering VISIBILITY rather than the final illumination is the other half:
    the noise lives entirely in the visibility term, while the sharp detail
    (N.L, albedo, geometry) lives in the factor it multiplies. Blurring the
    product would soften the shading terminator and bleed colour across
    material edges; blurring the ratio does neither.

    Separable: two 1-D passes, so cost is 2*(2R+1) taps instead of (2R+1)^2.
    """
    if radius <= 0:
        return frac
    posf = pos.astype(np.float64)
    normal = FACE_NORMAL_ARRAY[face]
    out = _filter_1d(frac, valid, face, posf, normal, radius, plane_tol, False)
    return _filter_1d(out, valid, face, posf, normal, radius, plane_tol, True)


def shadow_visible(hit, mat_at_hit, emissive_lut):
    """Was the light reached? Either the ray hit the lamp's own emissive
    geometry, or it never hit anything within its step budget."""
    return (~hit) | emissive_lut[mat_at_hit]


def hit_material(material, result):
    """Material index of the voxel a ray hit."""
    return material[result["z"]][result["y"]][result["x"]]


def is_emissive(material, result):
    """True if the hit surface emits light and must not be shaded."""
    return hit_material(material, result) in EMISSIVE


def emissive_color(material, result):
    """Full-brightness colour for an emissive hit -- no lighting applied."""
    return to_rgb8(PALETTE[hit_material(material, result)])


def hit_albedo(material, result):
    """Linear-RGB albedo of the voxel a ray hit. This is the whole of
    "option A2": the hardware hands back the hit voxel's coordinates, so the
    host resolves colour from its own material grid -- no FPGA involvement."""
    return PALETTE[hit_material(material, result)]


def background_color(result):
    """Linear-RGB colour for a ray that never hit anything."""
    return COLOR_TIMEOUT if result["timeout"] else COLOR_BACKGROUND


def to_rgb8(color):
    """Linear 0..1 RGB triple -> a triple of 0..255 ints."""
    return tuple(min(255, max(0, int(round(c * 255)))) for c in color)


def normalize_image(illum, albedo, hit_mask, image):
    """Turn per-pixel RGB illumination plus material albedo into final pixels.

    Each light's colour is split into a BRIGHTNESS and a TINT:

      brightness = max channel of the illumination. Stretched across the full
        available range over all lit pixels, then CONTRAST, then lifted into
        [SHADE_FLOOR, 1.0]. Only this scalar is normalised, so the stretch
        cannot distort colour -- normalising each channel independently would
        push a scene with little blue to full blue.
      tint = illumination / brightness, i.e. its chromaticity. Multiplied into
        the albedo, this is what paints the blue lamp's hue onto whatever it
        lights. Dividing by the MAX channel (rather than luminance) keeps
        every component <= 1, so a saturated light tints without clipping.

    A white light gives tint (1,1,1) and reduces exactly to plain albedo
    shading. Fully unlit pixels have no chromaticity, so they fall back to a
    neutral tint and simply sit at SHADE_FLOOR.

    The floor is applied AFTER contrast so CONTRAST still operates on a full
    0..1 range and the floor is a pure final lift, not something the contrast
    curve stretches back toward black.

    Operates on flat per-pixel arrays: illum and albedo are (N, 3) float,
    hit_mask is (N,) bool, and image is an (N, 3) uint8 array modified in
    place. Background pixels (hit_mask False) are left untouched."""
    illum = np.asarray(illum, dtype=np.float64)      # (N, 3)
    albedo = np.asarray(albedo, dtype=np.float64)    # (N, 3)
    hit_mask = np.asarray(hit_mask, dtype=bool)      # (N,)
    if not hit_mask.any():
        return

    mag = illum.max(axis=1)
    lit = mag[hit_mask]
    lo, hi = lit.min(), lit.max()
    span = hi - lo

    norm = (mag - lo) / span if span > 1e-9 else np.ones_like(mag)
    norm = 0.5 + (norm - 0.5) * CONTRAST       # spread away from mid-gray
    np.clip(norm, 0.0, 1.0, out=norm)
    norm = SHADE_FLOOR + (1.0 - SHADE_FLOOR) * norm    # lift off black

    # Unlit pixels have no chromaticity, so fall back to a neutral tint.
    safe = np.maximum(mag, 1e-9)[:, None]
    ok = mag[:, None] > 1e-9
    tint = np.where(ok, illum / safe, 1.0)

    rgb = albedo * tint * norm[:, None]
    out = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    image[hit_mask] = out[hit_mask]


# =============================================================================
# PNG output (grayscale, stdlib only)
# =============================================================================
def write_png(path, image, width, height):
    """Write a truecolour PNG from an (N, 3) or (H, W, 3) uint8 array."""
    def chunk(kind, data):
        body = kind + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    pixels = np.asarray(image, dtype=np.uint8).reshape(height, width, 3)
    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = np.concatenate(
        (np.zeros((height, 1), dtype=np.uint8), pixels.reshape(height, -1)),
        axis=1).tobytes()
    payload = b"".join([
        b"\x89PNG\r\n\x1a\n",
        # bit depth 8, colour type 2 = truecolour RGB (was 0 = greyscale)
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw, 6)),
        chunk(b"IEND", b""),
    ])
    with open(path, "wb") as handle:
        handle.write(payload)


def trace_all(dev, words, dda=None, max_steps=None, label=None, every=512):
    """Trace an (N, JOB_WORDS) job array in batches, returning decoded arrays
    indexed 0..N-1: hit, timeout, face and pos (N,3).

    Jobs must carry pixel_id = their own row index (job_words_batch is called
    that way below). Results come back out of order within a batch, so each
    one is matched to its row through a scratch lookup -- a batch spans far
    fewer rows than the pixel_id wrap period, so the mapping is unambiguous.
    """
    n = len(words)
    r_hit = np.zeros(n, dtype=bool)
    r_timeout = np.zeros(n, dtype=bool)
    r_face = np.full(n, 7, dtype=np.uint8)      # 7 = no known normal
    r_pos = np.zeros((n, 3), dtype=np.int32)
    lookup = np.full(1 << PIXEL_ID_BITS, -1, dtype=np.int64)
    t0 = time.monotonic()

    for start in range(0, n, BATCH_RAYS):
        stop = min(start + BATCH_RAYS, n)
        ids = np.arange(start, stop, dtype=np.int64)
        lookup[ids & PIXEL_ID_MASK] = ids

        if USE_BULK_PIPES:
            recs = dev.trace_batch(words[start:stop])
            info, xy, zs = recs[:, 0], recs[:, 1], recs[:, 2]
            rows = lookup[info & PIXEL_ID_MASK]
            if np.any(rows < 0):
                raise RuntimeError("result carried an id not in this batch")
            r_hit[rows] = (info & (1 << 14)) != 0
            r_timeout[rows] = (info & (1 << 15)) != 0
            r_face[rows] = ((info >> 16) & 0x7).astype(np.uint8)
            r_pos[rows, 0] = (xy & 0xFFFF).astype(np.int32)
            r_pos[rows, 1] = ((xy >> 16) & 0xFFFF).astype(np.int32)
            r_pos[rows, 2] = (zs & 0xFFFF).astype(np.int32)
        else:
            for k in range(start, stop):
                steps = int(max_steps[k]) if np.ndim(max_steps) else max_steps
                res = dev.trace_ray(dda_row(dda, k), k, steps)
                r_hit[k] = res["hit"]
                r_timeout[k] = res["timeout"]
                r_face[k] = res["face"]
                r_pos[k] = (res["x"], res["y"], res["z"])

        lookup[ids & PIXEL_ID_MASK] = -1
        if label and ((start // BATCH_RAYS) % every == 0 or stop == n):
            el = time.monotonic() - t0
            eta = el / stop * (n - stop) if stop else 0.0
            print(f"  {label}: {stop}/{n} rays  {el:.1f}s elapsed, "
                  f"~{eta:.0f}s left")
    return r_hit, r_timeout, r_face, r_pos


# =============================================================================
# Main
# =============================================================================
def main():
    if not 1 <= MAX_STEPS <= 1023:
        sys.exit("MAX_STEPS must be in 1..1023")
    if USE_BULK_PIPES:
        jobs_per_block = PIPE_BLOCK_SIZE // (JOB_WORDS * 4)
        if BATCH_RAYS % jobs_per_block:
            sys.exit(f"BATCH_RAYS must be a multiple of {jobs_per_block} so "
                     f"job bytes align to the {PIPE_BLOCK_SIZE}-byte pipe block")
        if BATCH_RAYS > MAX_BATCH_RAYS:
            sys.exit(f"BATCH_RAYS must be <= {MAX_BATCH_RAYS} so a whole batch "
                     f"fits the RTL job ({JOB_FIFO_DEPTH}) and result "
                     f"({RES_FIFO_DEPTH}) FIFOs -- exceeding it deadlocks")
    if WIDTH * HEIGHT > (1 << PIXEL_ID_BITS):
        print(f"Note: {WIDTH}x{HEIGHT} = {WIDTH * HEIGHT} pixels exceeds the "
              f"{1 << PIXEL_ID_BITS}-entry pixel_id space, so ids wrap and act "
              f"as a mod-{1 << PIXEL_ID_BITS} consistency check (see trace_ray).")

    dev = RaytracerDevice(BITFILE)
    dev.reset()

    print("Packing and loading scene...")
    material = make_scene()
    dev.load_scene(pack_scene(material))

    total = WIDTH * HEIGHT
    t0 = time.monotonic()

    # ---- ray setup: every pixel at once -------------------------------------
    eye, dirs = camera_rays(WIDTH, HEIGHT)
    rays = dda_init_batch(eye, dirs)
    live = np.nonzero(rays["valid"])[0]       # pixels whose ray enters the grid
    grid_misses = total - live.size
    sub = {k: rays[k][live] for k in ("voxel", "signs", "next", "inc")}
    words = job_words_batch(sub, np.arange(live.size, dtype=np.uint32),
                            MAX_STEPS)
    print(f"Ray setup: {live.size} rays in {time.monotonic() - t0:.1f}s "
          f"({grid_misses} never enter the grid)")

    dev.set_result_pipe(USE_BULK_PIPES)
    print(f"Rendering via {'bulk pipes' if USE_BULK_PIPES else 'wire path'}...")

    # ---- primary pass -------------------------------------------------------
    t_trace = time.monotonic()
    p_hit, p_timeout, p_face, p_pos = trace_all(
        dev, words, sub, MAX_STEPS, label="primary")
    print(f"Primary trace: {time.monotonic() - t_trace:.1f}s")

    # Scatter back to per-pixel arrays (row i of the job list is pixel live[i]).
    r_hit = np.zeros(total, dtype=bool)
    r_timeout = np.zeros(total, dtype=bool)
    r_face = np.full(total, 7, dtype=np.uint8)
    r_pos = np.zeros((total, 3), dtype=np.int32)
    r_hit[live] = p_hit
    r_timeout[live] = p_timeout
    r_face[live] = p_face
    r_pos[live] = p_pos

    if getattr(dev, "lost_results", 0):
        print(f"WARNING: {dev.lost_results} results never arrived "
              f"({dev.lost_results / total * 100:.2f}% of pixels left "
              f"unshaded) -- the result pipe is dropping data.")

    palette = np.array(PALETTE, dtype=np.float64)
    emissive = np.zeros(len(PALETTE), dtype=bool)
    emissive[list(EMISSIVE)] = True

    # ---- mirror bounces -----------------------------------------------------
    # A ray landing on a mirror is not shaded: it is reflected and followed.
    # Each pixel carries the ray that produced its CURRENT hit (needed both to
    # reflect and, later, as the view vector for specular) plus a throughput
    # multiplier accumulating what every mirror so far let through.
    cur_org = np.broadcast_to(np.asarray(eye, dtype=np.float64),
                              (total, 3)).copy()
    cur_dir = dirs.copy()
    throughput = np.ones((total, 3), dtype=np.float64)
    t_mirror = time.monotonic()
    bounce_rays = 0

    for bounce in range(MIRROR_BOUNCES if REFLECTIVITY.any() else 0):
        mats_now = np.zeros(total, dtype=np.uint8)
        mats_now[r_hit] = material[r_pos[r_hit, 2], r_pos[r_hit, 1],
                                   r_pos[r_hit, 0]]
        # Stop following a ray once its contribution can no longer move an
        # 8-bit pixel -- that, not the bounce cap, is what usually ends it.
        alive = (r_hit & (REFLECTIVITY[mats_now] > 0.0)
                 & (throughput.max(axis=1) > MIRROR_MIN_THROUGHPUT))
        idx = np.nonzero(alive)[0]
        if idx.size == 0:
            break

        face_i = r_face[idx]
        hp = ray_hit_points(cur_org[idx], cur_dir[idx], r_pos[idx], face_i)
        refl = reflect(cur_dir[idx], face_i)
        org = shadow_origins(hp, face_i)   # same one-voxel-off trick

        bdda = dda_from_points(org, refl)
        bwords = job_words_batch(bdda, np.arange(idx.size, dtype=np.uint32),
                                 MAX_STEPS)
        bounce_rays += idx.size
        b_hit, b_to, b_face, b_pos = trace_all(
            dev, bwords, bdda, MAX_STEPS,
            label=f"mirror bounce {bounce + 1}", every=4096)

        throughput[idx] *= (REFLECTIVITY[mats_now[idx]][:, None]
                            * palette[mats_now[idx]])
        cur_org[idx] = org
        cur_dir[idx] = refl
        r_hit[idx] = b_hit
        r_timeout[idx] = b_to
        r_face[idx] = b_face
        r_pos[idx] = b_pos
        print(f"  bounce {bounce + 1}: {idx.size} rays reflected")

    if bounce_rays:
        print(f"Mirror bounces: {time.monotonic() - t_mirror:.1f}s "
              f"({bounce_rays} reflection rays)")

    # ---- classify hits ------------------------------------------------------

    image = np.zeros((total, 3), dtype=np.uint8)
    # A ray that bounced off at least one mirror but never resolved onto a
    # surface -- whether it ran out of DDA steps mid-chain, or (the common
    # case here) it got redirected back out through the open camera-facing
    # face, which reports neither hit nor timeout -- still carries whatever
    # throughput it accumulated from the mirrors it already left. Show that
    # residual tint instead of flat black/grey, so a mirror corridor dims
    # toward a silvery vanishing point instead of a hard void. A ray that
    # never bounced at all (throughput still exactly 1) is a genuine
    # non-mirror-related timeout and keeps the flat debug grey.
    bounced = throughput.min(axis=1) < 1.0
    ghost = ~r_hit & bounced
    image[r_timeout & ~r_hit & ~bounced] = np.rint(
        np.array(COLOR_TIMEOUT) * 255)
    image[ghost] = np.clip(
        np.rint(throughput[ghost] * MIRROR_VOID_DARKEN * 255), 0, 255
    ).astype(np.uint8)

    mats = np.zeros(total, dtype=np.uint8)
    mats[r_hit] = material[r_pos[r_hit, 2], r_pos[r_hit, 1], r_pos[r_hit, 0]]
    lamp = r_hit & emissive[mats]
    # Emissive hits go straight in at full brightness and are excluded from
    # the shading mask, so normalize_image never stretches against them.
    # A lamp seen in a mirror is dimmed by every mirror on the way.
    image[lamp] = np.clip(
        np.rint(palette[mats[lamp]] * throughput[lamp] * 255), 0, 255
    ).astype(np.uint8)
    # A ray can still be sitting on a mirror when the bounce loop ends (cap
    # reached, or throughput fell below the floor) -- e.g. trapped in a
    # corner formed by several mirrors. That is an exhausted reflection
    # chain, not a surface to diffusely shade -- shading it with full
    # lighting would paint the mirror's own near-white albedo as if it were
    # directly lit, which can outshine every real light in the scene (see
    # `trapped` history). Instead show the accumulated throughput tint, same
    # as a bounce-leg timeout: a real mirror corridor dims toward a silvery
    # vanishing point rather than cutting to a hard black void.
    trapped = r_hit & ~lamp & (REFLECTIVITY[mats] > 0.0)
    image[trapped] = np.clip(
        np.rint(throughput[trapped] * MIRROR_VOID_DARKEN * 255), 0, 255
    ).astype(np.uint8)
    shaded = r_hit & ~lamp & ~trapped

    # ---- lighting, one light sample at a time -------------------------------
    t_shade = time.monotonic()
    illum = np.zeros((total, 3), dtype=np.float64)
    # Same sum without the visibility factor. Costs no rays, and dividing the
    # two isolates the (noisy) visibility from the (sharp) geometric term so
    # only the former gets filtered.
    illum_open = np.zeros((total, 3), dtype=np.float64)
    sh_idx = np.nonzero(shaded)[0]
    shadow_rays = 0

    if sh_idx.size:
        rng = np.random.default_rng(JITTER_SEED)
        normals = FACE_NORMAL_ARRAY[r_face[sh_idx]]
        # Exact surface points, so each pixel gets its own shadow-ray origin
        # rather than sharing its voxel's centre with ~64 neighbours.
        # cur_org/cur_dir are the ray that produced the FINAL hit, which after
        # a mirror bounce is not the camera ray.
        hit_pt = ray_hit_points(cur_org[sh_idx], cur_dir[sh_idx],
                                r_pos[sh_idx], r_face[sh_idx])
        origins = shadow_origins(hit_pt, r_face[sh_idx])
        pts = hit_pt
        for si, stratum in enumerate(LIGHT_STRATA):
            # Each surface point samples its own random spot in this stratum.
            lpos = jitter_strata(stratum, sh_idx.size, rng)
            cr, cg, cb = stratum["colour"]
            dx = lpos[:, 0] - pts[:, 0]
            dy = lpos[:, 1] - pts[:, 1]
            dz = lpos[:, 2] - pts[:, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            cos_theta = ((normals[:, 0] * dx + normals[:, 1] * dy
                          + normals[:, 2] * dz) / np.maximum(dist, 1e-9))
            # PRUNE: a sample behind the surface contributes nothing whether or
            # not it is occluded, so it never needs a shadow ray.
            face_on = cos_theta > 1e-9
            if not face_on.any():
                continue
            rows = sh_idx[face_on]
            weight = cos_theta[face_on]

            if ENABLE_SHADOWS:
                sdda, ssteps = shadow_jobs(origins[face_on], lpos[face_on])
                swords = job_words_batch(
                    sdda, np.arange(rows.size, dtype=np.uint32), ssteps)
                shadow_rays += rows.size
                s_hit, _, _, s_pos = trace_all(
                    dev, swords, sdda, ssteps,
                    label=f"shadow {si + 1}/{len(LIGHT_POINTS)}", every=4096)
                s_mat = np.zeros(rows.size, dtype=np.uint8)
                s_mat[s_hit] = material[s_pos[s_hit, 2], s_pos[s_hit, 1],
                                        s_pos[s_hit, 0]]
                illum_open[rows, 0] += cr * weight
                illum_open[rows, 1] += cg * weight
                illum_open[rows, 2] += cb * weight
                weight = weight * shadow_visible(s_hit, s_mat, emissive)

            illum[rows, 0] += cr * weight
            illum[rows, 1] += cg * weight
            illum[rows, 2] += cb * weight

        # Isolate visibility, smooth it, then re-apply it to the unfiltered
        # geometric term.
        if ENABLE_SHADOWS and SHADOW_FILTER_RADIUS > 0:
            t_filt = time.monotonic()
            lit = illum_open.max(axis=1) > 1e-9
            frac = np.divide(illum, illum_open,
                             out=np.ones_like(illum), where=illum_open > 1e-9)
            frac = filter_visibility(
                frac.reshape(HEIGHT, WIDTH, 3),
                (shaded & lit).reshape(HEIGHT, WIDTH),
                r_face.reshape(HEIGHT, WIDTH),
                r_pos.reshape(HEIGHT, WIDTH, 3),
                SHADOW_FILTER_RADIUS,
                SHADOW_FILTER_PLANE_TOL).reshape(total, 3)
            illum = illum_open * frac
            print(f"  visibility filter (r={SHADOW_FILTER_RADIUS}): "
                  f"{time.monotonic() - t_filt:.1f}s")

        illum *= LIGHT_INTENSITY

    # Everything reaching the camera passed through every mirror on the way,
    # so scale by throughput BEFORE normalising -- that way the brightness
    # stretch sees the dimming and deep reflections stay correctly darker.
    illum *= throughput

    albedo = np.zeros((total, 3), dtype=np.float64)
    albedo[shaded] = palette[mats[shaded]]

    normalize_image(illum, albedo, shaded, image)
    print(f"Shade: {time.monotonic() - t_shade:.1f}s  "
          f"({int(r_hit.sum())} hits, {int((~r_hit).sum())} misses, "
          f"{grid_misses} off-grid, {shadow_rays} shadow rays)")

    write_png(OUT_PNG, image, WIDTH, HEIGHT)
    print(f"Wrote {OUT_PNG}  (total {time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
