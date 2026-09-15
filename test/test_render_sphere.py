import os
import struct
import zlib

import cocotb
from cocotb.triggers import ClockCycles


IMG_W = 32
IMG_H = 32
SCENE = 32

CMD_WRITE_CONTEXT = 0x10
CMD_FILL_CACHE = 0x20
CMD_RUN_N = 0x31
CMD_READ_STATUS = 0x40
CMD_READ_REQUEST = 0x41
CMD_READ_RESULT = 0x42
CMD_POP_RESULT = 0x43

STATUS_ACTIVE = 0x01
STATUS_FREE = 0x02
STATUS_REQUEST = 0x04
STATUS_RESULT = 0x08
STATUS_READY = 0x20


async def qspi_set(dut, cs_n, sck, dq):
    dut.ui_in.value = (cs_n << 1) | sck
    dut.uio_in.value = dq & 0x0F
    await ClockCycles(dut.clk, 3)


async def qspi_nibble(dut, value):
    await qspi_set(dut, 0, 0, value)
    await qspi_set(dut, 0, 1, value)
    await qspi_set(dut, 0, 0, value)


async def qspi_byte(dut, value):
    await qspi_nibble(dut, (value >> 4) & 0x0F)
    await qspi_nibble(dut, value & 0x0F)


async def qspi_write(dut, values):
    await qspi_set(dut, 1, 0, 0)
    await qspi_set(dut, 0, 0, 0)
    for value in values:
        await qspi_byte(dut, value)
    await qspi_set(dut, 1, 0, 0)
    await ClockCycles(dut.clk, 4)


def sample_qspi_nibble(dut):
    oe = int(dut.uio_oe.value) & 0x0F
    assert oe == 0x0F, f"ASIC did not drive QSPI read nibble, uio_oe={oe:#x}"
    return int(dut.uio_out.value) & 0x0F


async def wait_for_qspi_drive(dut, max_cycles=20):
    for _ in range(max_cycles):
        if (int(dut.uio_oe.value) & 0x0F) == 0x0F:
            return
        await ClockCycles(dut.clk, 1)
    sample_qspi_nibble(dut)


async def qspi_read(dut, command, num_bytes):
    await qspi_set(dut, 1, 0, 0)
    await qspi_set(dut, 0, 0, 0)

    await qspi_nibble(dut, (command >> 4) & 0x0F)

    low = command & 0x0F
    await qspi_set(dut, 0, 0, low)
    await qspi_set(dut, 0, 1, low)

    await wait_for_qspi_drive(dut)
    await ClockCycles(dut.clk, 2)
    nibbles = [sample_qspi_nibble(dut)]
    for _ in range(num_bytes * 2 - 1):
        await qspi_set(dut, 0, 0, 0)
        await wait_for_qspi_drive(dut)
        await ClockCycles(dut.clk, 2)
        nibbles.append(sample_qspi_nibble(dut))
        await qspi_set(dut, 0, 1, 0)

    await qspi_set(dut, 1, 0, 0)
    await ClockCycles(dut.clk, 4)

    return [(nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles), 2)]


async def read_status(dut):
    data = await qspi_read(dut, CMD_READ_STATUS, 5)
    return {
        "raw": data[0],
        "free_id": data[1],
        "request_id": data[2],
        "result_count": data[3],
        "run_state": data[4],
    }


async def read_request(dut):
    data = await qspi_read(dut, CMD_READ_REQUEST, 4)
    return {
        "ctx": data[0],
        "tile_x": data[1],
        "tile_y": data[2],
        "tile_z": data[3],
    }


async def read_result(dut):
    data = await qspi_read(dut, CMD_READ_RESULT, 8)
    return {
        "ctx": data[0],
        "hit": bool(data[1] & 0x01),
        "timeout": bool(data[1] & 0x02),
        "pixel_id": data[2],
        "x": data[3],
        "y": data[4],
        "z": data[5],
        "steps": data[6],
        "face": data[7],
    }


def make_sphere_scene():
    scene = [[[False for _ in range(SCENE)] for _ in range(SCENE)] for _ in range(SCENE)]
    cx, cy, cz = 16, 16, 18
    r2 = 7 * 7
    for z in range(SCENE):
        for y in range(SCENE):
            for x in range(SCENE):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2:
                    scene[z][y][x] = True
    return scene


def tile_bytes(scene, tile_x, tile_y, tile_z):
    bits = 0
    x0 = tile_x * 4
    y0 = tile_y * 4
    z0 = tile_z * 4
    for z in range(4):
        for y in range(4):
            for x in range(4):
                sx = x0 + x
                sy = y0 + y
                sz = z0 + z
                occupied = (
                    0 <= sx < SCENE and
                    0 <= sy < SCENE and
                    0 <= sz < SCENE and
                    scene[sz][sy][sx]
                )
                if occupied:
                    bits |= 1 << ((z << 4) | (y << 2) | x)
    return [(bits >> shift) & 0xFF for shift in range(56, -1, -8)]


def ray_context_payload(px, py, pixel_id):
    # Orthographic +z ray. Make z timer smallest so the DDA chooses z.
    return [
        px & 0x3F,
        py & 0x3F,
        0x00,
        0x00,
        0x00, 0xF0,
        0x00, 0xF0,
        0x00, 0x00,
        0x00, 0x01,
        0x00, 0x01,
        0x00, 0x01,
        0x20,
        pixel_id & 0xFF,
    ]


async def load_context(dut, px, py, pixel_id):
    await qspi_write(dut, [CMD_WRITE_CONTEXT] + ray_context_payload(px, py, pixel_id))


async def fill_cache(dut, req, scene):
    payload = [
        req["ctx"],
        req["tile_x"],
        req["tile_y"],
        req["tile_z"],
    ] + tile_bytes(scene, req["tile_x"], req["tile_y"], req["tile_z"])
    await qspi_write(dut, [CMD_FILL_CACHE] + payload)


async def run_n(dut, budget):
    await qspi_write(dut, [CMD_RUN_N, budget & 0xFF])
    run_cycles = 256 if budget == 0 else budget
    await ClockCycles(dut.clk, run_cycles + 20)


async def pop_result(dut):
    await qspi_write(dut, [CMD_POP_RESULT])


def shade(result):
    if not result["hit"]:
        return 0
    depth_term = max(0, 255 - result["z"] * 5)
    face_bonus = 30 if result["face"] == 6 else 0
    return min(255, max(32, depth_term + face_bonus))


def write_pgm(path, image):
    with open(path, "wb") as handle:
        handle.write(f"P5\n{IMG_W} {IMG_H}\n255\n".encode("ascii"))
        for row in image:
            handle.write(bytes(row))


def png_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(path, image):
    raw = bytearray()
    for row in image:
        raw.append(0)
        raw.extend(row)
    payload = b"".join([
        b"\x89PNG\r\n\x1a\n",
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", IMG_W, IMG_H, 8, 0, 0, 0, 0)),
        png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
        png_chunk(b"IEND", b""),
    ])
    with open(path, "wb") as handle:
        handle.write(payload)


async def reset_dut(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0x02
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 4)


@cocotb.test()
async def render_sphere_32x32(dut):
    if os.getenv("RUN_RENDER_TEST") != "1":
        dut._log.info("Skipping optional render test. Set RUN_RENDER_TEST=1 to run it.")
        return

    await reset_dut(dut)

    reset_status = await read_status(dut)
    assert reset_status["raw"] & STATUS_FREE, f"bad reset status/read alignment: {reset_status}"

    scene = make_sphere_scene()
    image = [[0 for _ in range(IMG_W)] for _ in range(IMG_H)]
    next_pixel = 0
    results = 0
    hits = 0
    misses = 0
    tile_fills = 0
    run_commands = 0
    rays_loaded = 0
    ctx_to_pixel = {}
    max_iters = 200000
    last_progress = max_iters
    last_status = None

    while results < IMG_W * IMG_H:
        status = await read_status(dut)

        while (status["raw"] & STATUS_FREE) and next_pixel < IMG_W * IMG_H:
            ctx_id = status["free_id"]
            assert ctx_id < 5, f"invalid free context in status {status}"
            px = next_pixel % IMG_W
            py = next_pixel // IMG_W
            await load_context(dut, px, py, next_pixel)
            ctx_to_pixel[ctx_id] = next_pixel
            next_pixel += 1
            rays_loaded += 1
            last_progress = max_iters
            status = await read_status(dut)

        while status["raw"] & STATUS_REQUEST:
            req = await read_request(dut)
            assert req["ctx"] < 5, f"invalid request context {req}"
            await fill_cache(dut, req, scene)
            tile_fills += 1
            last_progress = max_iters
            status = await read_status(dut)

        if status["raw"] & STATUS_READY:
            await run_n(dut, 0)
            run_commands += 1
            last_progress = max_iters
            status = await read_status(dut)

        while status["raw"] & STATUS_RESULT:
            result = await read_result(dut)
            assert result["ctx"] in ctx_to_pixel, f"result for unknown context {result}"
            full_pixel_id = ctx_to_pixel.pop(result["ctx"])
            py = full_pixel_id // IMG_W
            px = full_pixel_id % IMG_W
            image[py][px] = shade(result)
            if result["hit"]:
                hits += 1
            else:
                misses += 1
            results += 1
            await pop_result(dut)
            last_progress = max_iters
            status = await read_status(dut)

        max_iters -= 1
        if max_iters % 5000 == 0:
            dut._log.info(
                "render progress: loaded=%d results=%d hits=%d misses=%d fills=%d runs=%d status=%s active_ctx=%s",
                rays_loaded,
                results,
                hits,
                misses,
                tile_fills,
                run_commands,
                status,
                sorted(ctx_to_pixel.items()),
            )
        last_status = status
        assert max_iters > 0 and (last_progress - max_iters) < 20000, (
            "render loop stalled: "
            f"loaded={rays_loaded} results={results} hits={hits} misses={misses} "
            f"fills={tile_fills} runs={run_commands} status={last_status} "
            f"active_ctx={sorted(ctx_to_pixel.items())}"
        )

    out_dir = os.path.join(os.getcwd(), "sim_build", "render")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "render_sphere.png")
    pgm_path = os.path.join(out_dir, "render_sphere.pgm")
    write_png(png_path, image)
    write_pgm(pgm_path, image)

    center = image[IMG_H // 2][IMG_W // 2]
    corner = image[0][0]
    assert results == IMG_W * IMG_H
    assert hits > 0
    assert hits < IMG_W * IMG_H
    assert center > 0
    assert corner == 0

    dut._log.info(
        "render complete: rays=%d results=%d hits=%d misses=%d tile_fills=%d run_cmds=%d png=%s",
        rays_loaded,
        results,
        hits,
        misses,
        tile_fills,
        run_commands,
        png_path,
    )
