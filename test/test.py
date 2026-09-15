import cocotb
from cocotb.triggers import ClockCycles


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


async def qspi_transaction(dut, values):
    await qspi_set(dut, 1, 0, 0)
    await qspi_set(dut, 0, 0, 0)
    for value in values:
        await qspi_byte(dut, value)
    await qspi_set(dut, 1, 0, 0)
    await ClockCycles(dut.clk, 4)


def status(dut):
    return int(dut.uo_out.value)


def context_payload(pixel_id):
    return [
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x00, 0x01,
        0x00, 0x02,
        0x00, 0x01,
        0x00, 0x01,
        0x00, 0x01,
        0x0A, pixel_id,
    ]


def cache_fill_payload(ctx_id, tag_x, tag_y, tag_z, tile_bytes):
    return [ctx_id, tag_x, tag_y, tag_z] + tile_bytes


@cocotb.test()
async def test_qspi_stream_smoke(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0x02
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    assert int(dut.uio_out.value) == 0
    assert int(dut.uio_oe.value) == 0

    # Load two contexts. Auto-allocation should place them in slots 0 and 1.
    await qspi_transaction(dut, [0x10] + context_payload(0x2A))

    assert status(dut) & 0x01  # at least one active context
    assert status(dut) & 0x02  # at least one free slot remains
    assert status(dut) & 0x04  # at least one tile-fill request exists

    await qspi_transaction(dut, [0x10] + context_payload(0x33))
    assert status(dut) & 0x01
    assert status(dut) & 0x02
    assert status(dut) & 0x04

    # Fill ctx 0's owned cache bank with tile (0,0,0). Bit 0 is empty and
    # bit 1 is solid, so ctx 0 can step once and then finish without another
    # host-side cache fill.
    await qspi_transaction(dut, [0x20] + cache_fill_payload(
        0x00, 0x00, 0x00, 0x00,
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02],
    ))
    assert status(dut) & 0x20  # one context is ready to step

    await qspi_transaction(dut, [0x31, 0x02])  # run two cached steps autonomously
    await ClockCycles(dut.clk, 20)
    assert status(dut) & 0x01
    assert status(dut) & 0x02
    assert status(dut) & 0x08  # result available

    dut.ena.value = 0
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 0
