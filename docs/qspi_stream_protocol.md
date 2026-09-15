# QSPI Stream Protocol

This design uses `ui_in[0]` as `qspi_sck`, `ui_in[1]` as active-low
`qspi_cs_n`, and `uio[3:0]` as the 4-bit bidirectional QSPI data bus.
Nibbles are transferred most-significant nibble first.

The core stores up to five in-flight ray contexts and five 4x4x4 aligned tile
cache banks. Context `N` can only refill/write cache bank `N`, but all contexts
probe all five banks when stepping. When a context finishes, its compact result
is queued and the context slot becomes free for a new ray.

Commands are one byte:

- `0x00` reset all contexts, results, and error state
- `0x10` auto-load one ray context, followed by 18 payload bytes
- `0x20` fill one context-owned 4x4x4 tile cache bank, followed by 12 payload bytes
- `0x30` step one ready context selected by the round-robin scheduler
- `0x31` run autonomously for N cached steps, followed by 1 budget byte
- `0x40` read status, 5 bytes
- `0x41` read one tile-fill request, 4 bytes
- `0x42` read oldest result, 8 bytes
- `0x43` pop oldest result after reading it

Context payload for `0x10`:

- byte 0: `voxel_x[5:0]`
- byte 1: `voxel_y[5:0]`
- byte 2: `voxel_z[5:0]`
- byte 3: `{unused[7:3], step_z_neg, step_y_neg, step_x_neg}`
- bytes 4-5: `timer_x[15:0]`
- bytes 6-7: `timer_y[15:0]`
- bytes 8-9: `timer_z[15:0]`
- bytes 10-11: `inc_x[15:0]`
- bytes 12-13: `inc_y[15:0]`
- bytes 14-15: `inc_z[15:0]`
- byte 16: `max_steps`
- byte 17: `pixel_id`

Cache fill payload for `0x20`:

- byte 0: context/cache bank id, 0 through 4
- byte 1: aligned tile tag x, equivalent to `voxel_x[5:2]`
- byte 2: aligned tile tag y, equivalent to `voxel_y[5:2]`
- byte 3: aligned tile tag z, equivalent to `voxel_z[5:2]`
- bytes 4-11: 64 occupancy bits for the tile, most-significant byte first

Within a cached tile, the occupancy bit index is:

```text
index = {voxel_z[1:0], voxel_y[1:0], voxel_x[1:0]}
```

So index 0 is in byte 11 bit 0, and index 63 is in byte 4 bit 7.

Status packet for `0x40`:

- byte 0: status bits
- byte 1: free context id, or `0xff` if no free slot exists
- byte 2: requesting context id, or `0xff` if no tile request exists
- byte 3: result FIFO count
- byte 4: `{unused[7], stop_reason[6:4], unused[3:1], run_active[0]}`

Status byte bits:

- bit 0: at least one active context
- bit 1: at least one free context slot
- bit 2: at least one tile-fill request exists
- bit 3: result available
- bit 4: all context slots full
- bit 5: at least one context is ready to step
- bit 6: QSPI data pins are being driven by the ASIC
- bit 7: protocol/error flag

`RUN_N` budget:

- Send `0x31, N`.
- `N = 1..255` runs for up to N internally scheduled cached steps.
- `N = 0` runs for up to 256 internally scheduled cached steps.

Run stops early if the result FIFO is full, no context is ready, every active
context is blocked on a tile miss, or an error occurs.

Stop reasons in status byte 4:

- `0`: idle/no stop reason
- `1`: result FIFO full
- `2`: blocked on at least one tile-fill request
- `3`: no ready context
- `4`: run budget exhausted

Tile-fill request packet for `0x41`:

- byte 0: requesting context id, or `0xff` if no request exists
- byte 1: requested aligned tile tag x
- byte 2: requested aligned tile tag y
- byte 3: requested aligned tile tag z

Result packet for `0x42`:

- byte 0: context id, or `0xff` if no result exists
- byte 1: `{unused[7:2], timeout, hit}`
- byte 2: pixel id
- byte 3: result x
- byte 4: result y
- byte 5: result z
- byte 6: step count
- byte 7: face id

Typical host loop:

1. Send `0x10` while the status packet reports a free slot.
2. Read `0x41` when status bit 2 is high.
3. Build the requested 4x4x4 aligned tile for that context id.
4. Send `0x20` with the context id, tile tag, and 64 occupancy bits.
5. Send `0x31, N` while status bit 5 is high. The core will keep stepping from
   cached data until the budget is exhausted, a tile miss blocks progress, a
   result is produced, or the result FIFO fills. `0x30` remains available for
   single-step debugging.
6. Read `0x42` when status bit 3 is high, then send `0x43` to pop it.
7. Keep loading new contexts whenever status bit 1 reports a free slot.
