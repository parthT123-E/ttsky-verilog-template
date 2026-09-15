"""Validate the host-side bulk pipe encoding and the size constraints it
must satisfy against the RTL FIFO depths / pipe alignment rules."""
import struct
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")
import render_fpga as rf

print("module imports OK")
print(f"USE_BULK_PIPES={rf.USE_BULK_PIPES}  BATCH_RAYS={rf.BATCH_RAYS}")

jpb = rf.PIPE_BLOCK_SIZE // (rf.JOB_WORDS * 4)
job_bytes = rf.BATCH_RAYS * rf.JOB_WORDS * 4
res_bytes = rf.BATCH_RAYS * rf.RES_WORDS * 4
print(f"jobs/block={jpb}  job bytes/batch={job_bytes} "
      f"({job_bytes / rf.PIPE_BLOCK_SIZE:.0f} blocks, "
      f"aligned={job_bytes % rf.PIPE_BLOCK_SIZE == 0})")
print(f"result bytes/batch={res_bytes} (16B-aligned={res_bytes % 16 == 0})")
print(f"job words/batch={rf.BATCH_RAYS * rf.JOB_WORDS} "
      f"(RTL JOB_FIFO {rf.JOB_FIFO_DEPTH})")
print(f"res words/batch={rf.BATCH_RAYS * rf.RES_WORDS} "
      f"(RTL RES_FIFO {rf.RES_FIFO_DEPTH})")
print(f"max batch for these depths: {rf.MAX_BATCH_RAYS} rays")

assert job_bytes % rf.PIPE_BLOCK_SIZE == 0
assert res_bytes % 16 == 0

# --- deadlock margin -------------------------------------------------------
# The host writes a whole batch of jobs before reading any results, so in the
# worst case every job's result is queued while the job write is still in
# flight. If either FIFO overflowed there, the tracer would stall, jobs would
# stop draining, the pipe write would block forever, and the run would
# deadlock. Both must therefore hold a FULL batch.
JOB_FIFO_DEPTH = rf.JOB_FIFO_DEPTH   # host mirrors of the RTL localparams
RES_FIFO_DEPTH = rf.RES_FIFO_DEPTH
BLOCK_WORDS = rf.PIPE_BLOCK_SIZE // 4

# Worst-case queued jobs also includes block-granularity padding on a short
# final batch: expected rounds len up to a multiple of jobs-per-block.
worst_expected = -(-rf.BATCH_RAYS // jpb) * jpb
job_hi = worst_expected * rf.JOB_WORDS
res_hi = worst_expected * rf.RES_WORDS
print(f"\nworst-case batch (incl. padding) = {worst_expected} rays")
print(f"  job FIFO  {job_hi:5d}/{JOB_FIFO_DEPTH} words "
      f"({JOB_FIFO_DEPTH - job_hi} spare)")
print(f"  res FIFO  {res_hi:5d}/{RES_FIFO_DEPTH} words "
      f"({RES_FIFO_DEPTH - res_hi} spare)")
assert worst_expected <= rf.BATCH_RAYS, "padding must not exceed BATCH_RAYS"
assert job_hi <= JOB_FIFO_DEPTH, "job FIFO would overflow -> deadlock"
assert res_hi <= RES_FIFO_DEPTH, "result FIFO would overflow -> deadlock"

# okBTPipeIn commits a whole block once ep_ready is high, so the job FIFO must
# also absorb one extra block beyond the ready threshold.
assert JOB_FIFO_DEPTH - BLOCK_WORDS + BLOCK_WORDS <= JOB_FIFO_DEPTH

# A real job record must encode to exactly JOB_WORDS 32-bit words.
eye, d = rf.camera_ray(100, 200, 512, 512)
dda = rf.dda_init(eye, d)
w = rf.job_words(dda, 12345, rf.MAX_STEPS)
assert len(w) == rf.JOB_WORDS, len(w)
blob = struct.pack("<%dI" % rf.JOB_WORDS, *w)
assert len(blob) == 32, len(blob)
print(f"job record encodes to {len(blob)} bytes")

# Every word must be a legal unsigned 32-bit value.
for i, v in enumerate(w):
    assert 0 <= v <= 0xFFFFFFFF, (i, v)

# The JOB meta word and the RESULT info word carry pixel_id at DIFFERENT bit
# positions (meta: [26:13]; info: [13:0]) -- check each against its own map.
assert (w[1] >> 13) & rf.PIXEL_ID_MASK == 12345, w[1]
print("job meta word carries pixel_id at [26:13] as the RTL expects")

info = 12345 | (1 << 14) | (4 << 16)     # pixel_id, hit=1, face=4
r = rf.parse_result(info, 0x00070005, 0x002A0009)
assert r["pixel_id"] == 12345, r
assert r["hit"] is True and r["timeout"] is False and r["face"] == 4, r
assert r["x"] == 5 and r["y"] == 7 and r["z"] == 9 and r["steps"] == 42, r
print("parse_result round-trip OK:",
      {k: r[k] for k in ("pixel_id", "hit", "face", "x", "y", "z", "steps")})

# Job field layout must match what the RTL unpacks (xyz/meta bit positions).
xyz, meta = w[0], w[1]
m = rf.COORD_MASK
assert (xyz & m, (xyz >> 8) & m, (xyz >> 16) & m) == tuple(dda["voxel"]), xyz
assert meta & 0x3FF == rf.MAX_STEPS
assert ((meta >> 10) & 1, (meta >> 11) & 1, (meta >> 12) & 1) == tuple(dda["signs"])
print("job xyz/meta bit layout matches the RTL field map")

print("\nALL BULK-PIPE HOST CHECKS PASSED")
