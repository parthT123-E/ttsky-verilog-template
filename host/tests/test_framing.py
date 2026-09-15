"""Unit-test the result framing logic against synthetic word streams, so the
recovery path is verified without hardware."""
import sys

sys.path.insert(0, r"c:\GitHub_Projects\ASIC\ttsky-ASIC\host")
import render_fpga as rf


def records(n, start_seq=0, start_pid=0):
    """Well-formed result words: info, xy, z_steps, tag."""
    w = []
    for i in range(n):
        pid = start_pid + i
        w += [pid | (1 << 14) | (2 << 16),                    # info: hit,face2
              5 | (7 << 16),                                  # xy
              9 | (42 << 16),                                 # z_steps
              (rf.RES_TAG_MAGIC << 24) | (start_seq + i)]     # tag
    return w


class Fake(rf.RaytracerDevice):
    def __init__(self):
        self._res_seq = 0
        self._res_words = []
        self._res_framed = False
        self.lost_results = 0


def case(name, words, want_pid, want_tag):
    d = Fake()
    d._res_words = list(words)
    d._frame_results()
    assert d._res_framed, f"{name}: framing not established"
    info, xy, zs, tag = d._res_words[:4]
    pid = info & rf.PIXEL_ID_MASK
    assert rf.is_tag(tag), f"{name}: word 3 is not a tag ({tag:08x})"
    seq = rf.tag_seq(tag)
    assert seq == want_tag, f"{name}: first tag {seq}, want {want_tag}"
    assert pid == want_pid, f"{name}: first pixel_id {pid}, want {want_pid}"
    assert d._res_seq == want_tag, f"{name}: seq not resynced ({d._res_seq})"
    print(f"  {name:34s} -> framed at pixel_id={pid}, seq={seq}, "
          f"lost={d.lost_results}")


print("framing recovery tests")
# Aligned stream: nothing to skip, starts at record 0.
case("aligned", records(4), want_pid=0, want_tag=0)

# One word lost at the head: record 0 is unrecoverable, so the stream
# legitimately resumes at record 1 / tag 1.
case("one word dropped at head", records(4)[1:], want_pid=1, want_tag=1)

# One stray word inserted at the head: nothing lost, still starts at 0.
case("one stray word inserted", [0xDEADBEEF] + records(4),
     want_pid=0, want_tag=0)

# A stream that can never be framed must raise, not silently mis-parse.
d = Fake()
d._res_words = [0x11111111] * 12
try:
    d._frame_results()
    raise AssertionError("expected framing failure")
except RuntimeError as exc:
    assert "Could not establish" in str(exc)
    print("  unframeable stream                 -> raised as expected")

# Not enough data yet: must wait rather than guess.
d = Fake()
d._res_words = records(1)
d._frame_results()
assert not d._res_framed
print("  single record                      -> defers until 2 records seen")

print("\nALL FRAMING TESTS PASSED")
