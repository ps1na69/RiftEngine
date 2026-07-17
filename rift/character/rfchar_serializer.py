"""
rift/character/rfchar_serializer.py

Binary reader/writer for the .rfchar format described in
RIFTENGINE_SPECIFICATION.md. The spec gives the field list but leaves the
exact framing of "variable length string" / "variable" PNG blobs
unspecified -- this module makes that concrete, documented here since
it's a decision, not just an implementation detail:

    string       := uint16 byte-length, then UTF-8 bytes
    blob (icon)  := uint32 byte-length, then raw bytes (0 length = no icon)
    frame PNG    := uint32 byte-length, then zlib-compressed PNG bytes
                    (the spec calls for PNG data that is itself zlib
                    compressed on top of PNG's own compression -- redundant
                    in practice, but that's what's written down, so that's
                    what this does)

File layout, in order:
    magic        4 bytes   b"RFCH"
    version      uint16
    num_slots    uint16
    name         string
    size         2x float32
    offset       2x float32
    [slots] x num_slots:
        slot name     string
        frame_count   uint16
        speed         float32 (fps)
        looping       uint8 (0/1)
        icon          blob
        [frames] x frame_count:
            duration_ms   float32
            png           frame PNG
    crc32        uint32   (CRC32 of every byte before this field)

Fully round-trip tested against synthetic PNGs (see delivery notes) --
unlike the renderer, none of this needs a GPU to verify.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

from PIL import Image

from .animation_slot import AnimationFrame, AnimationSlot

MAGIC = b"RFCH"
VERSION = 1


class RFCharFormatError(ValueError):
    pass


# -- low-level primitives ---------------------------------------------------

def _write_string(buf: io.BytesIO, s: str) -> None:
    data = s.encode("utf-8")
    if len(data) > 0xFFFF:
        raise RFCharFormatError(f"string too long for uint16 length prefix: {len(data)} bytes")
    buf.write(struct.pack("<H", len(data)))
    buf.write(data)


def _read_string(buf: io.BytesIO) -> str:
    (length,) = struct.unpack("<H", buf.read(2))
    return buf.read(length).decode("utf-8")


def _write_blob(buf: io.BytesIO, data: bytes) -> None:
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)


def _read_blob(buf: io.BytesIO) -> bytes:
    (length,) = struct.unpack("<I", buf.read(4))
    return buf.read(length)


def _encode_frame_png(rgba_bytes: bytes, width: int, height: int) -> bytes:
    image = Image.frombytes("RGBA", (width, height), rgba_bytes)
    png_buf = io.BytesIO()
    image.save(png_buf, format="PNG")
    return zlib.compress(png_buf.getvalue())


def _decode_frame_png(compressed: bytes) -> tuple[bytes, int, int]:
    png_bytes = zlib.decompress(compressed)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return image.tobytes(), image.width, image.height


# -- public API ---------------------------------------------------------------

def save_rfchar(
    path: str | Path,
    name: str,
    size: tuple[float, float],
    offset: tuple[float, float],
    slots: dict[str, AnimationSlot],
) -> None:
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<H", VERSION))
    buf.write(struct.pack("<H", len(slots)))

    _write_string(buf, name)
    buf.write(struct.pack("<2f", *size))
    buf.write(struct.pack("<2f", *offset))

    for slot in slots.values():
        _write_string(buf, slot.name)
        buf.write(struct.pack("<H", slot.frame_count))
        buf.write(struct.pack("<f", slot.speed_fps))
        buf.write(struct.pack("<B", 1 if slot.looping else 0))
        _write_blob(buf, slot.icon or b"")

        for frame in slot.frames:
            buf.write(struct.pack("<f", frame.duration_ms))
            compressed = _encode_frame_png(frame.rgba_bytes, frame.width, frame.height)
            _write_blob(buf, compressed)

    payload = buf.getvalue()
    checksum = zlib.crc32(payload) & 0xFFFFFFFF

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
        f.write(struct.pack("<I", checksum))


def load_rfchar(path: str | Path) -> tuple[str, tuple[float, float], tuple[float, float], dict[str, AnimationSlot]]:
    raw = Path(path).read_bytes()
    if len(raw) < 4 + 2 + 2 + 4:
        raise RFCharFormatError(f"{path} is too small to be a valid .rfchar file")

    payload, stored_checksum_bytes = raw[:-4], raw[-4:]
    (stored_checksum,) = struct.unpack("<I", stored_checksum_bytes)
    actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_checksum != stored_checksum:
        raise RFCharFormatError(
            f"{path}: CRC32 mismatch (file is corrupt or truncated) -- "
            f"expected {stored_checksum:#010x}, got {actual_checksum:#010x}"
        )

    buf = io.BytesIO(payload)
    magic = buf.read(4)
    if magic != MAGIC:
        raise RFCharFormatError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")

    (version,) = struct.unpack("<H", buf.read(2))
    if version != VERSION:
        raise RFCharFormatError(f"{path}: unsupported .rfchar version {version} (this reader supports {VERSION})")

    (num_slots,) = struct.unpack("<H", buf.read(2))
    name = _read_string(buf)
    size = struct.unpack("<2f", buf.read(8))
    offset = struct.unpack("<2f", buf.read(8))

    slots: dict[str, AnimationSlot] = {}
    for _ in range(num_slots):
        slot_name = _read_string(buf)
        (frame_count,) = struct.unpack("<H", buf.read(2))
        (speed,) = struct.unpack("<f", buf.read(4))
        (looping_byte,) = struct.unpack("<B", buf.read(1))
        icon = _read_blob(buf) or None

        frames = []
        for _ in range(frame_count):
            (duration_ms,) = struct.unpack("<f", buf.read(4))
            compressed_png = _read_blob(buf)
            rgba_bytes, width, height = _decode_frame_png(compressed_png)
            frames.append(AnimationFrame(duration_ms=duration_ms, rgba_bytes=rgba_bytes, width=width, height=height))

        slots[slot_name] = AnimationSlot(
            name=slot_name, speed_fps=speed, looping=bool(looping_byte), frames=frames, icon=icon
        )

    return name, size, offset, slots
