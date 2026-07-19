"""
rift/level/rflevel_serializer.py

Binary reader/writer for the .rflevel format described in
RIFTENGINE_SPECIFICATION.md. Same situation as .rfchar: the spec lists the
fields but not the exact byte framing, and "Custom data: JSON or binary
blob" is explicitly an either/or in the spec itself. This picks JSON
(human-diffable, and every actor's custom data is small key/value stuff --
health, damage, script refs -- so the binary-compactness argument for a
custom encoding doesn't really apply here) and documents the framing:

    string        := uint16 byte-length, then UTF-8 bytes
    custom_data   := uint32 byte-length, then UTF-8 JSON bytes
                     (empty object `{}` if an actor has none)

File layout, in order:
    magic         4 bytes   b"RFLV"
    version       uint16
    num_actors    uint16
    name          string
    size          2x float32
    [actors] x num_actors:
        actor_type      string   ("Character", "Sprite", "Light", ...)
        position        2x float32
        rotation        float32
        scale           2x float32
        resource_path   string   (path to .rfchar or texture; may be empty)
        custom_data     custom_data
    crc32         uint32   (CRC32 of every byte before this field)
"""

from __future__ import annotations

import io
import json
import struct
import zlib
from pathlib import Path

from .actor_manager import Actor

MAGIC = b"RFLV"
VERSION = 1


class RFLevelFormatError(ValueError):
    pass


# -- low-level primitives ---------------------------------------------------

def _write_string(buf: io.BytesIO, s: str) -> None:
    data = s.encode("utf-8")
    if len(data) > 0xFFFF:
        raise RFLevelFormatError(f"string too long for uint16 length prefix: {len(data)} bytes")
    buf.write(struct.pack("<H", len(data)))
    buf.write(data)


def _read_string(buf: io.BytesIO) -> str:
    (length,) = struct.unpack("<H", buf.read(2))
    return buf.read(length).decode("utf-8")


def _write_custom_data(buf: io.BytesIO, data: dict) -> None:
    encoded = json.dumps(data, separators=(",", ":")).encode("utf-8")
    buf.write(struct.pack("<I", len(encoded)))
    buf.write(encoded)


def _read_custom_data(buf: io.BytesIO) -> dict:
    (length,) = struct.unpack("<I", buf.read(4))
    raw = buf.read(length)
    return json.loads(raw) if raw else {}


# -- public API ---------------------------------------------------------------

def save_rflevel(path: str | Path, name: str, size: tuple[float, float], actors: list[Actor]) -> None:
    buf = io.BytesIO()
    buf.write(MAGIC)
    buf.write(struct.pack("<H", VERSION))
    buf.write(struct.pack("<H", len(actors)))

    _write_string(buf, name)
    buf.write(struct.pack("<2f", *size))

    for actor in actors:
        _write_string(buf, actor.actor_type)
        buf.write(struct.pack("<2f", *actor.position))
        buf.write(struct.pack("<f", actor.rotation))
        buf.write(struct.pack("<2f", *actor.scale))
        _write_string(buf, actor.resource_path or "")
        _write_custom_data(buf, actor.custom_data or {})

    payload = buf.getvalue()
    checksum = zlib.crc32(payload) & 0xFFFFFFFF

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)
        f.write(struct.pack("<I", checksum))


def load_rflevel(path: str | Path) -> tuple[str, tuple[float, float], list[Actor]]:
    raw = Path(path).read_bytes()
    if len(raw) < 4 + 2 + 2 + 4:
        raise RFLevelFormatError(f"{path} is too small to be a valid .rflevel file")

    payload, stored_checksum_bytes = raw[:-4], raw[-4:]
    (stored_checksum,) = struct.unpack("<I", stored_checksum_bytes)
    actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_checksum != stored_checksum:
        raise RFLevelFormatError(
            f"{path}: CRC32 mismatch (file is corrupt or truncated) -- "
            f"expected {stored_checksum:#010x}, got {actual_checksum:#010x}"
        )

    buf = io.BytesIO(payload)
    magic = buf.read(4)
    if magic != MAGIC:
        raise RFLevelFormatError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")

    (version,) = struct.unpack("<H", buf.read(2))
    if version != VERSION:
        raise RFLevelFormatError(f"{path}: unsupported .rflevel version {version} (this reader supports {VERSION})")

    (num_actors,) = struct.unpack("<H", buf.read(2))
    name = _read_string(buf)
    size = struct.unpack("<2f", buf.read(8))

    actors: list[Actor] = []
    for _ in range(num_actors):
        actor_type = _read_string(buf)
        position = struct.unpack("<2f", buf.read(8))
        (rotation,) = struct.unpack("<f", buf.read(4))
        scale = struct.unpack("<2f", buf.read(8))
        resource_path = _read_string(buf)
        custom_data = _read_custom_data(buf)

        actors.append(Actor(
            name=custom_data.get("name", f"{actor_type}_{len(actors)}"),
            actor_type=actor_type,
            position=position,
            rotation=rotation,
            scale=scale,
            resource_path=resource_path,
            custom_data=custom_data,
        ))

    return name, size, actors
