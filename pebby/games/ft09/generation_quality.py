"""Canonical novelty identities for FT09 procedural levels."""

from __future__ import annotations

import hashlib
import json


SPLITS = ("train", "validation", "test")


def _transforms():
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                yield swap, sx, sy


def _point(x, y, transform):
    swap, sx, sy = transform
    if swap:
        x, y = y, x
    return sx * x, sy * y


def _stencil_offsets(stencil, transform):
    offsets = []
    for row in range(3):
        for col in range(3):
            if int(stencil[row][col]):
                offsets.append(_point(col - 1, row - 1, transform))
    return tuple(sorted(offsets))


def _canonical_payload(spec, *, gameplay):
    variants = []
    palette = tuple(spec.get("palette", ()))
    palette_index = {int(colour): index for index, colour in enumerate(palette)}
    for transform in _transforms():
        anchors = []
        raw = []
        for cell in spec.get("cells", ()):
            x, y = _point(int(cell["x"]), int(cell["y"]), transform)
            anchors.append((x, y))
            if "stencil" in cell:
                raw.append(("special", x, y, _stencil_offsets(cell["stencil"], transform)))
            else:
                raw.append(("cell", x, y))
        for rule in spec.get("constraints", ()):
            x, y = _point(int(rule["x"]), int(rule["y"]), transform)
            anchors.append((x, y))
            if gameplay:
                relations = []
                for row in range(3):
                    for col in range(3):
                        if (row, col) == (1, 1):
                            continue
                        dx, dy = _point(col - 1, row - 1, transform)
                        relations.append((dx, dy, int(rule["mask"][row][col]) == 0))
                raw.append(("rule", x, y, palette_index[int(rule["colour"])], tuple(sorted(relations))))
            else:
                raw.append(("rule", x, y))
        if not anchors:
            raise ValueError("cannot fingerprint empty FT09 geometry")
        left = min(x for x, _ in anchors)
        top = min(y for _, y in anchors)
        normalized = []
        for item in raw:
            normalized.append((item[0], item[1] - left, item[2] - top, *item[3:]))
        prefix = ()
        if gameplay:
            prefix = (
                ("palette_size", len(palette)),
                ("budget", int(spec["budget"])),
                ("global_stencil", _stencil_offsets(spec["stencil"], transform)),
            )
        variants.append(json.dumps((prefix, sorted(normalized)), separators=(",", ":")))
    return min(variants)


def geometry_hash(spec):
    """Role-aware identity invariant to translation, rotation, and reflection."""
    return hashlib.sha256(_canonical_payload(spec, gameplay=False).encode()).hexdigest()


def gameplay_hash(spec):
    """Semantic identity also normalizing cosmetic palette-value relabeling."""
    return hashlib.sha256(_canonical_payload(spec, gameplay=True).encode()).hexdigest()


def identity_partition(identity):
    return SPLITS[int(identity, 16) % len(SPLITS)]
