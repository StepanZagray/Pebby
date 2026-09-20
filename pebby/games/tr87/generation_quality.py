"""Canonical TR87 identities and deterministic three-way geometry holdout."""

import hashlib
import json


SPLITS = ("train", "validation", "test")


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def geometry_identity(spec):
    """Canonical object geometry under translation and all eight D4 maps.

    The identity intentionally ignores glyph/color assignments. Tile anchors
    and connector anchors remain distinguished, so it catches reskinned or
    rigidly transformed official/cross-split copies without conflating
    genuinely different packing.
    """
    points = []
    for rule in spec["rules"]:
        x = rule["x"]
        for _ in rule["lhs"]:
            points.append(("tile", x, rule["y"]))
            x += 7
        last_lhs_x = x - 7
        points.append(("strip", last_lhs_x + 2, rule["y"] + 2))
        x = last_lhs_x + 10
        for _ in rule["rhs"]:
            points.append(("tile", x, rule["y"]))
            x += 7
    for label in ("source", "target"):
        row = spec[label]
        points.extend(("tile", row["x0"] + 7 * i, row["y"])
                      for i in range(len(row["symbols"])))
    variants = []
    for swap in (False, True):
        for sx in (-1, 1):
            for sy in (-1, 1):
                mapped = [(kind, sx * (y if swap else x), sy * (x if swap else y))
                          for kind, x, y in points]
                left = min(x for _, x, _ in mapped)
                top = min(y for _, _, y in mapped)
                variants.append(sorted((kind, x - left, y - top) for kind, x, y in mapped))
    return _digest(min(variants))


def geometry_partition(spec):
    fingerprint = geometry_identity(spec)
    return fingerprint, SPLITS[int(fingerprint, 16) % len(SPLITS)]


def gameplay_identity(spec):
    """Canonical gameplay under per-family cyclic digit relabeling."""
    offsets = {}

    def normalized(symbol):
        family, digit = symbol[0], int(symbol[1])
        offsets.setdefault(family, digit)
        return family + str((digit - offsets[family]) % 7)

    value = {
        "mode": spec["mode"],
        "mechanics": spec["mechanics"],
        "rules": [([normalized(s) for s in rule["lhs"]],
                   [normalized(s) for s in rule["rhs"]]) for rule in spec["rules"]],
        "source": [normalized(s) for s in spec["source"]["symbols"]],
        "target": [normalized(s) for s in spec["target"]["symbols"]],
    }
    return _digest(value)
