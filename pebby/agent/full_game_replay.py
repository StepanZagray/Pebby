"""Bounded replay of public H8 observations paired with cached outcome labels.

Outcome row IDs are not raw observation row IDs. Only ``rows.npy`` performs that
translation; labels always use the original outcome indices. No cached encoder
features, successor images, bank geometry, or solver information enter ``batch``.
"""

import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

PUBLIC = ("frames", "history_valid", "previous_actions")
TARGET_SCHEMA = {
    "optimal": ("uint8", ()), "next_optimal": ("uint8", (4,)),
    "next_player_cell": ("int16", (4, 2)), "next_triple": ("int16", (4, 3)),
    "next_steps": ("int16", (4,)), "next_lives": ("int16", (4,)),
    "distances": ("int16", (4,)), "lost_life": ("bool", (4,)),
    "terminal": ("bool", (4,)), "won": ("bool", (4,)),
    "player_cell": ("int16", (2,)), "current_triple": ("int16", (3,)),
    "current_steps": ("int16", ()), "current_lives": ("int16", ()),
}
PUBLIC_SCHEMA = {
    "frames": ("uint8", (8, 64, 64)),
    "history_valid": ("bool", (8,)), "previous_actions": ("int64", (8,)),
}


def _stat(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


class RawOutcomeReplay:
    """Read-only mappings with independent, bounded tensor copies per batch.

    Default verification hashes all opened small arrays and both manifests, but
    checks recorded inode/size/timestamps for the large frames array. Device
    number changes across mounts are recorded and allowed. This
    is weaker than rehashing pixels; ``verify_hashes=True`` also hashes frames.
    The bank is read line by line for seed/difficulty only and bound by its hash
    for this run, not cryptographically tied to the outcome-cache publication.

    Sampling chooses a tier, then a level uniformly, then a root. An event draw
    prefers that level's event roots if any; otherwise it falls back to all roots
    of that same level. Thus level balance survives event oversampling, while
    ``event_fraction`` is a mixture probability, not an exact batch quota.
    """

    def __init__(self, cache_root, split, verify_hashes=False, guard=None, *,
                 source_root=None, bank_path=None, max_batch_size=4096,
                 drop_pages_every=32):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not isinstance(drop_pages_every, int) or drop_pages_every < 0:
            raise ValueError("drop_pages_every must be nonnegative")
        self._arrays, self._stats, self._hashes = {}, {}, {}
        self._closed, self._batches = False, 0
        self.guard = guard
        self.max_batch_size, self.drop_pages_every = max_batch_size, drop_pages_every
        self.cache_root, self.split = Path(cache_root).resolve(), split
        try:
            manifest_path = self.cache_root / "manifest.json"
            self._bind(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("status") != "complete" or not manifest.get("sources_unchanged")
                    or not manifest.get("validation_disjoint")
                    or manifest.get("public_inputs") != list(PUBLIC)
                    or manifest.get("current_source_id") != 0 or manifest.get("current_branch") != -1
                    or manifest.get("official_frames_or_routes_used") is not False):
                raise ValueError("outcome cache is not a verified generated current-public cache")
            bindings = manifest["source_sha256"]
            source_sha = [digest for path, digest in bindings.items()
                          if Path(path).name == f"{split}.npz"]
            if len(source_sha) != 1:
                raise ValueError("ambiguous source NPZ binding")
            candidates = [Path(path).parent for path in bindings
                          if Path(path).name == "manifest.json"
                          and Path(path).parent.name.startswith(source_sha[0] + "-")]
            if len(candidates) != 1:
                raise ValueError("ambiguous raw-public source binding")
            original_source = candidates[0]
            self.source_root = (Path(source_root).resolve() if source_root is not None else
                                self.cache_root.parent / "reference-world-base-v1" / "array-cache" / original_source.name)
            source_manifest_path = self.source_root / "manifest.json"
            self._bind(source_manifest_path, bindings[str(original_source / "manifest.json")])
            source_manifest = json.loads(source_manifest_path.read_text())
            if source_manifest["source_sha256"] != source_sha[0]:
                raise ValueError("raw source manifest SHA256 disagrees with outcome source")
            for key in ("rows", "seeds", *TARGET_SCHEMA):
                info = manifest["arrays"][split][key]
                self._open(key, self.cache_root / split / f"{key}.npy", info)
            self.rows, self.seeds = self._arrays["rows"], self._arrays["seeds"]
            count = len(self.rows)
            for key, (dtype, tail) in {"rows": ("int64", ()), "seeds": ("int64", ()), **TARGET_SCHEMA}.items():
                self._schema(self._arrays[key], key, count, dtype, tail)
            selection = manifest["selection"][split]
            if (count == 0 or count != selection["rows"] or len(np.unique(self.rows)) != count
                    or np.any(self.rows < 0)
                    or hashlib.sha256(self.rows.tobytes()).hexdigest() != selection["source_rows_sha256"]):
                raise ValueError("invalid published outcome row selection")
            device_changes = {}
            for key in (*PUBLIC, "seeds"):
                original_path = str(original_source / f"{key}.npy")
                path = self.source_root / f"{key}.npy"
                info = source_manifest["arrays"][key]
                if info["sha256"] != bindings[original_path]:
                    raise ValueError(f"raw source hash binding disagrees: {key}")
                recorded = manifest["source_stats_after"][original_path]
                current = _stat(path)
                if current[1:] != recorded[1:]:
                    raise ValueError(f"raw source stats changed: {key}")
                if current[0] != recorded[0]:
                    device_changes[key] = {"recorded": recorded[0], "current": current[0]}
                self._open("source_" + key, path, info, hash_file=verify_hashes or key != "frames")
            source_count = len(self._arrays["source_seeds"])
            for key, (dtype, tail) in PUBLIC_SCHEMA.items():
                self._schema(self._arrays["source_" + key], key, source_count, dtype, tail)
            source_seeds = self._arrays["source_seeds"]
            if source_seeds.ndim != 1 or source_seeds.dtype.kind not in "iu":
                raise ValueError("invalid source seed schema")
            if np.any(self.rows >= source_count) or not np.array_equal(source_seeds[self.rows], self.seeds):
                raise ValueError("outcome/source row and seed alignment mismatch")
            unique, counts = np.unique(self.seeds, return_counts=True)
            if len(unique) != selection["levels"] or not np.all(counts == selection["roots_per_level"]):
                raise ValueError("outcome selection level coverage mismatch")
            self.bank_path = (Path(bank_path).resolve() if bank_path is not None else
                              self.cache_root.parent / "ls20-reference-unequal-v1" / f"{split}.jsonl")
            self._bind(self.bank_path)
            tiers = {}
            with self.bank_path.open() as handle:
                for line in handle:
                    self._guard()
                    row = json.loads(line)
                    seed, tier = row["seed"], row["difficulty"]
                    if (type(seed) is not int or type(tier) is not int or not 1 <= tier <= 7
                            or seed in tiers):
                        raise ValueError("invalid or duplicate bank seed/difficulty")
                    tiers[seed] = tier
            if set(map(int, unique)) != set(tiers):
                raise ValueError("bank and replay level coverage differ")
            self.tiers = np.array([tiers[int(seed)] for seed in self.seeds], dtype=np.int8)
            self.tiers.flags.writeable = False
            a = self._arrays
            self.events = (a["lost_life"].any(1) | a["terminal"].any(1) | a["won"].any(1)
                           | (a["next_triple"] != a["current_triple"][:, None, :]).any((1, 2))
                           | (a["next_steps"] > a["current_steps"][:, None]).any(1))
            self.events.flags.writeable = False
            self._level_rows, self._event_rows, self._tier_seeds = {}, {}, {}
            order = np.argsort(self.seeds, kind="stable")
            for seed, indices in zip(unique, np.split(order, np.cumsum(counts)[:-1])):
                self._level_rows[int(seed)] = indices
                self._event_rows[int(seed)] = indices[self.events[indices]]
            for tier in sorted(set(tiers.values())):
                self._tier_seeds[tier] = np.array([s for s in unique if tiers[int(s)] == tier])
            self.metadata = dict(
                split=split, rows=count, levels=len(unique), public_inputs=list(PUBLIC),
                target_names=list(TARGET_SCHEMA), source_root=str(self.source_root),
                cache_root=str(self.cache_root), bank_path=str(self.bank_path),
                source_sha256=source_sha[0], max_batch_size=max_batch_size,
                source_device_number_changes=device_changes,
                source_stat_fields_checked=["inode", "size", "mtime_ns", "ctime_ns"],
                pixel_binding="sha256_and_recorded_stats" if verify_hashes else "recorded_stats_only",
                verification_caveat=("Raw pixels freshly hashed." if verify_hashes else
                                     "Raw pixels were not rehashed; recorded inode/size/timestamps are a weaker binding; mount device changes are allowed."),
                bank_binding="sha256_at_open; not bound by outcome publication",
                verified_sha256=dict(self._hashes), bound_stats=dict(self._stats),
                zero_optimal_rows_retained=int(np.count_nonzero(a["optimal"] == 0)),
                event_rows=int(self.events.sum()),
                tier_levels={str(tier): len(seeds) for tier, seeds in self._tier_seeds.items()},
                event_definition="life loss, terminal, win, triple change, or step-budget increase",
                event_sampling="within uniformly sampled level; fall back to that level's roots if no events",
            )
            self.verify_unchanged()
        except BaseException:
            self.close()
            raise

    def _guard(self):
        if self.guard is not None:
            self.guard()

    def _bind(self, path, expected=None, *, hash_file=True):
        self._guard()
        before = _stat(path)
        if hash_file:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self._guard()
                    digest.update(chunk)
            value = digest.hexdigest()
            if expected is not None and value != expected:
                raise ValueError(f"SHA256 mismatch: {path}")
            self._hashes[str(path)] = value
        if _stat(path) != before:
            raise ValueError(f"source changed while opening: {path}")
        self._stats[str(path)] = before

    def _open(self, key, path, info, *, hash_file=True):
        self._bind(path, info["sha256"], hash_file=hash_file)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        self._arrays[key] = value
        if list(value.shape) != info["shape"] or value.dtype.str != info["dtype"]:
            raise ValueError(f"array schema mismatch: {path}")

    @staticmethod
    def _schema(value, key, count, dtype, tail):
        if value.shape != (count, *tail) or value.dtype != np.dtype(dtype):
            raise ValueError(f"array schema mismatch: {key}")

    def __len__(self):
        return len(self.rows)

    def verify_unchanged(self):
        if self._closed:
            raise RuntimeError("replay is closed")
        self._guard()
        for path, before in self._stats.items():
            if _stat(Path(path)) != before:
                raise ValueError(f"replay input stats changed: {path}")

    def batch(self, outcome_indices, device="cpu"):
        """Return three public tensors and independent outcome-label tensors."""
        import torch
        self.verify_unchanged()
        indices = np.asarray(outcome_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("outcome indices must be a one-dimensional integer array")
        if len(indices) > self.max_batch_size:
            raise ValueError("batch exceeds max_batch_size")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("outcome index out of bounds")
        raw_rows = self.rows[indices]
        public = {key: torch.from_numpy(np.array(self._arrays["source_" + key][raw_rows], copy=True)).to(device)
                  for key in PUBLIC}
        targets = {key: torch.from_numpy(np.array(self._arrays[key][indices], copy=True)).to(device)
                   for key in TARGET_SCHEMA}
        self.verify_unchanged()
        self._batches += 1
        if self.drop_pages_every and self._batches % self.drop_pages_every == 0:
            self.release_pages()
        return public, targets

    def sample(self, batch_size, rng, tier_weights=None, event_fraction=.25):
        """Return outcome indices, with replacement; never filter zero policy masks."""
        self.verify_unchanged()
        if (not isinstance(batch_size, int) or not 0 < batch_size <= self.max_batch_size
                or not np.isfinite(event_fraction) or not 0 <= event_fraction <= 1):
            raise ValueError("invalid batch size or event_fraction")
        available = np.array(sorted(self._tier_seeds))
        if tier_weights is None:
            weights = np.ones(len(available))
        else:
            if not set(tier_weights).issubset(set(available)):
                raise ValueError("tier_weights contains an unavailable tier")
            weights = np.array([tier_weights.get(int(tier), 0.) for tier in available], dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.max() <= 0:
            raise ValueError("tier weights must be finite, nonnegative, and have positive mass")
        weights /= weights.max()
        tiers = rng.choice(available, size=batch_size, p=weights / weights.sum())
        indices = np.empty(batch_size, dtype=np.int64)
        for i, tier in enumerate(tiers):
            seed = int(rng.choice(self._tier_seeds[int(tier)]))
            rows = self._level_rows[seed]
            if rng.random() < event_fraction and len(self._event_rows[seed]):
                rows = self._event_rows[seed]
            indices[i] = rng.choice(rows)
        return indices

    def release_pages(self):
        """Drop clean mmap pages when supported; tensor batches own their memory."""
        for value in self._arrays.values():
            backing = getattr(value, "_mmap", None)
            if backing is not None and not backing.closed and hasattr(backing, "madvise"):
                backing.madvise(mmap.MADV_DONTNEED)

    def close(self):
        """Idempotently close all mappings, including after partial construction."""
        self._closed = True
        for value in self._arrays.values():
            backing = getattr(value, "_mmap", None)
            if backing is not None and not backing.closed:
                backing.close()
        self._arrays.clear()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("replay is closed")
        return self

    def __exit__(self, *_):
        self.close()
