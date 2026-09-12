"""Pretrain ``glyph_model.GlyphEncoder`` on generated-only carried-glyph crops.

Data contract (NPZ per split, written by ``tools/build_glyph_data.py``)
----------------------------------------------------------------------
``glyphs uint8[N, 6, 6]`` palette indices of the public crop rows 55..60,
columns 3..8; ``triples uint8[N, 3]`` shape 0..5 / colour 0..3 / rotation
0..3; ``seeds int32[N]`` level seed of the source state; ``views uint8[N]``
0 = current frame, 1..4 = the actual successor of each action; ``meta`` JSON
with ``format == "pebby.ls20-glyphs.v1"``, ``source == "generated_only"``,
the crop and the split name.

Training unit: every optimizer step draws ``--batch-size`` DISTINCT levels
uniformly (rows are pre-indexed per seed) and one random row per chosen level,
so a level with many states does not dominate and rare glyph patterns keep the
coverage the level generator gave them. Glyph patterns are NOT deduplicated
and classes are NOT rebalanced by validation; validation only measures.

Refused: train/validation seed overlap, fewer than ``--min-train-levels``
distinct training levels, fewer distinct levels than the batch size, labels or
pixels outside their public ranges, wrong format/source/crop/split metadata,
a batch size that is not a power of two up to 1024.

    uv run python -m pebby.agent.glyph_train --train data/ls20-glyph-train.npz \
        --validation data/ls20-glyph-validation.npz --steps 400 --batch-size 1024 \
        --min-train-levels 10000 --checkpoint-out checkpoints/ls20-glyph.pt

CPU by default; the classifier is ~38k parameters. Nothing here imports the
game, an oracle or official level data.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .glyph_model import (GLYPH_COLUMNS, GLYPH_FIELDS, GLYPH_ROWS, GLYPH_SIZE, GLYPH_SIZES, GLYPH_SOURCE,
                          PALETTE, GlyphEncoder, load_glyph_checkpoint, save_glyph_checkpoint)

GLYPH_DATA_FORMAT = "pebby.ls20-glyphs.v1"
VIEWS = 5
BATCH_UNIT = "distinct_level"


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_glyph_split(path, split):
    """Read and validate one glyph NPZ split; returns numpy arrays plus ``meta``."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"missing glyph data file: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("glyphs", "triples", "seeds", "views", "meta") if name not in archive.files]
        if missing:
            raise ValueError(f"{path} lacks arrays: {missing}")
        data = {name: archive[name] for name in ("glyphs", "triples", "seeds", "views")}
        try:
            meta = json.loads(str(archive["meta"].item()))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}: meta must be a JSON string ({error})") from error
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: meta must be a JSON object")
    crop = {"rows": list(GLYPH_ROWS), "columns": list(GLYPH_COLUMNS)}
    found = meta.get("crop", {})
    if meta.get("format") != GLYPH_DATA_FORMAT or meta.get("source") != GLYPH_SOURCE:
        raise ValueError(f"{path}: expected format {GLYPH_DATA_FORMAT!r} with source {GLYPH_SOURCE!r}, "
                         f"got {meta.get('format')!r} / {meta.get('source')!r}")
    if not isinstance(found, dict) or [list(found.get(k, [])) for k in crop] != list(crop.values()):
        raise ValueError(f"{path}: crop {found} differs from the encoder crop {crop}")
    if meta.get("split") != split:
        raise ValueError(f"{path}: meta split {meta.get('split')!r} is not {split!r}")
    count = len(data["seeds"])
    if count == 0:
        raise ValueError(f"{path} holds no glyphs")
    for name, shape in (("glyphs", (count, GLYPH_SIZE, GLYPH_SIZE)), ("triples", (count, 3)),
                        ("seeds", (count,)), ("views", (count,))):
        if data[name].shape != shape or not np.issubdtype(data[name].dtype, np.integer):
            raise ValueError(f"{path}: {name} must be integer {shape}, got {data[name].dtype} {data[name].shape}")
    if data["glyphs"].min() < 0 or data["glyphs"].max() >= PALETTE:
        raise ValueError(f"{path}: glyph pixels must be palette indices 0..{PALETTE - 1}")
    if np.any(data["triples"] < 0) or np.any(data["triples"] >= np.array(GLYPH_SIZES)):
        raise ValueError(f"{path}: triples must lie inside {GLYPH_SIZES}")
    if data["views"].min() < 0 or data["views"].max() >= VIEWS:
        raise ValueError(f"{path}: views must be 0..{VIEWS - 1}")
    data["meta"] = meta
    return data


def disjoint_seeds(train, validation):
    shared = np.intersect1d(np.unique(train["seeds"]), np.unique(validation["seeds"]))
    if len(shared):
        raise ValueError(f"{len(shared)} level seed(s) appear in both glyph splits, e.g. {shared[:5].tolist()}")


class LevelSampler:
    """Rows pre-indexed per level seed; a batch is one random row of each of ``k`` distinct levels."""

    def __init__(self, seeds):
        seeds = np.asarray(seeds)
        self.rows = np.argsort(seeds, kind="stable")
        self.levels, self.starts, self.counts = np.unique(seeds[self.rows], return_index=True, return_counts=True)

    def __len__(self):
        return len(self.levels)

    def sample(self, batch_size, rng):
        if batch_size > len(self.levels):
            raise ValueError(f"a batch of {batch_size} distinct levels needs more than {len(self.levels)} levels")
        chosen = rng.choice(len(self.levels), batch_size, replace=False)
        offsets = (rng.random(batch_size) * self.counts[chosen]).astype(np.int64)
        return self.rows[self.starts[chosen] + offsets]


def evaluate(model, data, batch_size=8192):
    """Per-field, joint and per-view accuracies plus the mean loss over ALL rows of a split."""
    glyphs, triples, views = data["glyphs"], data["triples"], np.asarray(data["views"])
    hits, losses, count = [], 0., len(glyphs)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            logits = model(torch.from_numpy(np.ascontiguousarray(glyphs[start:start + batch_size])))
            labels = torch.from_numpy(np.asarray(triples[start:start + batch_size], dtype=np.int64))
            losses += float(GlyphEncoder.loss(logits, labels)) * len(logits)
            hits.append(torch.stack([scores.argmax(-1) == labels[:, index] for index, scores in
                                     enumerate(logits.split(GLYPH_SIZES, -1))], dim=-1))
    hits = torch.cat(hits)
    joint = hits.all(-1)
    per_view = {str(view): float(joint[torch.from_numpy(views == view)].float().mean())
                for view in np.unique(views)}
    return {"rows": int(count), "loss": losses / count,
            **{f"{name}_accuracy": float(hits[:, index].float().mean()) for index, name in enumerate(GLYPH_FIELDS)},
            "joint_accuracy": float(joint.float().mean()), "joint_accuracy_by_view": per_view}


def train_encoder(train, steps, batch_size, lr, seed, device, log=None):
    """Fixed-step Adam training; returns the model and the loss trace."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = LevelSampler(train["seeds"])
    model = GlyphEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    model.train()
    for step in range(1, steps + 1):
        index = sampler.sample(batch_size, rng)
        if len(np.unique(train["seeds"][index])) != batch_size:
            raise RuntimeError("batch rows are not from distinct levels")
        glyphs = torch.from_numpy(np.ascontiguousarray(train["glyphs"][index])).to(device)
        labels = torch.from_numpy(np.asarray(train["triples"][index], dtype=np.int64)).to(device)
        loss = GlyphEncoder.loss(model(glyphs), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == steps or step % 20 == 0:
            trace.append({"step": step, "loss": float(loss.detach())})
            if log:
                log(f"step {step}/{steps} loss {float(loss.detach()):.4f}")
    return model.eval(), trace


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True, help="Disjoint level seeds; measures only")
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/ls20-glyph.pt"))
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024, help="Distinct levels per step; power of two <= 1024")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-train-levels", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.batch_size < 1 or args.batch_size > 1024 or args.batch_size & (args.batch_size - 1):
        parser.error("batch-size must be a power of two, capped at 1024")
    if not 0 < args.lr < 1:
        parser.error("lr must be in (0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is False")
    torch.set_num_threads(max(1, args.threads))
    started = time.monotonic()
    try:
        train = load_glyph_split(args.train, "train")
        validation = load_glyph_split(args.validation, "validation")
        disjoint_seeds(train, validation)
        train_levels = len(np.unique(train["seeds"]))
        if train_levels < args.min_train_levels:
            raise ValueError(f"training data has {train_levels} distinct levels, fewer than --min-train-levels")
        if train_levels < args.batch_size:
            raise ValueError(f"training data has {train_levels} distinct levels, fewer than the batch size")
    except ValueError as error:
        parser.error(str(error))
    hashes = {"train": file_digest(args.train), "validation": file_digest(args.validation)}
    counts = {"train_rows": int(len(train["seeds"])), "train_levels": int(train_levels),
              "validation_rows": int(len(validation["seeds"])),
              "validation_levels": int(len(np.unique(validation["seeds"])))}
    print(f"{counts['train_rows']:,} training glyphs over {counts['train_levels']} levels | "
          f"{counts['validation_rows']:,} validation glyphs over {counts['validation_levels']} held-out levels | "
          f"{args.steps} steps x {args.batch_size} distinct levels, Adam lr {args.lr}, {args.device}", flush=True)
    model, trace = train_encoder(train, args.steps, args.batch_size, args.lr, args.seed, torch.device(args.device),
                                 log=lambda line: print(line, flush=True))
    model = model.to("cpu")
    results = {"train": evaluate(model, train), "validation": evaluate(model, validation)}
    for split, scores in results.items():
        print(f"{split}: " + " ".join(f"{name} {scores[f'{name}_accuracy']:.4f}" for name in GLYPH_FIELDS)
              + f" joint {scores['joint_accuracy']:.4f} loss {scores['loss']:.4f}", flush=True)
    metadata = {
        "source": GLYPH_SOURCE, "train": str(args.train), "validation": str(args.validation), "hashes": hashes,
        "train_seeds": sorted(int(s) for s in np.unique(train["seeds"])),
        "validation_seeds": sorted(int(s) for s in np.unique(validation["seeds"])),
        "counts": counts, "results": results, "steps": args.steps, "batch_size": args.batch_size,
        "batch_unit": BATCH_UNIT, "optimizer": {"type": "Adam", "lr": args.lr}, "seed": args.seed,
        "device": args.device, "loss_trace": trace, "data_meta": {"train": train["meta"], "validation": validation["meta"]},
        "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": time.monotonic() - started}
    checkpoint = save_glyph_checkpoint(args.checkpoint_out, model, **metadata)
    restored, _ = load_glyph_checkpoint(args.checkpoint_out)  # strict reload or no checkpoint
    with torch.inference_mode():
        probe = torch.from_numpy(np.ascontiguousarray(validation["glyphs"][:256]))
        if not torch.equal(restored(probe), model(probe)):
            raise SystemExit("reloaded glyph checkpoint does not reproduce the trained outputs")
    report = {key: value for key, value in checkpoint.items() if key != "weights"}
    report["checkpoint"] = str(args.checkpoint_out)
    path = args.report_out or args.checkpoint_out.with_suffix(".training.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"Saved {args.checkpoint_out} ({checkpoint['parameters']:,} parameters)\nReport {path}", flush=True)
    print("Held-out glyph accuracy measures perception only; nothing here is a policy result.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
