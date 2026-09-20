"""Evaluate a pinned checkpoint in one seven-level, level-reset-only LS20 session.

No training, search, teacher, stall escape, repeated fresh games or fragment union.
Existing checkpoints use movement argmax plus an explicit GAME_OVER RESET adapter;
voluntary reset learning and persistent game memory are not implemented.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_memory_reserve():
    """Keep the user's six-GiB host-memory reserve before and after model load."""
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 6 * 1024 ** 3:
        raise RuntimeError("evaluation requires at least 6 GiB available host memory")
    return available


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU threads for batch-one inference (default: 1)")
    caps = parser.add_mutually_exclusive_group(required=True)
    caps.add_argument("--per-level-max-actions", type=int)
    caps.add_argument("--foundation-human-baseline-caps", action="store_true",
                      help="Use 5 times each vendored human baseline; all retries share this cap")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.checkpoint_sha256):
        parser.error("--checkpoint-sha256 must be exactly 64 hexadecimal characters")
    if args.per_level_max_actions is not None and args.per_level_max_actions <= 0:
        parser.error("--per-level-max-actions must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.report_out.exists():
        raise FileExistsError(args.report_out)
    checkpoint = args.checkpoint.resolve()
    expected = args.checkpoint_sha256.lower()
    if digest(checkpoint) != expected:
        raise ValueError("checkpoint SHA256 does not match the required exact hash")
    memory_before = require_memory_reserve()
    import torch
    torch.set_num_threads(args.torch_threads)
    from arcengine import base_game
    from pebby.agent import competition, model
    from pebby.agent import history, world_model
    from pebby.ls20 import shipped
    from pebby.ls20 import env as environment_module
    from pebby.ls20.env import UPSTREAM
    from pebby.agent import world_position_recall as position
    from pebby.agent import neural_outcome_policy as outcomes
    from pebby.agent import neural_outcome_planner
    from pebby.agent import spatial_outcome_policy as spatial_outcomes
    from pebby.agent import spatial_outcome_planner
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if metadata.get("format") == spatial_outcomes.FORMAT:
        load = spatial_outcomes.load_checkpoint
    elif metadata.get("format") == outcomes.FORMAT:
        load = outcomes.load_checkpoint
    elif metadata.get("format") == position.POSITION_RECALL_FORMAT:
        load = position.load_checkpoint
    else:
        load = model.load_checkpoint
    del metadata
    policy, metadata = load(checkpoint, device=args.device)
    memory_after = require_memory_reserve()
    level_caps = ([5 * baseline for baseline in shipped.HUMAN_BASELINE]
                  if args.foundation_human_baseline_caps else [args.per_level_max_actions] * shipped.LEVEL_COUNT)
    # Captured before constructing the only official game instance.
    sources = [Path(__file__).resolve(), UPSTREAM]
    sources += [Path(module.__file__).resolve() for module in
                (competition, model, history, world_model, position, outcomes,
                 neural_outcome_planner, spatial_outcomes, spatial_outcome_planner,
                 environment_module, shipped, base_game)]
    hashes = {str(path): digest(path) for path in sources}
    session = competition.CompetitionSession()
    if session.level_count != shipped.LEVEL_COUNT:
        raise ValueError("expected exactly seven official LS20 levels")
    report = competition.run_competition(competition.FourMovementDecision(policy, args.device), session,
                                         per_level_caps=level_caps)
    if digest(checkpoint) != expected or any(digest(path) != sha for path, sha in hashes.items()):
        raise RuntimeError("checkpoint or evaluator sources changed during evaluation")
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=expected,
                  checkpoint_format=metadata.get("format"), checkpoint_config=policy.config(),
                  source_sha256=hashes, source_unchanged=True,
                  execution=dict(device=args.device, torch_version=str(torch.__version__),
                                 arcengine_version=version("arcengine"), torch_threads=torch.get_num_threads(),
                                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                 precision="FP32", compiled=False),
                  budget_basis="5x_vendored_human_baseline_per_level" if args.foundation_human_baseline_caps
                               else "explicit_per_level_action_cap",
                  human_baseline_actions=list(shipped.HUMAN_BASELINE),
                  memory_available_before_load=memory_before, memory_available_after_load=memory_after,
                  host_memory_reserve_bytes=6 * 1024 ** 3,
                  limitations=["Local protocol parity; no official API scorecard or verified benchmark score.",
                               "Four movement outputs cannot choose voluntary RESET; GAME_OVER reset is a protocol adapter.",
                               "History clears at boundaries; persistent game memory is not implemented."])
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"completed": report["completed"], "levels_completed": report["levels_completed"],
                      "actions": report["actions"], "resets": report["resets"],
                      "report": str(args.report_out)}))
    return report


if __name__ == "__main__":
    main()
