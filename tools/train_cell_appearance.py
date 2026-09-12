"""Bounded generated-only visible-cell perception experiment; no game policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from pebby.agent.cell_appearance import CellAppearance, ROLE_NAMES, ATTRIBUTE_SIZES, FORMAT, cell_patches


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_examples(path, proof_path):
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('status') != 'complete' or proof['output_npz']['sha256'] != digest(path):
        raise ValueError('dataset does not match completed label audit')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    expected_mask = data['fully_visible'] & ~data['hud_overlap']
    if not np.array_equal(data['label_mask'], expected_mask):
        raise ValueError('labels must require fully public, non-HUD cells')
    if 'support7_label_mask' not in data or 'support9_label_mask' not in data:
        raise ValueError('audited wider-neighborhood visibility masks are required')
    if np.any(data['support7_label_mask'] & ~expected_mask):
        raise ValueError('7x7 support mask cannot exceed the fully visible cell mask')
    if np.any(data['support9_label_mask'] & ~data['support7_label_mask']):
        raise ValueError('9x9 support mask cannot exceed 7x7 support')
    if set(data['split']) != {'train', 'validation'}:
        raise ValueError('expected generated train and validation splits')
    if len(set(map(int, data['seeds']))) != len(data['seeds']):
        raise ValueError('initial-state level seeds must be distinct')
    patches = cell_patches(torch.from_numpy(data['frames']))
    examples = {}
    for split in ('train', 'validation'):
        rows = np.flatnonzero(data['split'] == split)
        seeds = data['seeds'][rows]
        lower, upper = (0, 1_000_000) if split == 'train' else (1_000_000, 2_000_000)
        if np.any((seeds < lower) | (seeds >= upper)):
            raise ValueError('generated seed split boundary violated')
        mask = torch.from_numpy(data['support7_label_mask'][rows])
        if not bool(mask.any(-1).all()):
            raise ValueError('every evaluated level needs a fully public neighborhood')
        role_masks = torch.from_numpy(data['roles'][rows].astype(np.int64))[mask]
        attributes = torch.from_numpy(data['goal_attrs'][rows].astype(np.int64))[mask]
        goals = role_masks.bitwise_and(2).bool()
        if not goals.any():
            raise ValueError('split has no visible goals')
        for column, size in enumerate(ATTRIBUTE_SIZES):
            if not bool(((attributes[goals, column] >= 0) & (attributes[goals, column] < size)).all()):
                raise ValueError('visible goal attribute outside category range')
        examples[split] = dict(patches=patches[rows][mask], role_masks=role_masks,
                               role_bits=role_masks[:, None].bitwise_and(1 << torch.arange(8)).ne(0),
                               attributes=attributes, goals=goals, levels=len(rows),
                               level_index=torch.arange(len(rows))[:, None].expand(-1, 144)[mask])
    return examples, proof


@torch.no_grad()
def evaluate(model, example, majority):
    output = torch.cat([model.patch_logits(p) for p in example['patches'].split(1024)])
    role_logits, *attribute_logits = output.split((8, *ATTRIBUTE_SIZES), dim=-1)
    predicted = role_logits >= 0
    labels = example['role_bits']
    role_metrics = {}
    for index, name in enumerate(ROLE_NAMES):
        actual, estimate = labels[:, index], predicted[:, index]
        tp = int((actual & estimate).sum()); fp = int((~actual & estimate).sum())
        fn = int((actual & ~estimate).sum())
        role_metrics[name] = dict(positives=int(actual.sum()), true_positive=tp,
                                  false_positive=fp, false_negative=fn,
                                  f1=2 * tp / max(1, 2 * tp + fp + fn))
    goals = example['goals']
    correct = torch.stack([scores[goals].argmax(-1) == example['attributes'][goals, column]
                           for column, scores in enumerate(attribute_logits)], dim=-1)
    attrs_majority = (example['attributes'][goals] == majority['attributes']).float().mean(0)
    seen = torch.tensor([p.numpy().tobytes() in majority['patches'] for p in example['patches']])
    goal_seen = seen[goals]
    overlap = dict(exact_train_patch_matches=int(seen.sum()),
                   goal_train_patch_matches=int(goal_seen.sum()),
                   unseen_goal_patches=int((~goal_seen).sum()),
                   unseen_goal_joint_accuracy=float(correct[~goal_seen].all(-1).float().mean())
                   if bool((~goal_seen).any()) else None)
    semantic_correct = (predicted == labels).all(-1)
    semantic_correct[goals] &= correct.all(-1)
    board_correct = [bool(semantic_correct[example['level_index'] == level].all())
                     for level in range(example['levels'])]
    return dict(levels=example['levels'], visible_cells=len(labels), visible_goals=int(goals.sum()),
                roles=role_metrics, role_macro_f1=sum(m['f1'] for m in role_metrics.values()) / 8,
                exact_role_accuracy=float((predicted == labels).all(-1).float().mean()),
                goal_attribute_accuracy=correct.float().mean(0).tolist(),
                goal_joint_accuracy=float(correct.all(-1).float().mean()),
                goal_detected_and_attributes_correct=float((predicted[goals, 1] & correct.all(-1)).float().mean()),
                template_overlap=overlap,
                exact_visible_board_accuracy=sum(board_correct) / len(board_correct),
                role_majority_exact_accuracy=float((labels == majority['roles']).all(-1).float().mean()),
                goal_attribute_majority_accuracy=attrs_majority.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ls20-visible-cell-labels.npz'))
    parser.add_argument('--proof', type=Path, default=Path('artifacts/world-visible-cell-label-feasibility.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    args = parser.parse_args()
    if args.out.exists() or args.checkpoint.exists():
        raise FileExistsError('refusing existing experiment outputs')
    if args.steps < 1:
        raise ValueError('steps must be positive')
    def timeout(*_):
        raise TimeoutError('ten-minute perception experiment deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(600)
    print('PID', os.getpid(), flush=True)
    torch.set_num_threads(1); torch.manual_seed(42)
    started = time.monotonic()
    paths = [args.data, args.proof, Path(__file__), Path('pebby/agent/cell_appearance.py')]
    hashes = {str(path): digest(path) for path in paths}
    examples, proof = load_examples(args.data, args.proof)
    train = examples['train']; generator = torch.Generator().manual_seed(43)
    groups = [torch.where(train['role_masks'] == mask)[0] for mask in train['role_masks'].unique(sorted=True)]
    majority = dict(roles=train['role_bits'].float().mean(0) >= .5,
                    patches={p.numpy().tobytes() for p in train['patches']},
                    attributes=torch.tensor([train['attributes'][train['goals'], j].bincount().argmax()
                                             for j in range(3)]))
    model = CellAppearance()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    curve = []
    for step in range(args.steps):
        # Balance observed role combinations using TRAIN rows only. The batch
        # unit here is a cell patch; the main world-policy level batch is unchanged.
        categories = torch.randint(len(groups), (1024,), generator=generator)
        indices = torch.empty(1024, dtype=torch.long)
        for category, pool in enumerate(groups):
            positions = torch.where(categories == category)[0]
            indices[positions] = pool[torch.randint(len(pool), (len(positions),), generator=generator)]
        labels = train['role_bits'][indices].float(); goals = train['goals'][indices]
        optimizer.zero_grad(set_to_none=True)
        role, *attributes = model.patch_logits(train['patches'][indices]).split((8, *ATTRIBUTE_SIZES), dim=-1)
        role_loss = F.binary_cross_entropy_with_logits(role, labels)
        attribute_loss = sum(F.cross_entropy(scores[goals], train['attributes'][indices][goals, j])
                             for j, scores in enumerate(attributes)) / 3 if goals.any() else role.sum() * 0
        loss = role_loss + attribute_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            row = dict(step=step + 1, role_loss=float(role_loss.detach()), attribute_loss=float(attribute_loss.detach()))
            curve.append(row); print(json.dumps(row), flush=True)
    model.eval()
    results = {split: evaluate(model, example, majority) for split, example in examples.items()}
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('experiment input changed')
    metadata = dict(format=FORMAT, parameters=model.parameter_count(), training_steps=args.steps,
                    source_hashes=hashes, input_contract='public 7x7 pixels only',
                    validation_used_for_training_or_selection=False, selected='fixed final update',
                    initial_state_only=True)
    torch.save(dict(**metadata, weights=model.state_dict()), args.checkpoint)
    report = dict(status='complete', pid=os.getpid(), **metadata, batch_size=1024,
                  batch_unit='cell patch, sampled with replacement from train-only role groups',
                  device='cpu', cpu_threads=1, checkpoint=str(args.checkpoint),
                  checkpoint_sha256=digest(args.checkpoint), results=results, training_curve=curve,
                  elapsed_seconds=time.monotonic() - started,
                  limitations=['Perception component only; no transition prediction, planning or game completion.',
                               'Initial states only; no overlap, solved-goal, animation or reset coverage.',
                               'Scores require all49 neighborhood pixels visible and outside HUD; no learned visibility model.',
                               'Distinct generated levels share sprite templates; this does not test unseen artwork.',
                               'The production world policy and its 1024-distinct-level batches are unchanged.'])
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    signal.alarm(0)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
