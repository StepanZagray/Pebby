"""Bounded generated initial-state visibility fitting; no engine or policy inputs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from pebby.agent.cell_visibility import CellVisibility, FORMAT


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_data(path, proof_path):
    proof = json.loads(Path(proof_path).read_text())
    if (proof.get('status') != 'complete' or proof.get('initial_state_only') is not True
            or proof.get('output_npz', {}).get('sha256') != digest(path)):
        raise ValueError('dataset requires matching completed initial-state proof')
    for source, expected in proof.get('sources', {}).items():
        if digest(source) != expected:
            raise ValueError('audited source hash changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    n = len(data['seeds'])
    if (data['frames'].shape != (n, 64, 64) or data['frames'].dtype != np.uint8
            or data['frames'].max() > 15):
        raise ValueError('invalid public frame array')
    if set(data['split']) != {'train', 'validation'} or len(set(data['seeds'])) != n:
        raise ValueError('distinct train and validation seeds required')
    for split, seed in zip(data['split'], data['seeds']):
        if not (0 <= seed < 1_000_000 if split == 'train' else 1_000_000 <= seed < 2_000_000):
            raise ValueError('generated split seed boundary violated')
    for key in ('fully_visible', 'hud_overlap', 'label_mask', 'support7_label_mask'):
        if data[key].dtype != bool or data[key].shape != (n, 144):
            raise ValueError('invalid audited mask shape or dtype')
    if not np.array_equal(data['label_mask'], data['fully_visible'] & ~data['hud_overlap']):
        raise ValueError('invalid fully-visible base mask')
    if np.any(data['support7_label_mask'] & ~data['label_mask']):
        raise ValueError('support mask exceeds visible base mask')
    # Diagnostic partition only: these coordinates are NEVER model inputs or
    # postprocessing masks. Boundary has priority where exclusions overlap.
    rows, cols = np.divmod(np.arange(144), 12)
    boundary = (5 * rows - 1 < 0) | (3 + 5 * cols + 7 > 64)
    hud = (5 * rows - 1 + 7 > 52) & ~boundary
    labels = data['support7_label_mask']
    if np.any(labels[:, boundary | hud]):
        raise ValueError('support mask marks fixed excluded region visible')
    reasons = dict(boundary=np.broadcast_to(boundary, labels.shape).copy(),
                   hud=np.broadcast_to(hud, labels.shape).copy(),
                   fog=(~labels) & ~(boundary | hud)[None])
    return data, reasons, proof


def binary_metrics(predicted, labels, reasons):
    predicted = np.asarray(predicted, bool); labels = np.asarray(labels, bool)
    fp = predicted & ~labels; fn = ~predicted & labels
    return dict(cells=int(labels.size), positive_labels=int(labels.sum()),
                true_positive=int((predicted & labels).sum()), false_positive=int(fp.sum()),
                false_negative=int(fn.sum()), true_negative=int((~predicted & ~labels).sum()),
                accuracy=float((predicted == labels).mean()), exact_boards=int((predicted == labels).all(1).sum()),
                exact_board_fraction=float((predicted == labels).all(1).mean()),
                false_positive_by_reason={key: int((fp & mask).sum()) for key, mask in reasons.items()},
                negative_labels_by_reason={key: int(mask.sum()) for key, mask in reasons.items()})


@torch.no_grad()
def evaluate(model, frames, labels, reasons, fog_records):
    logits = torch.cat([model(chunk) for chunk in frames.split(128)])
    predicted = logits.ge(0).numpy()
    result = binary_metrics(predicted, labels.numpy(), reasons)
    result['bce'] = float(F.binary_cross_entropy_with_logits(logits, labels.float()))
    result['fog_boards'] = int(fog_records.sum())
    result['exact_fog_boards'] = int((predicted == labels.numpy()).all(1)[fog_records].sum())
    result['always_visible'] = binary_metrics(np.ones_like(predicted), labels.numpy(), reasons)
    return result


def save_progress(path, model, step, planned_steps, hashes, curve):
    """Atomically retain TRAIN-only progress without declaring final selection."""
    temporary = path.with_suffix('.tmp.pt')
    torch.save(dict(format=FORMAT, weights=model.state_dict(), completed_updates=step,
                    planned_updates=planned_steps, status='training_progress',
                    source_hashes=hashes, curve=curve,
                    validation_used_for_training_or_selection=False), temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ls20-visible-cell-labels-2k.npz'))
    parser.add_argument('--proof', type=Path, default=Path('artifacts/world-visible-cell-labels-2k-proof.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    if args.out.exists() or args.checkpoint.exists():
        raise FileExistsError('refusing existing outputs')
    if not 1 <= args.steps <= 500:
        raise ValueError('steps must be 1..500')
    def deadline(*_):
        raise TimeoutError('visibility experiment 600-second deadline')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(600)
    started = time.monotonic(); print('PID', os.getpid(), flush=True)
    torch.set_num_threads(1); torch.manual_seed(20260912)
    paths = [args.data, args.proof, Path(__file__), Path('pebby/agent/cell_visibility.py'),
             Path('pebby/agent/cell_appearance.py'), Path('tools/prepare_visible_cell_labels.py')]
    hashes = {str(p): digest(p) for p in paths}
    data, reasons, proof = load_data(args.data, args.proof)
    split = {}
    for name in ('train', 'validation'):
        selected = data['split'] == name
        split[name] = dict(frames=torch.from_numpy(data['frames'][selected]),
                           labels=torch.from_numpy(data['support7_label_mask'][selected]),
                           reasons={key: value[selected] for key, value in reasons.items()},
                           fog_records=~data['fully_visible'][selected].all(1))
    assert len(split['train']['frames']) == 2000 and len(split['validation']['frames']) == 500
    model = CellVisibility(); assert model.parameter_count() < 150_000
    initial = copy.deepcopy(model.state_dict()); measurements = []
    # Largest allowed batch first. Disposable TRAIN-only timing steps never
    # select weights; reset the exact initial state and optimizer afterward.
    for batch_size in (1024, 512, 256, 128, 64):
        trial = torch.optim.AdamW(model.parameters(), lr=.003)
        start = time.monotonic()
        for _ in range(2):
            trial.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(split['train']['frames'][:batch_size]),
                                                       split['train']['labels'][:batch_size].float())
            loss.backward(); trial.step()
        seconds = (time.monotonic() - start) / 2
        measurements.append(dict(batch_size=batch_size, seconds_per_update=seconds))
        model.load_state_dict(initial)
        if seconds * args.steps < 450:
            break
    else:
        raise RuntimeError('no practical full batch within CPU time budget')
    del trial
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    generator = torch.Generator().manual_seed(20260913)
    observed = torch.zeros(2000, dtype=torch.bool); curve = []
    progress_path = args.checkpoint.with_suffix('.progress.pt')
    if progress_path.exists():
        raise FileExistsError(progress_path)
    save_progress(progress_path, model, 0, args.steps, hashes, curve)
    print(json.dumps(dict(parameters=model.parameter_count(), batch_size=batch_size,
                          batch_measurements=measurements)), flush=True)
    for step in range(args.steps):
        # Each update samples distinct TRAIN level identities, never validation.
        indices = torch.randperm(2000, generator=generator)[:batch_size]
        observed[indices] = True
        optimizer.zero_grad(set_to_none=True)
        logits = model(split['train']['frames'][indices])
        loss = F.binary_cross_entropy_with_logits(logits, split['train']['labels'][indices].float())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            row = dict(step=step + 1, train_batch_bce=float(loss.detach()),
                       elapsed_seconds=time.monotonic() - started)
            curve.append(row); print(json.dumps(row), flush=True)
            save_progress(progress_path, model, step + 1, args.steps, hashes, curve)
    model.eval()
    results = {name: evaluate(model, **example) for name, example in split.items()}
    if any(digest(p) != sha for p, sha in hashes.items()):
        raise ValueError('experiment input/source changed')
    metadata = dict(format=FORMAT, parameters=model.parameter_count(), steps=args.steps,
                    batch_size=batch_size, seed=20260912, source_hashes=hashes,
                    input_contract='public full64x64 integer palette frame only',
                    initial_state_only=True, validation_used_for_training_or_selection=False,
                    selected='fixed final update; fixed logit0 threshold; no geometry postprocessing')
    torch.save(dict(**metadata, weights=model.state_dict()), args.checkpoint)
    report = dict(status='complete', **metadata, pid=os.getpid(), cpu_threads=1, device='cpu',
                  batch_unit='distinct generated TRAIN levels per update', training_levels_seen=int(observed.sum()),
                  train_levels=2000, validation_levels=500, split_seed_overlap=0,
                  batch_measurements=measurements, curve=curve, results=results,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                  progress_checkpoint=str(progress_path), progress_checkpoint_sha256=digest(progress_path),
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  limitations=['Only initial-state visibility; no dynamics, histories, animation, reset or solved-goal coverage.',
                               'Prediction is not a certified visibility mask; local ambiguity motivates global evidence but global accuracy is empirical.',
                               'Labels/player/fog/geometry are absent from forward input; exclusion reasons enter diagnostic metrics only.',
                               'No official data, policy changes, Oracle use, threshold tuning or validation selection.'])
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    signal.alarm(0); print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
