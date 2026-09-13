"""One matched continuation epoch of the bound fresh reference model.

Run each arm in a new process/output directory. No optimizer state is resumed.
The existing trainer verifies NPZ and extracted array hashes once; stat guards
then detect mutation without re-reading the large mmap cache after fitting.
"""
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import statistics
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'checkpoints/ls20-reference-base-v1.pt'
PARENT_SHA = '6db7f40d4008ff809e9b3a8b05d6a9a18801b343f02e52a9f585eece5e31cff9'
INITIAL_SHA = 'e7b9009fdd35fc5b8350b0fe4dd96ebfb73c6fa1d12e76a466501bd31ed276db'
DATA = ROOT / 'data/reference-world-base-v1'
STEPS = 318


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def local_time():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def memory_check():
    available = next(int(line.split()[1]) * 1024 for line in
                     Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < 6 * 2**30:
        raise RuntimeError('MemAvailable below required 6 GiB')
    return available


def validate_parent(parent):
    expected = dict(kind='random', seed=42, weights_sha256=INITIAL_SHA, optimizer_state='new')
    if parent.get('initialization') != expected:
        raise ValueError('parent must have verified fresh random seed42 initialization')
    if any(parent.get(k) for k in ('initialize_checkpoint', 'initialize_glyph_checkpoint',
                                   'glyph_source', 'cell_source', 'on_policy_source')):
        raise ValueError('old auxiliary weights/data forbidden')
    if any(parent.get(k) != v for k, v in dict(epochs=10, optimizer_steps=3180,
                                              batch_size=1024, samples=325802).items()):
        raise ValueError('parent training bounds differ')
    train, validation = parent['train_seeds'], parent['validation_seeds']
    if len(set(train)) != 10000 or len(set(validation)) != 500 or set(train) & set(validation):
        raise ValueError('parent requires 10000/500 disjoint levels')
    for split in ('train', 'validation'):
        if Path(parent[split]).resolve() != DATA / (split + '.npz'):
            raise ValueError('parent data source differs')
    if parent['config'].get('cell_recall'):
        raise ValueError('cell source forbidden')


def training_arguments(args, parent):
    from tools.run_reference_base_pipeline import START, END
    argv = ['--train', str(DATA / 'train.npz'), '--validation', str(DATA / 'validation.npz'),
            '--data-cache-dir', str(DATA / 'array-cache'), '--initialize-checkpoint', str(PARENT),
            '--checkpoint-out', str(args.out_dir / 'model.pt'), '--report-out', str(args.out_dir / 'training.json'),
            '--epochs', '1', '--batch-size', '1024', '--seed', str(args.seed), '--lr', str(args.lr),
            '--weight-decay', str(parent['weight_decay']), '--device', 'cuda', '--precision', 'bf16',
            '--compile-core', '--temporal-backend', 'math', '--checkpoint-encoder', '--encoder-chunk-size', '128',
            '--select-on', 'last', '--drop-last', '--curriculum', '--min-train-levels', '10000',
            '--require-verified-data', '--require-winning-coverage', '--require-exact-distances',
            '--curriculum-start', *map(str, START), '--curriculum-end', *map(str, END)]
    for key, value in parent['config'].items():
        if key == 'architecture':
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend((flag, str(value)))
    weights = dict(parent['loss_weights'], grounding=args.grounding_weight, sigreg=args.sigreg_weight)
    for key, value in weights.items():
        argv.extend(('--' + key.replace('_', '-') + '-weight', str(value)))
    return argv


class SourceGuard:
    def __init__(self, hashed, large):
        self.hashes = {str(p): digest(p) for p in hashed}
        self.stats = {str(p): self.stat(p) for p in large}

    @staticmethod
    def stat(path):
        st = Path(path).stat()
        return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

    def verify(self):
        for path, sha in self.hashes.items():
            if digest(path) != sha:
                raise ValueError(f'bound source changed: {path}')
        for path, stat in self.stats.items():
            if self.stat(path) != stat:
                raise ValueError(f'bound data changed: {path}')


def source_guard():
    hashed = [PARENT, Path(__file__).resolve(), ROOT / 'tools/run_reference_base_pipeline.py',
              DATA / 'manifest.json', DATA / 'build-report.json']
    hashed += sorted((ROOT / 'pebby/agent').glob('*.py'))
    large = []
    report = json.loads((DATA / 'build-report.json').read_text())
    if report.get('status') != 'complete' or report.get('sources_unchanged') is not True:
        raise ValueError('completed source-bound cache required')
    for split in ('train', 'validation'):
        receipt = DATA / split / 'merged.json'
        sha = json.loads(receipt.read_text())['sha256']
        if report['splits'][split]['sha256'] != sha:
            raise ValueError('cache receipt/report mismatch')
        manifests = list((DATA / 'array-cache').glob(sha + '-*/manifest.json'))
        if len(manifests) != 1:
            raise ValueError('exactly one existing source-hashed mmap cache required')
        manifest = json.loads(manifests[0].read_text())
        if manifest['source_sha256'] != sha:
            raise ValueError('mmap source hash mismatch')
        hashed += [receipt, manifests[0]]
        large += [DATA / (split + '.npz')]
        large += [manifests[0].parent / (name + '.npy') for name in manifest['arrays']]
    return SourceGuard(hashed, large)


class StepMonitor:
    def __init__(self, out_dir, total=STEPS):
        self.out_dir, self.total, self.steps = out_dir, total, 0
        self.recent = deque(maxlen=20)
        self.previous = time.monotonic()

    @contextmanager
    def watch(self, optimizer):
        original = optimizer.step
        @wraps(original)
        def step(*args, **kwargs):
            if self.steps >= self.total:
                raise RuntimeError('optimizer update budget exceeded')
            memory_check()
            result = original(*args, **kwargs)
            now = time.monotonic()
            self.steps += 1
            if self.steps > 3:
                self.recent.append(now - self.previous)
            self.previous = now
            if self.steps % 20 == 0 or self.steps == self.total:
                median = statistics.median(self.recent) if self.recent else None
                progress = dict(time=local_time(), optimizer_steps=self.steps, total_steps=self.total,
                                median_recent_step_seconds=median,
                                remaining_training_seconds=(self.total-self.steps)*median if median else None,
                                eta_scope='training updates only; excludes validation and evaluation')
                write(self.out_dir / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            return result
        with patch.object(optimizer, 'step', step):
            yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--grounding-weight', type=float, default=1.)
    parser.add_argument('--sigreg-weight', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    stat = Path(f'/proc/{os.getpid()}/stat').read_text()
    status = dict(status='starting', pid=os.getpid(), start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
                  started=local_time(), optimizer_steps=0)
    write(args.out_dir / 'status.json', status)
    monitor = StepMonitor(args.out_dir)
    try:
        memory_check()
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
        os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        import torch
        from pebby.agent import world_train
        if digest(PARENT) != PARENT_SHA:
            raise ValueError('parent checkpoint SHA256 differs')
        parent = torch.load(PARENT, map_location='cpu', weights_only=False)
        validate_parent(parent)
        arguments = training_arguments(args, parent)
        expected_weights = world_train.initial_state_sha256(SimpleNamespace(state_dict=lambda: parent['weights']))
        del parent
        guard = source_guard()
        write(args.out_dir / 'provenance.json', dict(parent_sha256=PARENT_SHA, argv=arguments,
              code_and_receipt_sha256=guard.hashes, large_data_stat_guard=guard.stats,
              expected_initial_named_weights_sha256=expected_weights,
              array_verification='existing trainer source-hashed cached loader', optimizer_state='new'))
        original = world_train.run_epoch
        def run_epoch(model, tensors, device, weights, batch_size, optimizer=None, *pos, **kw):
            if optimizer is None:
                return original(model, tensors, device, weights, batch_size, optimizer, *pos, **kw)
            if kw.get('steps_per_epoch') != STEPS or kw.get('total_steps') != STEPS:
                raise ValueError('expected exactly 318 matched updates')
            guard.verify()
            actual_weights = world_train.initial_state_sha256(model)
            if actual_weights != expected_weights:
                raise ValueError('initial named weights differ from parent checkpoint')
            write(args.out_dir / 'initial-weights.json', dict(sha256=actual_weights, parent_match=True))
            monitor.previous = time.monotonic()
            with monitor.watch(optimizer):
                result = original(model, tensors, device, weights, batch_size, optimizer, *pos, **kw)
            if monitor.steps != STEPS:
                raise ValueError('incomplete optimizer update budget')
            guard.verify()
            return result
        original_batches = world_train.curriculum_batches
        def curriculum_batches(tensors, sampler, *pos, **kw):
            for index, batch in enumerate(original_batches(tensors, sampler, *pos, **kw)):
                if index == 0:
                    seeds = list(sampler.last_level_seeds)
                    batch_hash = hashlib.sha256()
                    for name, value in sorted(batch.items()):
                        array = value.detach().cpu().numpy()
                        batch_hash.update(name.encode())
                        batch_hash.update(str((array.shape, array.dtype)).encode())
                        batch_hash.update(array.tobytes())
                    write(args.out_dir / 'first-batch.json', dict(level_count=len(seeds),
                          distinct_levels=len(set(seeds)),
                          batch_input_sha256=batch_hash.hexdigest(),
                          ordered_seed_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest()))
                yield batch
        from pebby.agent import world_cache
        original_digest = world_cache.digest
        expected_sources = {str(DATA / (split + '.npz')):
                            json.loads((DATA / split / 'merged.json').read_text())['sha256']
                            for split in ('train', 'validation')}
        def checked_digest(path):
            actual = original_digest(path)
            expected = expected_sources.get(str(Path(path).resolve()))
            if expected is not None and actual != expected:
                raise ValueError('NPZ differs from bound cache receipt')
            return actual
        with patch.object(world_train, 'run_epoch', run_epoch), \
             patch.object(world_train, 'curriculum_batches', curriculum_batches), \
             patch.object(world_cache, 'digest', checked_digest):
            world_train.main(arguments)
        guard.verify()
        status.update(status='complete', sources_unchanged=True)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(finished=local_time(), optimizer_steps=monitor.steps,
                      wrapper_hooks_restored=True, wrapper_child_processes_started=0)
        write(args.out_dir / 'status.json', status)


if __name__ == '__main__':
    main()
