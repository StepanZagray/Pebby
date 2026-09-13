"""Strict, fixed-budget joint continuation after verified position-recall warmup.

No training runs on import. Existing training/data/objective code stays unchanged;
all adapter hooks are scoped and every optimizer is newly constructed by main.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch

from tools import train_reference_onpolicy as onpolicy
from tools import train_reference_repair as repair

WARMUP_REPORT_FORMAT = 'pebby_position_recall_warmup_report_v1'
CONTROL = repair.ROOT / 'artifacts/reference-onpolicy-repair-v1/fit/first-batch.json'
CONTROL_RECEIPT = dict(level_count=1024, distinct_levels=1024,
    batch_input_sha256='72ac073f24c59eca6bea607466627db78336da68c69a0142cce981173ff1843f',
    ordered_seed_sha256='0682cad681da8ed0e0ac2b70857cd5b9ae7c99077dd937af96e84c182f3cc1c4')
EXPECTED_LOSSES = dict(prediction=1., sigreg=.1, policy=1., value=.5,
                       imagined_value=.5, player=.1, grounding=1., glyph=1., successor_policy=1.)
BINDING_FIELDS = ('source_checkpoint', 'source_checkpoint_sha256', 'feature_cache_binding',
                  'source_code_bindings', 'normalization_folded', 'base_initialization')


def validate_parent(parent):
    """Validate the protected fresh source, independently of repair.main."""
    from pebby.agent.world_model import WORLD_MODEL_FORMAT
    if parent.get('format') != WORLD_MODEL_FORMAT:
        raise ValueError('protected parent has unsupported format')
    expected = dict(kind='random', seed=42, weights_sha256=repair.INITIAL_SHA, optimizer_state='new')
    if parent.get('initialization') != expected:
        raise ValueError('protected parent must have fresh seed42 root initialization')
    if any(parent.get(key) for key in ('initialize_checkpoint', 'initialize_glyph_checkpoint',
                                      'glyph_source', 'cell_source', 'on_policy_source')):
        raise ValueError('protected parent contains forbidden auxiliary lineage')
    if parent.get('loss_weights') != EXPECTED_LOSSES:
        raise ValueError('protected parent original nine loss weights differ')
    if any(parent.get(key) != value for key, value in
           dict(epochs=10, optimizer_steps=3180, batch_size=1024, samples=325802).items()):
        raise ValueError('protected parent training bounds differ')
    train, validation = parent.get('train_seeds', []), parent.get('validation_seeds', [])
    if (len(train) != 10000 or len(set(train)) != 10000 or len(validation) != 500 or
            len(set(validation)) != 500 or set(train) & set(validation)):
        raise ValueError('protected parent needs 10000/500 disjoint seeds')
    for split in ('train', 'validation'):
        if Path(parent.get(split, '')).resolve() != repair.DATA / (split + '.npz'):
            raise ValueError('protected parent data source differs')
    config = parent['config']
    if (config.get('architecture') != 'world' or config.get('history') != 8 or
            config.get('cell_recall') or not config.get('grounding') or
            not config.get('glyph_recall') or not config.get('state_recall')):
        raise ValueError('protected parent config differs')


def validate_warmup_weights(warmup, parent):
    """Reject any changed frozen tensor, including buffers and player/glyph heads."""
    import torch
    from pebby.agent.world_position_recall import POSITION_INPUTS, POSITION_RECALL_VERSION
    if warmup['config'] != {**parent['config'], 'position_recall': POSITION_RECALL_VERSION}:
        raise ValueError('warmup config differs from protected parent plus position recall')
    before, after = parent['weights'], warmup['weights']
    if set(before) != set(after):
        raise ValueError('warmup named weight keys differ')
    for name, value in after.items():
        original = before[name]
        shape = original.shape
        if name == 'projector.0.weight':
            shape = (original.shape[0], original.shape[1] + POSITION_INPUTS)
        if value.shape != shape or value.dtype != original.dtype or not torch.isfinite(value).all():
            raise ValueError(f'warmup weight shape/dtype/finiteness differs: {name}')
        if not name.startswith(('projector.', 'grounding_head.')) and not torch.equal(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                original.detach().cpu().contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError(f'warmup changed non-head weight: {name}')


def validate_warmup(checkpoint_path, report_path):
    import torch
    from pebby.agent import world_position_recall as position
    checkpoint_path, report_path = Path(checkpoint_path).resolve(), Path(report_path).resolve()
    if repair.digest(repair.PARENT) != repair.PARENT_SHA:
        raise ValueError('protected parent SHA256 differs')
    parent = torch.load(repair.PARENT, map_location='cpu', weights_only=True)
    validate_parent(parent)
    model, warmup = position.load_checkpoint(checkpoint_path)
    del model
    report = json.loads(report_path.read_text())
    if (report.get('format') != WARMUP_REPORT_FORMAT or report.get('status') != 'complete' or
            report.get('only_projector_grounding_updated') is not True or
            report.get('normalization_folded') is not True or
            report.get('finite_gate_validation', {}).get('passed') is not True or
            report.get('normalizer_fold_validation', {}).get('passed') is not True):
        raise ValueError('completed, verified finite-gate warmup with folded normalizer required')
    if report.get('warmup_checkpoint_sha256') != repair.digest(checkpoint_path):
        raise ValueError('warmup checkpoint differs from completion report')
    provenance = warmup.get('warmup_provenance', {})
    if any(key not in report or provenance.get(key) != report[key] for key in BINDING_FIELDS):
        raise ValueError('warmup report/checkpoint provenance differs')
    if (Path(provenance['source_checkpoint']).resolve() != repair.PARENT or
            provenance['source_checkpoint_sha256'] != repair.PARENT_SHA or
            provenance['base_initialization'] != parent['initialization'] or
            Path(provenance.get('report_path', '')).resolve() != report_path or
            provenance.get('warmup_epochs') != 100 or provenance.get('batch_size') != 1024):
        raise ValueError('warmup protected parent/root/report/training binding differs')
    for key in ('train_seeds', 'validation_seeds'):
        if warmup.get(key) != parent[key] or report.get(key) != parent[key]:
            raise ValueError('warmup seeds must exactly equal protected disjoint parent seeds')
    binding = provenance['feature_cache_binding']
    manifest = Path(binding['path']).resolve()
    if manifest.is_dir():
        manifest = manifest / 'manifest.json'
    if repair.digest(manifest) != binding['manifest_sha256']:
        raise ValueError('warmup feature cache manifest differs')
    bindings = provenance['source_code_bindings']
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('warmup source code bindings required')
    for path, sha in bindings.items():
        if not Path(path).is_absolute() or repair.digest(path) != sha:
            raise ValueError(f'warmup source code binding differs: {path}')
    validate_warmup_weights(warmup, parent)
    return parent, warmup, report, manifest


def training_arguments(args, parent, supplement):
    options = SimpleNamespace(out_dir=args.out_dir, seed=42, lr=1e-4,
                              grounding_weight=1., sigreg_weight=.1)
    source = {**parent, 'config': {key: value for key, value in parent['config'].items()
                                 if key != 'position_recall'}}
    result = repair.training_arguments(options, source)
    result[result.index('--initialize-checkpoint') + 1] = str(args.warmup_checkpoint)
    return result + ['--on-policy-data', str(supplement), '--on-policy-fraction', '.25',
                     '--on-policy-auxiliary-fraction', '0']


def source_record(module):
    path = Path(module.__file__).resolve()
    return dict(module=module.__name__, path=str(path), sha256=repair.digest(path))


def experiment_provenance(args, warmup, report):
    from pebby.agent import world_position_recall, world_training_objectives
    return dict(experiment='public_position_recall_v1',
                parent_checkpoint=str(repair.PARENT), parent_sha256=repair.PARENT_SHA,
                base_initialization=warmup['warmup_provenance']['base_initialization'],
                warmup_checkpoint=str(args.warmup_checkpoint),
                warmup_checkpoint_sha256=report['warmup_checkpoint_sha256'],
                warmup_report=str(args.warmup_report), warmup_report_sha256=repair.digest(args.warmup_report),
                warmup_provenance=warmup['warmup_provenance'], optimizer_state='new',
                variant_source=source_record(world_position_recall),
                training_objective_source={**source_record(world_training_objectives),
                                           'variant': 'original', 'prediction_target_detached': False})


@contextmanager
def model_context(provenance, expected_weights):
    """Adapt only trainer-local model IO; restore even when fitting raises."""
    from pebby.agent import world_train, world_position_recall as position, world_training_objectives
    def initialize(target, source):
        if type(target) is not position.PositionRecallPolicy or type(source) is not position.PositionRecallPolicy:
            raise ValueError('joint initialization requires position checkpoint subtype')
        if target.config() != source.config() or world_train.initial_state_sha256(source) != expected_weights:
            raise ValueError('joint initialization differs from verified warmup')
        target.load_state_dict(source.state_dict(), strict=True)
        return []
    def save(path, model, **metadata):
        # world_train copies subtype metadata into .last; the subtype owns these reserved fields.
        metadata.pop('position_source', None)
        metadata.update(provenance)
        metadata['initialization'] = dict(kind='verified_position_warmup', optimizer_state='new',
                                           weights_sha256=expected_weights,
                                           parent_sha256=repair.PARENT_SHA,
                                           warmup_checkpoint_sha256=provenance['warmup_checkpoint_sha256'])
        return position.save_checkpoint(path, model, **metadata)
    with patch.object(world_train, 'WorldPolicy', position.PositionRecallPolicy), \
            patch.object(world_train, 'initialize_from_checkpoint', initialize), \
            patch.object(world_train, 'load_world_checkpoint', position.load_checkpoint), \
            patch.object(world_train, 'save_world_checkpoint', save), \
            patch.object(world_train, 'world_losses', world_training_objectives.world_losses), \
            patch.object(world_train, 'training_objective_source', lambda: provenance['training_objective_source']):
        yield


def batch_receipt(batch, sampler):
    seeds = list(sampler.last_level_seeds)
    batch_hash = hashlib.sha256()
    for name, value in sorted(batch.items()):
        array = value.detach().cpu().numpy()
        batch_hash.update(name.encode())
        batch_hash.update(str((array.shape, array.dtype)).encode())
        batch_hash.update(array.tobytes())
    return dict(level_count=len(seeds), distinct_levels=len(set(seeds)),
                batch_input_sha256=batch_hash.hexdigest(),
                ordered_seed_sha256=hashlib.sha256(json.dumps(seeds).encode()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup-checkpoint', required=True, type=Path)
    parser.add_argument('--warmup-report', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=repair.ROOT / 'data/reference-onpolicy-v1')
    parser.add_argument('--first-batch-control', type=Path, default=CONTROL)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    stat = Path(f'/proc/{os.getpid()}/stat').read_text()
    status = dict(status='starting', pid=os.getpid(), start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
                  started=repair.local_time(), optimizer_steps=0)
    repair.write(args.out_dir / 'status.json', status)
    monitor = repair.StepMonitor(args.out_dir)
    try:
        repair.memory_check()
        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
        os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        from pebby.agent import world_train, world_cache
        parent, warmup, report, manifest = validate_warmup(args.warmup_checkpoint, args.warmup_report)
        supplement, supplement_sha = onpolicy.validate_supplement(args.data_dir)
        guard = onpolicy.extended_guard(args.data_dir, supplement, supplement_sha, repair.source_guard)
        bound = [Path(__file__).resolve(), args.warmup_checkpoint, args.warmup_report,
                 args.first_batch_control, manifest]
        guard.hashes.update({str(path): repair.digest(path) for path in bound})
        guard.hashes.update(warmup['warmup_provenance']['source_code_bindings'])
        control = json.loads(args.first_batch_control.read_text())
        if any(control.get(key) != value for key, value in CONTROL_RECEIPT.items()):
            raise ValueError('matched first-batch control requires 1024 distinct levels')
        expected = world_train.initial_state_sha256(SimpleNamespace(state_dict=lambda: warmup['weights']))
        provenance = experiment_provenance(args, warmup, report)
        arguments = training_arguments(args, parent, supplement)
        del parent, warmup
        repair.write(args.out_dir / 'provenance.json', dict(**provenance, argv=arguments,
                     expected_initial_named_weights_sha256=expected,
                     code_and_receipt_sha256=guard.hashes, large_data_stat_guard=guard.stats,
                     first_batch_control=str(args.first_batch_control)))
        original_epoch, original_batches = world_train.run_epoch, world_train.curriculum_batches
        def run_epoch(model, tensors, device, weights, batch_size, optimizer=None, *pos, **kw):
            if optimizer is None:
                return original_epoch(model, tensors, device, weights, batch_size, optimizer, *pos, **kw)
            if (kw.get('steps_per_epoch') != repair.STEPS or kw.get('total_steps') != repair.STEPS or
                    batch_size != 1024 or weights != EXPECTED_LOSSES or monitor.steps != 0):
                raise ValueError('expected exactly one 318-update original-loss epoch at B1024')
            if optimizer.state:
                raise ValueError('joint optimizer must start with empty state')
            guard.verify()
            actual = world_train.initial_state_sha256(model)
            if actual != expected:
                raise ValueError('joint initial named weights differ from verified warmup')
            repair.write(args.out_dir / 'initial-weights.json', dict(sha256=actual, warmup_match=True))
            monitor.previous = time.monotonic()
            with monitor.watch(optimizer):
                result = original_epoch(model, tensors, device, weights, batch_size, optimizer, *pos, **kw)
            if monitor.steps != repair.STEPS:
                raise ValueError('incomplete joint update budget')
            guard.verify()
            return result
        def batches(tensors, sampler, *pos, **kw):
            for index, batch in enumerate(original_batches(tensors, sampler, *pos, **kw)):
                if index == 0:
                    receipt = batch_receipt(batch, sampler)
                    if any(receipt[key] != control.get(key) for key in receipt):
                        raise ValueError('joint first batch differs from matched on-policy control')
                    repair.write(args.out_dir / 'first-batch.json', {**receipt, 'control_match': True})
                yield batch
        original_digest = world_cache.digest
        expected_sources = {str(repair.DATA / (split + '.npz')):
                            json.loads((repair.DATA / split / 'merged.json').read_text())['sha256']
                            for split in ('train', 'validation')}
        expected_sources[str(supplement)] = supplement_sha
        def checked_digest(path):
            actual = original_digest(path)
            if str(Path(path).resolve()) in expected_sources and actual != expected_sources[str(Path(path).resolve())]:
                raise ValueError('NPZ differs from bound receipt')
            return actual
        with model_context(provenance, expected), patch.object(world_train, 'run_epoch', run_epoch), \
                patch.object(world_train, 'curriculum_batches', batches), \
                patch.object(world_cache, 'digest', checked_digest):
            world_train.main(arguments)
        if monitor.steps != repair.STEPS:
            raise ValueError('joint trainer did not complete 318 updates')
        guard.verify()
        training_path = args.out_dir / 'training.json'
        training = json.loads(training_path.read_text())
        training.update(provenance)
        training['initialization'] = dict(kind='verified_position_warmup', optimizer_state='new',
                                            weights_sha256=expected, parent_sha256=repair.PARENT_SHA,
                                            warmup_checkpoint_sha256=report['warmup_checkpoint_sha256'])
        training['optimizer_steps'] = monitor.steps
        repair.write(training_path, training)
        status.update(status='complete', sources_unchanged=True,
                      checkpoint_sha256=repair.digest(args.out_dir / 'model.pt'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(finished=repair.local_time(), optimizer_steps=monitor.steps,
                      wrapper_hooks_restored=True, wrapper_child_processes_started=0)
        repair.write(args.out_dir / 'status.json', status)


if __name__ == '__main__':
    main()
