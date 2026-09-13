"""CPU-only failure-path, tensor-lineage and scoped adapter contracts."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from pebby.agent import evaluate, world_train, world_position_recall as position
from pebby.agent.world_model import WorldModelConfig, WorldPolicy, WORLD_MODEL_FORMAT
from tools import evaluate_reference_position as evaluation
from tools import train_reference_position as runner
from tools import train_reference_repair as repair


@pytest.fixture
def tiny_pair():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    config = WorldModelConfig(channels=8, blocks=1, heads=2, expansion=2, loops=1, history=8,
        temporal_layers=1, hud_channels=4, hud_tokens=2, latent=8, reduce=1,
        predictor_blocks=1, predictor_hidden=16, value_hidden=8, max_distance=8,
        lookahead_depth=2, summary=2, readout_hidden=4, ranker_hidden=4,
        sigreg_projections=8, sigreg_knots=3, state_recall=True, glyph_recall=True,
        query_readout=True, grounding=True)
    base = WorldPolicy(config)
    custom = position.PositionRecallPolicy(config)
    position.initialize_from_base(custom, base)
    yield base, custom
    torch.set_num_threads(old)


def parent_metadata(base):
    return dict(format=WORLD_MODEL_FORMAT, config=base.config(), weights=base.state_dict(),
        initialization=dict(kind='random', seed=42, weights_sha256=repair.INITIAL_SHA, optimizer_state='new'),
        loss_weights=runner.EXPECTED_LOSSES.copy(), weight_decay=.0001,
        epochs=10, optimizer_steps=3180, batch_size=1024, samples=325802,
        train_seeds=list(range(10000)), validation_seeds=list(range(10000, 10500)),
        train=str(repair.DATA / 'train.npz'), validation=str(repair.DATA / 'validation.npz'))


def test_exact_arguments_and_no_custom_argparse_flags(tmp_path, tiny_pair):
    parent = parent_metadata(tiny_pair[0])
    parent['config']['position_recall'] = position.POSITION_RECALL_VERSION
    args = argparse.Namespace(out_dir=tmp_path, warmup_checkpoint=tmp_path / 'warmup.pt')
    argv = runner.training_arguments(args, parent, tmp_path / 'train.npz')
    assert '--position-recall' not in argv
    parsed = world_train.build_parser().parse_args(argv)
    assert parsed.initialize_checkpoint == args.warmup_checkpoint
    assert (parsed.epochs, parsed.batch_size, parsed.seed, parsed.lr) == (1, 1024, 42, 1e-4)
    assert parsed.on_policy_fraction == .25 and parsed.on_policy_auxiliary_fraction == 0
    assert parsed.successor_policy_weight == 1 and parsed.grounding_weight == 1 and parsed.sigreg_weight == .1
    assert parsed.drop_last and parsed.curriculum and parsed.min_train_levels == 10000
    assert parsed.require_verified_data and parsed.require_winning_coverage and parsed.require_exact_distances


def test_tensor_lineage_rejects_frozen_mutation_and_accepts_heads(tiny_pair):
    base, custom = tiny_pair
    parent = parent_metadata(base)
    checkpoint = dict(config=custom.config(), weights=copy.deepcopy(custom.state_dict()))
    runner.validate_warmup_weights(checkpoint, parent)
    for name in ('projector.0.weight', 'grounding_head.player.weight'):
        if name in checkpoint['weights']:
            checkpoint['weights'][name].add_(.125)
    runner.validate_warmup_weights(checkpoint, parent)
    name = next(name for name in checkpoint['weights'] if name.startswith('player_head.'))
    checkpoint['weights'][name].add_(1)
    with pytest.raises(ValueError, match='non-head weight'):
        runner.validate_warmup_weights(checkpoint, parent)


def test_tensor_lineage_rejects_config_shape_nan(tiny_pair):
    base, custom = tiny_pair
    parent = parent_metadata(base)
    checkpoint = dict(config=custom.config(), weights=copy.deepcopy(custom.state_dict()))
    checkpoint['config']['history'] = 1
    with pytest.raises(ValueError, match='config'):
        runner.validate_warmup_weights(checkpoint, parent)
    checkpoint['config'] = custom.config()
    checkpoint['weights']['projector.0.weight'] = base.state_dict()['projector.0.weight']
    with pytest.raises(ValueError, match='shape'):
        runner.validate_warmup_weights(checkpoint, parent)
    checkpoint['weights'] = copy.deepcopy(custom.state_dict())
    checkpoint['weights']['projector.0.weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finiteness'):
        runner.validate_warmup_weights(checkpoint, parent)


@pytest.fixture
def warmup_files(tmp_path, tiny_pair, monkeypatch):
    base, custom = tiny_pair
    parent = parent_metadata(base)
    parent_path = tmp_path / 'parent.pt'
    torch.save(parent, parent_path)
    monkeypatch.setattr(repair, 'PARENT', parent_path)
    monkeypatch.setattr(repair, 'PARENT_SHA', repair.digest(parent_path))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    source = tmp_path / 'source.py'
    source.write_text('pass\n')
    report_path, checkpoint_path = tmp_path / 'report.json', tmp_path / 'warmup.pt'
    provenance = dict(source_checkpoint=str(parent_path), source_checkpoint_sha256=repair.PARENT_SHA,
        feature_cache_binding=dict(path=str(manifest), manifest_sha256=repair.digest(manifest)),
        source_code_bindings={str(source): repair.digest(source)}, normalization_folded=True,
        base_initialization=parent['initialization'], report_path=str(report_path), warmup_epochs=100, batch_size=1024)
    checkpoint = position.save_checkpoint(checkpoint_path, custom, warmup_provenance=provenance,
                    train_seeds=parent['train_seeds'], validation_seeds=parent['validation_seeds'])
    report = dict(format=runner.WARMUP_REPORT_FORMAT, status='complete',
        only_projector_grounding_updated=True, finite_gate_validation=dict(passed=True),
        normalizer_fold_validation=dict(passed=True),
        warmup_checkpoint_sha256=repair.digest(checkpoint_path),
        train_seeds=parent['train_seeds'], validation_seeds=parent['validation_seeds'],
        **{key: provenance[key] for key in runner.BINDING_FIELDS})
    repair.write(report_path, report)
    return checkpoint_path, report_path, checkpoint, report


def test_full_warmup_validation_actual_checkpoint_and_mutation(warmup_files):
    path, report_path, checkpoint, report = warmup_files
    runner.validate_warmup(path, report_path)
    name = next(key for key in checkpoint['weights'] if key.startswith('player_head.'))
    checkpoint['weights'][name].add_(.5)
    torch.save(checkpoint, path)
    report['warmup_checkpoint_sha256'] = repair.digest(path)
    repair.write(report_path, report)
    with pytest.raises(ValueError, match='non-head weight'):
        runner.validate_warmup(path, report_path)


@pytest.mark.parametrize('mutation', ['gate', 'incomplete', 'overlap', 'parent', 'source', 'unknown_format'])
def test_reject_unverified_or_unbound_warmup(warmup_files, mutation):
    path, report_path, checkpoint, report = warmup_files
    if mutation == 'gate':
        report['finite_gate_validation']['passed'] = False
    elif mutation == 'incomplete':
        report['status'] = 'running'
    elif mutation == 'overlap':
        checkpoint['train_seeds'][0] = checkpoint['validation_seeds'][0]
        report['train_seeds'] = checkpoint['train_seeds']
    elif mutation == 'parent':
        checkpoint['warmup_provenance']['source_checkpoint_sha256'] = 'wrong'
        report['source_checkpoint_sha256'] = 'wrong'
    elif mutation == 'source':
        source = next(iter(checkpoint['warmup_provenance']['source_code_bindings']))
        Path(source).write_text('changed')
    else:
        checkpoint['format'] = 'unknown'
    torch.save(checkpoint, path)
    report['warmup_checkpoint_sha256'] = repair.digest(path)
    repair.write(report_path, report)
    with pytest.raises(ValueError):
        runner.validate_warmup(path, report_path)


def test_scoped_model_hooks_io_metadata_and_exception_restoration(tmp_path, tiny_pair):
    _, custom = tiny_pair
    expected = world_train.initial_state_sha256(custom)
    provenance = dict(warmup_checkpoint_sha256='test', optimizer_state='new',
                      training_objective_source=dict(variant='original'))
    names = ('WorldPolicy', 'initialize_from_checkpoint', 'load_world_checkpoint',
             'save_world_checkpoint', 'world_losses', 'training_objective_source')
    originals = {name: getattr(world_train, name) for name in names}
    with pytest.raises(RuntimeError, match='sentinel'):
        with runner.model_context(provenance, expected):
            target = world_train.WorldPolicy(custom.cfg)
            assert world_train.initialize_from_checkpoint(target, custom) == []
            assert world_train.initial_state_sha256(target) == expected
            checkpoint = world_train.save_world_checkpoint(tmp_path / 'joint.pt', target)
            _, loaded = world_train.load_world_checkpoint(tmp_path / 'joint.pt')
            assert loaded['format'] == position.POSITION_RECALL_FORMAT
            assert loaded['initialization']['optimizer_state'] == 'new'
            assert loaded['initialization']['weights_sha256'] == expected
            last_meta = {key: value for key, value in checkpoint.items()
                         if key not in ('format', 'config', 'parameters', 'weights')}
            world_train.save_world_checkpoint(tmp_path / 'last.pt', target, **last_meta)
            raise RuntimeError('sentinel')
    assert all(getattr(world_train, name) is value for name, value in originals.items())


def test_failed_runner_atomic_status_and_no_overwrite(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError('memory low')
    monkeypatch.setattr(repair, 'memory_check', fail)
    out = tmp_path / 'joint'
    argv = ['--out-dir', str(out), '--warmup-checkpoint', 'missing.pt', '--warmup-report', 'missing.json']
    with pytest.raises(RuntimeError, match='memory low'):
        runner.main(argv)
    status = json.loads((out / 'status.json').read_text())
    assert status['status'] == 'failed' and status['optimizer_steps'] == 0
    assert status['wrapper_hooks_restored'] and status['wrapper_child_processes_started'] == 0
    assert not list(out.glob('*.tmp'))
    with pytest.raises(FileExistsError):
        runner.main(argv)


def test_eval_scoped_loader_args_and_provenance(tmp_path, tiny_pair, monkeypatch):
    path, bank, report = tmp_path / 'model.pt', tmp_path / 'bank.jsonl', tmp_path / 'eval.json'
    position.save_checkpoint(path, tiny_pair[1])
    bank.write_text('{}\n')
    original_loader, original_argv = evaluate.load_checkpoint, sys.argv
    def fake_main():
        argv = sys.argv
        assert argv[argv.index('--on-stall') + 1] == 'repeat'
        assert argv[argv.index('--max-actions') + 1] == '300'
        assert argv[argv.index('--protocol') + 1] == 'strict'
        model, metadata = evaluate.load_checkpoint(path, 'cpu')
        assert type(model) is position.PositionRecallPolicy
        assert metadata['format'] == position.POSITION_RECALL_FORMAT
        repair.write(report, dict(protocol='strict', completed=0))
    monkeypatch.setattr(evaluate, 'main', fake_main)
    result = evaluation.main(['--checkpoint', str(path), '--bank', str(bank), '--device', 'cpu',
                              '--report-out', str(report)])
    assert evaluate.load_checkpoint is original_loader and sys.argv is original_argv
    adapter = result['position_evaluation_adapter']
    assert adapter['checkpoint_sha256'] == repair.digest(path)
    assert adapter['sources_unchanged'] and adapter['checkpoint']['position_source']['detach'] == 'appended_feature_copy_only'
    assert json.loads(report.read_text()) == result


def test_eval_exception_restores_loader_and_argv(tmp_path, monkeypatch):
    path, bank = tmp_path / 'bad.pt', tmp_path / 'bank.jsonl'
    torch.save(dict(format='not-position'), path)
    bank.write_text('{}\n')
    original_loader, original_argv = evaluate.load_checkpoint, sys.argv
    monkeypatch.setattr(evaluate, 'main', lambda: evaluate.load_checkpoint(path, 'cpu'))
    with pytest.raises(ValueError, match='unsupported checkpoint'):
        evaluation.main(['--checkpoint', str(path), '--bank', str(bank), '--device', 'cpu',
                         '--report-out', str(tmp_path / 'report.json')])
    assert evaluate.load_checkpoint is original_loader and sys.argv is original_argv
