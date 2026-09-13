import argparse
import json
from pathlib import Path

import pytest
import torch

from tools import train_reference_repair as repair
from pebby.agent import world_train


def parent():
    return torch.load(repair.PARENT, map_location='cpu', weights_only=False)


def test_exact_configuration_and_loss_propagation(tmp_path):
    source = parent()
    args = argparse.Namespace(out_dir=tmp_path, seed=19, lr=1e-4,
                              grounding_weight=2., sigreg_weight=.02)
    parsed = world_train.build_parser().parse_args(repair.training_arguments(args, source))
    for name, value in source['config'].items():
        if name != 'architecture':
            assert getattr(parsed, name) == value
    assert parsed.grounding_weight == 2.
    assert parsed.sigreg_weight == .02
    assert parsed.policy_weight == source['loss_weights']['policy']
    assert parsed.initialize_checkpoint == repair.PARENT
    assert parsed.epochs == 1 and parsed.batch_size == 1024 and parsed.seed == 19
    assert parsed.compile_core and parsed.checkpoint_encoder and not parsed.checkpoint_loops
    assert parsed.encoder_chunk_size == 128 and parsed.temporal_backend == 'math'
    assert parsed.require_verified_data and parsed.require_winning_coverage and parsed.require_exact_distances


def test_parent_freshness():
    source = parent()
    repair.validate_parent(source)
    source['initialization']['seed'] = 0
    with pytest.raises(ValueError, match='fresh'):
        repair.validate_parent(source)
    source = parent()
    source['initialize_checkpoint'] = 'old.pt'
    with pytest.raises(ValueError, match='old'):
        repair.validate_parent(source)


def test_guard_detects_source_and_data_changes(tmp_path):
    code, data = tmp_path / 'source.py', tmp_path / 'data.npy'
    code.write_text('first')
    data.write_bytes(b'first')
    guard = repair.SourceGuard([code], [data])
    guard.verify()
    code.write_text('second')
    with pytest.raises(ValueError, match='source changed'):
        guard.verify()
    guard = repair.SourceGuard([code], [data])
    data.write_bytes(b'second')
    with pytest.raises(ValueError, match='data changed'):
        guard.verify()


def test_monitor_preserves_optimizer_and_restores_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, 'memory_check', lambda: 8 * 2**30)
    plain = torch.nn.Parameter(torch.tensor([1.]))
    watched = torch.nn.Parameter(plain.detach().clone())
    first = torch.optim.AdamW([plain], lr=.01)
    second = torch.optim.AdamW([watched], lr=.01)
    monitor = repair.StepMonitor(tmp_path, total=4)
    original = second.step
    with monitor.watch(second):
        for _ in range(4):
            for parameter, optimizer in ((plain, first), (watched, second)):
                optimizer.zero_grad()
                parameter.square().sum().backward()
                optimizer.step()
        with pytest.raises(RuntimeError, match='budget'):
            second.step()
    assert second.step == original
    assert torch.equal(plain, watched)
    assert monitor.steps == 4
    assert json.loads((tmp_path / 'progress.json').read_text())['optimizer_steps'] == 4


def test_failed_status_and_refuse_overwrite(tmp_path, monkeypatch):
    out = tmp_path / 'arm'
    def fail():
        raise RuntimeError('low memory')
    monkeypatch.setattr(repair, 'memory_check', fail)
    with pytest.raises(RuntimeError, match='low memory'):
        repair.main(['--out-dir', str(out)])
    status = json.loads((out / 'status.json').read_text())
    assert status['status'] == 'failed' and status['optimizer_steps'] == 0
    assert status['wrapper_hooks_restored']
    with pytest.raises(FileExistsError):
        repair.main(['--out-dir', str(out)])
