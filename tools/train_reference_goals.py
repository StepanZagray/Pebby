"""Gated, matched 318-update goal-preservation continuation; no training on import.

Deployment remains a standard WorldPolicy. The fresh auxiliary head lives outside
its state dict; every policy save has a separately hashed diagnostic head sidecar.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import torch

from pebby.agent import cell_appearance, world_goal_objective as objective
from pebby.agent import world_train, world_training_objectives
from tools import train_reference_onpolicy as onpolicy
from tools import train_reference_repair as repair

VALIDATION_BANK = repair.ROOT / 'data/ls20-reference-unequal-v1/validation.jsonl'
CONTROL = repair.ROOT / 'artifacts/reference-onpolicy-repair-v1/fit/first-batch.json'
CONTROL_BATCH = dict(batch_input_sha256='72ac073f24c59eca6bea607466627db78336da68c69a0142cce981173ff1843f',
                     ordered_seed_sha256='0682cad681da8ed0e0ac2b70857cd5b9ae7c99077dd937af96e84c182f3cc1c4')


def validate_gate(teacher_path, audit_path, approval_path):
    """No GPU allocation before explicit completed-dynamic-audit authorization."""
    teacher_path, audit_path, approval_path = map(lambda p: Path(p).resolve(),
                                                (teacher_path, audit_path, approval_path))
    approval = json.loads(approval_path.read_text())
    audit = json.loads(audit_path.read_text())
    teacher_sha = repair.digest(teacher_path)
    if (approval.get('format') != 'pebby.goal-teacher-approval.v1' or
            approval.get('approved') is not True or
            approval.get('scope') != 'goal_preservation_training' or
            approval.get('audit_sha256') != repair.digest(audit_path) or
            approval.get('teacher_sha256') != teacher_sha or
            approval.get('parent_sha256') != repair.PARENT_SHA):
        raise ValueError('explicit root-approved dynamic audit and teacher binding required')
    if audit.get('status') != 'complete' or approval.get('dynamic_audit_accepted') is not True:
        raise ValueError('completed accepted dynamic audit required; initial-only results forbidden')
    # Approval binds the exact audit schema/content; audit-specific metric acceptance
    # is the root reviewer's decision, never inferred from an initial-only score.
    if repair.digest(repair.PARENT) != repair.PARENT_SHA:
        raise ValueError('parent checkpoint binding differs')
    audit_bindings = {(repair.ROOT / name).resolve(): sha
                      for name, sha in audit.get('source_sha256', {}).items()}
    if (audit_bindings.get(teacher_path) != teacher_sha or
            'dynamic' not in audit.get('scope', '').lower() or
            any(audit.get(key) is not True for key in
                ('model_weights_unchanged', 'source_files_unchanged', 'workers_reaped'))):
        raise ValueError('dynamic audit must bind this frozen teacher and completed cleanup')
    for path, expected in audit_bindings.items():
        if repair.digest(path) != expected:
            raise ValueError(f'dynamic audit source binding differs: {path}')
    audit_code = audit_path.parent / 'audit.py'
    if repair.digest(audit_code) != audit.get('code_sha256'):
        raise ValueError('dynamic audit code binding differs')
    checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != cell_appearance.FORMAT or checkpoint.get('seed') != 42 or
            checkpoint.get('parameters') != 110166 or not checkpoint.get('source_bindings')):
        raise ValueError('fresh generated pixel candidate schema required')
    bound = {str(teacher_path): teacher_sha, str(audit_path): repair.digest(audit_path),
             str(approval_path): repair.digest(approval_path),
             str(audit_code): repair.digest(audit_code),
             **{str(path): sha for path, sha in audit_bindings.items()}}
    parity_binding = approval.get('runtime_selection_parity', {})
    parity_path = Path(parity_binding.get('path', '')).resolve()
    if not parity_path.is_file() or repair.digest(parity_path) != parity_binding.get('sha256'):
        raise ValueError('approved runtime selection parity receipt required')
    parity = json.loads(parity_path.read_text())
    if (parity.get('status') != 'complete' or parity.get('source_files_unchanged') is not True or
            parity.get('states') != 8004 or parity.get('runtime_selected') != 6701 or
            parity.get('annotations_current_then_actual_order_exact') is not True or
            any(parity.get(key) != 0 for key in ('geometry_support_different_cells',
                'selection_disagreement_cells', 'selected_non_goal', 'selected_incorrect_joint_triples'))):
        raise ValueError('completed exact runtime goal selection parity required')
    parity_sources = {(repair.ROOT / name).resolve(): sha
                      for name, sha in parity.get('source_sha256', {}).items()}
    for path in (teacher_path, audit_path, Path(objective.__file__).resolve()):
        if parity_sources.get(path) != repair.digest(path):
            raise ValueError('runtime parity must bind the current objective, teacher and audit')
    for path, expected in parity_sources.items():
        if repair.digest(path) != expected:
            raise ValueError(f'runtime parity source binding differs: {path}')
        bound[str(path)] = expected
    bound[str(parity_path)] = repair.digest(parity_path)
    for name, record in checkpoint['source_bindings'].items():
        path = (repair.ROOT / name).resolve()
        if repair.digest(path) != record['sha256']:
            raise ValueError(f'teacher fresh-data/source binding differs: {path}')
        bound[str(path)] = record['sha256']
    initial_report = teacher_path.parent / 'report.json'
    initial = json.loads(initial_report.read_text())
    if (initial.get('status') != 'complete' or initial.get('old_weights_loaded') is not False or
            initial.get('candidate_checkpoint_sha256') != teacher_sha):
        raise ValueError('completed fresh candidate publication required')
    bound[str(initial_report)] = repair.digest(initial_report)
    control = json.loads(CONTROL.read_text())
    if any(control.get(key) != value for key, value in CONTROL_BATCH.items()):
        raise ValueError('protected first-batch control binding differs')
    bound[str(CONTROL)] = repair.digest(CONTROL)
    fog = {}
    for bank, split, count in ((onpolicy.BANK, 'train', 10000), (VALIDATION_BANK, 'validation', 500)):
        if approval.get('bank_sha256', {}).get(split) != repair.digest(bank):
            raise ValueError('approval must bind current generated TRAIN/validation banks')
        specs = [json.loads(line) for line in bank.read_text().splitlines()]
        if len(specs) != count:
            raise ValueError('fresh bank level count differs')
        for spec in specs:
            if (spec['seed'] in fog or spec.get('source') != 'generated_only' or
                    type(spec.get('fog')) is not bool):
                raise ValueError('invalid, overlapping, or non-generated fog bank')
            fog[spec['seed']] = spec['fog']
        bound[str(bank)] = repair.digest(bank)
        seed_files = [Path(name) for name in bound if name.endswith('/' + split + '-seeds.npy')]
        if len(seed_files) != 1:
            raise ValueError('teacher must bind one fresh per-cell seed file for each split')
        import numpy as np
        actual_seeds = set(map(int, np.unique(np.load(seed_files[0], allow_pickle=False))))
        if actual_seeds != {spec['seed'] for spec in specs}:
            raise ValueError('teacher training/validation seeds differ from current bound banks')
    # Constructing the frozen teacher must leave base initialization RNG untouched.
    with torch.random.fork_rng(devices=[]):
        teacher = cell_appearance.CellAppearance()
    teacher.load_state_dict(checkpoint['weights'], strict=True)
    return teacher.eval().requires_grad_(False), fog, bound


def source_record(module):
    path = Path(module.__file__).resolve()
    return {'path': str(path), 'sha256': repair.digest(path)}


@contextmanager
def objective_context(teacher, fog_by_seed, bindings):
    auxiliary = objective.GoalObjective(teacher, fog_by_seed)
    tracked_tensors = {}
    original_groups, original_epoch = world_train.parameter_groups, world_train.run_epoch
    original_tensors, original_save = world_train.as_tensors, world_train.save_world_checkpoint
    original_guard, original_write = repair.source_guard, repair.write
    policy = None

    def source():
        return dict(variant='actual_refined_goal_preservation', lambda_goal=1.,
                    target='teacher_soft_cross_entropy_mean_three_attributes',
                    normalization='none', head_seed=42, head_parameters=10126,
                    deployed_parameters=1198165, teacher_threshold=.99,
                    goal_threshold=.99, teacher_in_policy_forward=False,
                    imagined_semantic_loss=False, bindings=bindings,
                    objective=source_record(objective), base=source_record(world_training_objectives),
                    teacher_source=source_record(cell_appearance), wrapper=source_record(__import__(__name__, fromlist=[''])))

    def groups(model, weight_decay):
        nonlocal policy
        if policy is not None:
            raise RuntimeError('only one policy/optimizer initialization allowed')
        if model.parameter_count() != 1198165 or model.cfg.channels != 64:
            raise ValueError('deployed architecture must match protected control')
        policy = model
        device = next(model.parameters()).device
        auxiliary.head.to(device)
        auxiliary.teacher.to(device)
        return original_groups(model, weight_decay) + original_groups(auxiliary.head, weight_decay)

    def tensors(data):
        result = original_tensors(data)
        tracked_tensors[id(result)] = torch.as_tensor(data['seeds'].copy(), dtype=torch.long)
        return result

    def epoch(model, tensors_, device, weights, batch_size, optimizer=None, *args, **kwargs):
        if model is not policy:
            raise ValueError('objective requires the bound initialized policy')
        auxiliary.head.train(optimizer is not None)
        original_batches = world_train.curriculum_batches
        original_clip = torch.nn.utils.clip_grad_norm_

        def annotated_batches(tensors, sampler, *pos, **kw):
            # The repair recorder yields first, hashing the exact unaugmented batch.
            for batch in original_batches(tensors, sampler, *pos, **kw):
                yield {**batch, 'goal_seeds': torch.tensor(sampler.last_level_seeds, dtype=torch.long)}

        def clip(parameters, *pos, **kw):
            params = list(parameters)
            if {id(p) for p in params} != {id(p) for p in model.parameters()}:
                raise ValueError('unexpected clipping parameter set')
            return original_clip([*params, *auxiliary.head.parameters()], *pos, **kw)

        if optimizer is not None:
            actual = [p for group in optimizer.param_groups for p in group['params']]
            expected = [p for p in (*model.parameters(), *auxiliary.head.parameters()) if p.requires_grad]
            if len(actual) != len(expected) or {id(p) for p in actual} != {id(p) for p in expected}:
                raise ValueError('optimizer must contain all and only policy/head trainable parameters')
            if optimizer.state:
                raise ValueError('goal repair requires an empty new optimizer state')
        else:
            tensors_ = {**tensors_, 'goal_seeds': tracked_tensors[id(tensors_)]}
        with patch.object(world_train, 'curriculum_batches', annotated_batches), \
                patch.object(torch.nn.utils, 'clip_grad_norm_', clip):
            return original_epoch(model, tensors_, device, weights, batch_size, optimizer, *args, **kwargs)

    def save(path, model, **metadata):
        sidecar = Path(path).with_suffix('.goal-head.pt')
        temporary = sidecar.with_suffix(sidecar.suffix + '.tmp')
        torch.save(dict(format='pebby.goal-head.v1', weights={k: v.detach().cpu() for k, v in
                        auxiliary.head.state_dict().items()}, training_objective_source=source()), temporary)
        temporary.replace(sidecar)
        metadata['goal_head_sidecar'] = dict(path=str(sidecar.resolve()), sha256=repair.digest(sidecar),
                                           parameters=sum(p.numel() for p in auxiliary.head.parameters()))
        return original_save(path, model, **metadata)

    def guard():
        result = original_guard()
        for path, expected in bindings.items():
            if repair.digest(path) != expected:
                raise ValueError(f'goal source binding changed: {path}')
        result.hashes.update(bindings)
        for module in (objective, world_training_objectives, cell_appearance):
            record = source_record(module)
            result.hashes[record['path']] = record['sha256']
        result.hashes[str(Path(__file__).resolve())] = repair.digest(__file__)
        return result

    def write(path, value):
        if path.name == 'first-batch.json':
            if (value.get('level_count') != 1024 or value.get('distinct_levels') != 1024 or
                    any(value.get(key) != expected for key, expected in CONTROL_BATCH.items())):
                raise ValueError('goal first batch differs from protected on-policy control')
            value = {**value, 'control_match': True}
        if path.name == 'provenance.json':
            value = {**value, 'training_objective_source': source()}
        return original_write(path, value)

    with patch.object(world_train, 'world_losses', auxiliary), \
            patch.object(world_train, 'training_objective_source', source), \
            patch.object(world_train, 'parameter_groups', groups), \
            patch.object(world_train, 'as_tensors', tensors), \
            patch.object(world_train, 'run_epoch', epoch), \
            patch.object(world_train, 'save_world_checkpoint', save), \
            patch.object(repair, 'source_guard', guard), patch.object(repair, 'write', write):
        yield auxiliary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'out-dir', 'teacher', 'dynamic-audit', 'approval'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    teacher, fog, bindings = validate_gate(args.teacher, args.dynamic_audit, args.approval)
    with objective_context(teacher, fog, bindings):
        return onpolicy.main(['--data-dir', str(args.data_dir), '--out-dir', str(args.out_dir)])


if __name__ == '__main__':
    main()
