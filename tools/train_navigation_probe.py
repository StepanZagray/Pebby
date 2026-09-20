"""Bounded, source-bound experiments for learning and transferring navigation.

Run with python -m tools.train_navigation_probe. The default three arms compare
encoder adaptation separately from a direct readout, with common public inputs,
fresh controller initialization, policy objective and identical sampled roots.
No checkpoint is promoted and confirmation examples never enter fitting.
"""
import argparse
from collections import defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('frozen-outcomes', 'finetune-outcomes', 'frozen-direct')
STAGES = ('adjacent', 'open', 'detour')
PUBLIC_KEYS = ('frames', 'history_valid', 'previous_actions')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    paths = [Path(__file__), ROOT / 'tools/evaluate_navigation_probe.py',
             ROOT / 'third_party/ls20/ls20.py']
    paths += sorted((ROOT / 'pebby/agent').glob('*.py'))
    paths += sorted((ROOT / 'pebby/ls20').glob('*.py'))
    return {str(path.resolve()): digest(path) for path in paths}


class Budget:
    def __init__(self, seconds, reserve_gib=6):
        self.started = time.monotonic()
        self.seconds = seconds
        self.reserve = reserve_gib * 2**30

    def __call__(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError('declared experiment wall-clock budget exhausted')
        available = next(int(line.split()[1]) * 1024 for line in
                         Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < self.reserve:
            raise MemoryError('six GiB host memory reserve breached')


def tensor_batch(arrays, indices, device):
    return {name: torch.as_tensor(np.asarray(value)[indices], device=device)
            for name, value in arrays.items()
            if np.asarray(value).dtype.kind in 'biuf'}


def scores(policy, items):
    return policy(*(items[name] for name in PUBLIC_KEYS))


def sample_indices(arrays, cases, count, rng, stages, replay_previous=0.):
    """Balance stages, then cases, then roots; path length confers no extra mass."""
    by_stage = {stage: [i for i, case in enumerate(cases) if case['stage'] == stage]
                for stage in stages}
    rows = np.asarray(arrays['case_index'])
    by_case = {i: np.flatnonzero(rows == i) for ids in by_stage.values() for i in ids}
    if any(not ids for ids in by_stage.values()) or any(not len(v) for v in by_case.values()):
        raise ValueError('every selected stage and case needs training roots')
    selected = []
    for _ in range(count):
        if replay_previous and len(stages) > 1:
            stage = rng.choice(stages[:-1]) if rng.random() < replay_previous else stages[-1]
        else:
            stage = rng.choice(stages)
        case = int(rng.choice(by_stage[stage]))
        selected.append(int(rng.choice(by_case[case])))
    return np.asarray(selected, dtype=np.int64)


def evaluate_roots(policy, arrays, cases, device, batch_size, guard):
    """Expose micro counts and case/group means without pretending roots are iid."""
    was_training = policy.training
    policy.eval()
    rows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(arrays['optimal']), batch_size):
                guard()
                stop = min(start + batch_size, len(arrays['optimal']))
                items = tensor_batch(arrays, slice(start, stop), device)
                logits = scores(policy, items).float()
                if logits.shape != (stop-start, 4) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('policy must return four finite scores per root')
                masks = items['optimal'].long()
                bits = (masks[:, None] & (1 << torch.arange(4, device=device))) != 0
                probabilities = (bits * logits.softmax(-1)).sum(-1)
                choices = logits.argmax(-1)
                correct = bits.gather(1, choices[:, None]).squeeze(-1)
                for offset in range(stop-start):
                    index = start + offset
                    rows.append(dict(case_index=int(arrays['case_index'][index]),
                                     correct=bool(correct[offset]), action_index=int(choices[offset]),
                                     valid=bool(bits[offset].any()),
                                     optimal_probability=float(probabilities[offset]),
                                     random_expected=float(bits[offset].float().mean())))
    finally:
        policy.train(was_training)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case_index']].append(row)
    per_case = []
    for index, values in sorted(grouped.items()):
        valid = [r for r in values if r['valid']]
        if not valid:
            continue
        case = cases[index]
        per_case.append(dict(id=case['id'], group_id=case['group_id'], stage=case['stage'],
                             roots=len(values), accuracy=sum(r['correct'] for r in valid)/len(valid),
                             first_correct=values[0]['correct'], first_action=values[0]['action_index'],
                             mean_optimal_probability=sum(r['optimal_probability'] for r in valid)/len(valid)))
    valid = [row for row in rows if row['valid']]
    groups = defaultdict(list)
    for case in per_case:
        groups[case['group_id']].append(case['accuracy'])
    per_stage = {}
    for stage in dict.fromkeys(c['stage'] for c in cases):
        current = [c for c in per_case if c['stage'] == stage]
        per_stage[stage] = dict(cases=len(current),
                               case_accuracy=np.mean([c['accuracy'] for c in current]).item() if current else None,
                               first_accuracy=np.mean([c['first_correct'] for c in current]).item() if current else None)
    case_by_id = {case['id']:case for case in cases}
    pairs = defaultdict(list)
    for row in per_case:
        if row['stage'] in ('adjacent','open'):
            pairs[case_by_id[row['id']]['pair_id']].append(row)
    paired = [values for values in pairs.values() if len(values) == 2]
    goal_pairs = dict(pairs=len(paired),both_correct=sum(all(r['first_correct'] for r in pair) for pair in paired),
                      action_changed=sum(pair[0]['first_action']!=pair[1]['first_action'] for pair in paired),
                      excluded_stages=['detour'],
                      reason='Opposite detour goals can share the same optimal perpendicular first actions.')
    return dict(roots=len(rows), valid_roots=len(valid), correct=sum(r['correct'] for r in valid),
                micro_accuracy=sum(r['correct'] for r in valid)/len(valid) if valid else None,
                random_expected=sum(r['random_expected'] for r in valid)/len(valid) if valid else None,
                case_accuracy=np.mean([c['accuracy'] for c in per_case]).item() if per_case else None,
                group_accuracy=np.mean([np.mean(v) for v in groups.values()]).item() if groups else None,
                per_stage=per_stage, per_case=per_case,goal_pairs=goal_pairs,
                limits='Related roots and goal pairs share a group; these are descriptive diagnostics, not independent trials.')


def select_rollout_cases(cases, per_stage):
    """Deterministic selection established before model scores are observed."""
    selected = []
    for stage in dict.fromkeys(case['stage'] for case in cases):
        current=[case for case in cases if case['stage']==stage]
        order = ('up','down','upper-left','lower-right','left','right','upper-right','lower-left')
        directions=[d for d in order if any(c['direction']==d for c in current)]
        by_direction={d:[c for c in current if c['direction']==d] for d in directions}
        for i in range(min(per_stage,len(current))):
            direction=directions[i % len(directions)]
            pool=by_direction[direction]
            # Cover directions first and rotate the start/glyph group too.
            selected.append(pool[(i//len(directions)+i % len(directions)) % len(pool)])
    return selected


def evaluate_rollouts(policy, cases, device, cap, guard):
    from pebby.agent.history import PolicyHistory
    from pebby.agent.navigation_diagnostics import optimal_actions, distance_to_goal
    from pebby.ls20.env import Ls20Scenario
    from pebby.ls20.generate import build_level
    from pebby.ls20.names import ACTION_IDS
    was_training = policy.training
    policy.eval()
    episodes = []
    try:
        with torch.inference_mode():
            for case in cases:
                guard()
                env = Ls20Scenario(build_level(case['spec']), 0)
                history = PolicyHistory(policy, device)
                history.observe(env.reset())
                trace, won, first_error = [], False, None
                for step in range(cap):
                    guard()
                    before = list(env.player_cell())
                    mask = optimal_actions(case['spec'], before)
                    logits = history.scores()
                    if tuple(logits.shape) != (4,) or not bool(torch.isfinite(logits).all()):
                        raise ValueError('policy returned invalid movement scores')
                    choice = int(logits.argmax())
                    if first_error is None and not mask & (1 << choice):
                        first_error = step
                    lives = env.lives()
                    observation = env.perform(ACTION_IDS[choice])
                    trace.append(dict(step=step, before=before, after=list(env.player_cell()),
                                      action_index=choice, optimal_mask=mask, logits=logits.tolist(),
                                      distance_before=distance_to_goal(case['spec'], before),
                                      lives_after=env.lives(), won=observation.won))
                    if observation.finished:
                        won = observation.won
                        break
                    if observation.frame is None:
                        raise RuntimeError('nonterminal action has no public observation')
                    history.observe(observation.frame, choice, reset=env.lives() < lives)
                episodes.append(dict(id=case['id'], group_id=case['group_id'], stage=case['stage'],
                                     won=won, actions=len(trace), optimal_length=case['optimal_length'],
                                     excess_actions=len(trace)-case['optimal_length'] if won else None,
                                     lives_left=env.lives(), first_error=first_error, trace=trace))
    finally:
        policy.train(was_training)
    per_stage = {}
    for stage in dict.fromkeys(c['stage'] for c in cases):
        current = [r for r in episodes if r['stage'] == stage]
        per_stage[stage] = dict(episodes=len(current), wins=sum(r['won'] for r in current),
                               win_rate=sum(r['won'] for r in current)/len(current))
    return dict(episodes=len(episodes), wins=sum(r['won'] for r in episodes),
                per_stage=per_stage, traces=episodes, action_cap=cap,
                protocol='fresh generated episodes, three native lives, strict movement argmax; no reset adapter')


def mastered(stage, roots, rollouts, accuracy, wins):
    a = roots['per_stage'].get(stage, {})
    b = rollouts['per_stage'].get(stage, {})
    return (a.get('cases', 0) > 0 and b.get('episodes', 0) > 0
            and a['first_accuracy'] >= accuracy and a['case_accuracy'] >= accuracy
            and b['win_rate'] >= wins)


def update_evidence(model, initial):
    result = {}
    for category in ('encoder', 'planner', 'direct_head'):
        names = [n for n, _ in model.named_parameters() if n.startswith(category + '.')]
        changed, squared, frozen_changed = [], 0., []
        parameters = dict(model.named_parameters())
        for name in names:
            parameter = parameters[name]
            delta = parameter.detach().cpu().float() - initial[name].float()
            if bool(delta.any()):
                changed.append(name)
                if not parameter.requires_grad:
                    frozen_changed.append(name)
            squared += float(delta.square().sum())
        result[category] = dict(changed_tensors=changed, update_l2=math.sqrt(squared),
                                frozen_changed=frozen_changed)
        if frozen_changed:
            raise RuntimeError('frozen parameters were modified')
    return result


def train_arm(args, name, encoder, planner, arrays, cases, guard, report, out):
    from pebby.agent.navigation_probe import NavigationProbe, stage_loss, make_checkpoint
    from pebby.agent.neural_outcome_policy import weights_sha256
    torch.manual_seed(args.seed)
    mode, readout = name.split('-')
    model = NavigationProbe(copy.deepcopy(encoder), copy.deepcopy(planner),
                            encoder_mode=mode, readout=readout).to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameter_groups(encoder_lr=args.encoder_lr,
                                  controller_lr=args.lr), weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    record = dict(status='running', arm=name, parameter_audit=model.parameter_audit(),
                  steps=[], evaluations=[], sampled_batches=[], recollections=[], stage_advances=[])
    record.update(initial_encoder_sha256=weights_sha256(model.encoder.state_dict()),
                  initial_planner_sha256=weights_sha256(model.planner.state_dict()))
    with torch.no_grad():
        first = tensor_batch(arrays['train'],slice(0,min(4,len(arrays['train']['optimal']))),args.device)
        record['initial_logits'] = scores(model,first).cpu().tolist()
    report['arms'][name] = record
    stages = list(args.stages)
    stage_index = 0
    recent = None

    def evaluation(step):
        training = evaluate_roots(model, arrays['train'], cases['train'], args.device, args.batch_size, guard)
        development = evaluate_roots(model, arrays['development'], cases['development'], args.device, args.batch_size, guard)
        rollouts = evaluate_rollouts(model, select_rollout_cases(cases['development'], args.rollouts_per_stage),
                                     args.device, args.rollout_cap, guard)
        record['evaluations'].append(dict(step=step, train=training, development=development, rollouts=rollouts))
        write_json(out/'report.json', report)
        print(json.dumps(dict(event='evaluation', arm=name, step=step, train=training['case_accuracy'],
                              development=development['case_accuracy'], wins=rollouts['wins'],
                              episodes=rollouts['episodes'])), flush=True)
        return development, rollouts

    evaluation(0)
    model.train()
    for step in range(1, args.steps+1):
        guard()
        current_stages = stages if args.schedule == 'mixed' else stages[:stage_index+1]
        indices = sample_indices(arrays['train'], cases['train'], args.batch_size, rng, current_stages,
                                 replay_previous=.25 if args.schedule == 'mastery' else 0.)
        items = tensor_batch(arrays['train'], indices, args.device)
        recent_indices = []
        if recent is not None:
            count = min(args.batch_size, max(1, round(args.batch_size*args.replay_fraction)))
            recent_indices = rng.choice(len(recent['optimal']), size=count).tolist()
            supplemental = tensor_batch(recent, recent_indices, args.device)
            for key in items.keys() & supplemental.keys():
                items[key][:count] = supplemental[key]
        record['sampled_batches'].append(dict(base=indices.tolist(), recent=recent_indices))
        optimizer.zero_grad(set_to_none=True)
        predicted = model.predict(*(items[k] for k in PUBLIC_KEYS))
        losses = stage_loss(predicted, items, objective=args.objective)
        if not bool(torch.isfinite(losses['total'])):
            raise FloatingPointError('nonfinite training loss')
        losses['total'].backward()
        gradients, missing = {}, []
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise RuntimeError('gradient reached a frozen parameter')
                continue
            if parameter.grad is None:
                missing.append(parameter_name)
            elif not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError('nonfinite gradient: '+parameter_name)
            else:
                category = parameter_name.split('.')[0]
                gradients[category] = gradients.get(category, 0.) + float(parameter.grad.square().sum())
        if missing:
            raise RuntimeError('trainable parameters have no gradient: '+', '.join(missing))
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_norm)
        optimizer.step()
        record['steps'].append(dict(step=step, loss=float(losses['total'].detach()),
                                   losses={k: float(v.detach()) for k,v in losses['losses'].items()},
                                   gradient_norm=float(norm), gradients={k: math.sqrt(v) for k,v in gradients.items()},
                                   stages=current_stages))
        if step % args.eval_every == 0 or step == args.steps:
            development, rollouts = evaluation(step)
            if (args.schedule == 'mastery' and stage_index+1 < len(stages)
                    and all(mastered(stage, development, rollouts,args.mastery_accuracy,args.mastery_wins)
                            for stage in stages[:stage_index+1])):
                stage_index += 1
                record['stage_advances'].append(dict(step=step, next_stage=stages[stage_index]))
        if args.recollect_every and step % args.recollect_every == 0 and step < args.steps:
            from pebby.agent.navigation_diagnostics import collect_examples
            selected = select_rollout_cases([c for c in cases['train'] if c['stage'] in current_stages],
                                           args.recollect_cases)
            model.eval()
            def choose_action(**public):
                guard()
                with torch.inference_mode():
                    inputs = [torch.as_tensor(public[k], device=args.device)[None] for k in PUBLIC_KEYS]
                    return int(model(*inputs)[0].argmax())
            recent = collect_examples(selected, history=encoder.cfg.history, choose_action=choose_action,
                                      max_actions=args.recollect_cap, guard=guard)
            record['recollections'].append(dict(step=step, cases=[c['id'] for c in selected], roots=len(recent['optimal'])))
            model.train()
    record['updates'] = update_evidence(model, initial)
    final=record['evaluations'][-1]
    record['final_mastery'] = {stage:mastered(stage,final['development'],final['rollouts'],
                                            args.mastery_accuracy,args.mastery_wins) for stage in stages}
    if mode == 'finetune' and not record['updates']['encoder']['changed_tensors']:
        raise RuntimeError('finetune arm did not update encoder parameters')
    if mode == 'frozen' and record['updates']['encoder']['changed_tensors']:
        raise RuntimeError('frozen arm changed encoder parameters')
    checkpoint = out/(name+'.pt')
    metadata = dict(arm=name, seed=args.seed, objective=args.objective, parent_sha256=report['parent_sha256'],
                    bank_sha256=report['bank_sha256'], official_training_inputs=False,
                    confirmation_used=False, steps=args.steps, source_sha256=report['source_sha256'])
    torch.save(make_checkpoint(model, metadata), checkpoint)
    record.update(status='complete', checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
    write_json(out/'report.json', report)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--objective', choices=('policy', 'outcomes'), default='policy')
    parser.add_argument('--controller-init', choices=('fresh', 'retained'), default='fresh')
    parser.add_argument('--width',type=int,help='fresh spatial controller width for a separate capacity sweep')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--schedule', choices=('mixed', 'mastery'), default='mixed')
    parser.add_argument('--groups-per-split', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-seed', type=int, default=20260913)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder-lr', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--clip-norm', type=float, default=1.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--rollouts-per-stage', type=int, default=4)
    parser.add_argument('--rollout-cap', type=int, default=40)
    parser.add_argument('--mastery-accuracy', type=float, default=.95)
    parser.add_argument('--mastery-wins', type=float, default=.9)
    parser.add_argument('--recollect-every', type=int, default=0)
    parser.add_argument('--recollect-cases', type=int, default=2)
    parser.add_argument('--recollect-cap', type=int, default=16)
    parser.add_argument('--replay-fraction', type=float, default=.25)
    args = parser.parse_args(argv)
    for name in ('groups_per_split','steps','batch_size','eval_every','threads','max_seconds',
                 'rollouts_per_stage','rollout_cap','recollect_cases','recollect_cap','lr','encoder_lr','clip_norm'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            parser.error(name+' must be positive and finite')
    for name in ('mastery_accuracy','mastery_wins','replay_fraction'):
        if not 0 < getattr(args,name) <= 1:
            parser.error(name+' must be in (0,1]')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or args.recollect_every < 0:
        parser.error('weight decay and recollection interval must be nonnegative')
    if len(set(args.arms)) != len(args.arms) or len(set(args.stages)) != len(args.stages):
        parser.error('arms and stages must be unique')
    if args.objective == 'outcomes' and 'frozen-direct' in args.arms:
        parser.error('direct readout comparison requires the common policy objective')
    if len(args.arms)>1 and (args.schedule == 'mastery' or args.recollect_every):
        parser.error('adaptive curriculum/recollection require one arm; they change model-dependent training exposure')
    if args.controller_init == 'retained' and 'frozen-direct' in args.arms:
        parser.error('readout comparison requires fresh controller initialization for both final heads')
    if args.width is not None and (args.width<=0 or args.controller_init != 'fresh'):
        parser.error('--width must be positive and requires fresh controller initialization')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.device == 'cpu':
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU experiment isolation requires a fresh process without initialized CUDA')
        # Recent torch AdamW checks accelerator graph capture even with CPU
        # parameters. Hide CUDA before its first device query in this process.
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if digest(args.parent) != args.parent_sha256:
        raise ValueError('parent checkpoint SHA256 differs')
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Budget(args.max_seconds)
    report = dict(format='pebby.navigation-experiment.v1', status='running', pid=os.getpid(),
                  started_local=datetime.now().astimezone().isoformat(),
                  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  parent_sha256=args.parent_sha256, source_sha256=source_hashes(), arms={},
                  confirmation_used=False, promoted=False,
                  limits=['Generated matching-goal navigation only; no seven-level success claim.',
                          'No persistent memory, learned RESET, multistep search or pretrained planner added.',
                          ('Policy-only run: physical/value/event losses are disabled; their accuracies are not learning targets.'
                           if args.objective == 'policy' else 'Outcome run: unweighted policy, physical, value and event losses; no event-positive admission gate.'),
                          'Direct readout changes final-head capacity and initialization; it is an architectural control.',
                          'Small group counts and short fits do not establish convergence or a capacity ceiling.'])
    write_json(args.out/'report.json', report)
    print(json.dumps(dict(event='started',pid=os.getpid(),out=str(args.out))), flush=True)
    try:
        guard()
        torch.set_num_threads(args.threads)
        from pebby.agent.navigation_diagnostics import make_cases, collect_examples, public_frame_overlap
        from pebby.agent.spatial_outcome_policy import load_checkpoint
        from pebby.agent.spatial_outcome_planner import SpatialOutcomePlanner
        cases = make_cases(seed=args.data_seed, groups_per_split=args.groups_per_split, stages=tuple(args.stages),guard=guard)
        bank = dict(format='pebby.navigation-bank.v1', data_seed=args.data_seed, splits=cases,
                    split_roles=dict(train='fit',development='diagnosis and mastery',confirmation='separate explicit evaluation only'),
                    initial_frame_overlap=public_frame_overlap(cases))
        if any(bank['initial_frame_overlap'].values()):
            raise ValueError('generated bank has cross-split initial-frame overlap')
        write_json(args.out/'bank.json',bank)
        report['bank_sha256'] = digest(args.out/'bank.json')
        parent,_ = load_checkpoint(args.parent,'cpu')
        arrays = {split:collect_examples(cases[split],history=parent.encoder.cfg.history,guard=guard)
                  for split in ('train','development')}
        report['roots'] = {s:len(a['optimal']) for s,a in arrays.items()}
        report['retained_baseline'] = dict(
            development=evaluate_roots(parent.to(args.device),arrays['development'],cases['development'],args.device,args.batch_size,guard),
            rollouts=evaluate_rollouts(parent,select_rollout_cases(cases['development'],args.rollouts_per_stage),
                                       args.device,args.rollout_cap,guard))
        parent.to('cpu')
        torch.manual_seed(args.seed)
        config=parent.planner.config()
        if args.width is not None:
            config['width']=args.width
        planner = (SpatialOutcomePlanner(config) if args.controller_init == 'fresh'
                   else copy.deepcopy(parent.planner))
        for arm in args.arms:
            train_arm(args,arm,parent.encoder,planner,arrays,cases,guard,report,args.out)
        records = list(report['arms'].values())
        if len(records)>1:
            for record in records[1:]:
                for key in ('initial_encoder_sha256','initial_planner_sha256','sampled_batches'):
                    if record[key] != records[0][key]:
                        raise RuntimeError('matched experiment differs in '+key)
            report['matched_initialization_and_batches'] = True
        if {'frozen-outcomes','finetune-outcomes'} <= report['arms'].keys():
            torch.testing.assert_close(torch.tensor(report['arms']['frozen-outcomes']['initial_logits']),
                                       torch.tensor(report['arms']['finetune-outcomes']['initial_logits']),
                                       atol=1e-7,rtol=1e-5)
            report['initial_encoder_comparison_logits_match'] = dict(atol=1e-7,rtol=1e-5)
        if digest(args.parent) != args.parent_sha256 or any(digest(p)!=h for p,h in report['source_sha256'].items()):
            raise RuntimeError('source or parent checkpoint changed during experiment')
        report.update(status='complete',source_unchanged=True,parent_unchanged=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(finished_local=datetime.now().astimezone().isoformat(),
                      elapsed_seconds=time.monotonic()-guard.started,cuda_initialized=torch.cuda.is_initialized())
        write_json(args.out/'report.json', report)
    return report


if __name__ == '__main__':
    main()
