"""Procedural seven-tier generator calibrated to aggregate LS20 reference facts.

Geometry, positions, targets and routes are newly sampled. Complete contextual
search and real-engine replay certify each accepted row. Rejection statistics
are retained; higher tiers never silently fall back to an easier profile.
"""
from collections import Counter
import random

from . import names
from .generate import FORMAT, GENERATOR_VERSION, RESERVED, _connected, build_level
from .generation_quality import geometry_d4_partition, route_budget_slack
from .reference_profiles import DIFFICULTY_VERSION, DIFFICULTIES, PROFILES, KINDS, profile_errors, structural_metrics


def draft(rng, difficulty):
    p=PROFILES[difficulty]
    side=8 if difficulty==1 else 10
    left,top=rng.randint(1,11-side),rng.randint(1,11-side)
    free={(left+x,top+y) for x in range(side) for y in range(side)}-RESERVED
    target=rng.randint(*p['free_cells'])
    # Remove cells without disconnecting the board; varying the removal order
    # yields irregular rooms, bottlenecks and intermediate corridor densities.
    candidates=sorted(free)
    rng.shuffle(candidates)
    for cell in candidates:
        if len(free)<=target: break
        remaining=free-{cell}
        if len(_connected(remaining,min(remaining)))==len(remaining):free=remaining
    if len(free)!=target:return None
    rails,cyclers,walked=[],[],set()
    for kind,shape,length in p['rails']:
        options=[]
        for x,y in sorted(free-walked):
            if shape=='ring':
                cells={(x+dx,y+dy) for dx in range(3) for dy in range(3) if dx in (0,2) or dy in (0,2)}
                if cells<=free and not cells&walked:options.append(cells)
            else:
                for dx,dy in ((1,0),(0,1)):
                    cells={(x+dx*i,y+dy*i) for i in range(length)}
                    if cells<=free and not cells&walked:options.append(cells)
        if not options:return None
        cells=rng.choice(options);walked|=cells
        rails.append({'cells':sorted(cells)})
        cyclers.append({'cell':rng.choice(sorted(cells)),'kind':kind})
    spots=sorted(free-walked);rng.shuffle(spots)
    initial=[rng.randrange(size) for size in (6,4,4)]
    start=spots.pop()
    for kind in p['changed_kinds']:
        if kind not in {c['kind'] for c in cyclers}:cyclers.append({'cell':spots.pop(),'kind':kind})
    goals=[]
    for _ in range(p['goals']):
        for retry in range(20):
            triple=initial.copy()
            for kind in p['changed_kinds']:
                index=KINDS.index(kind);size=(6,4,4)[index]
                triple[index]=(initial[index]+rng.randrange(1,size))%size
            if triple not in [g['triple'] for g in goals]:break
        else:return None
        goals.append({'cell':spots.pop(),'triple':triple})
    refills=[spots.pop() for _ in range(p['refills'])]
    launchers=[]
    for _ in range(p['launchers']):
        options=[]
        for cell in spots:
            for dx,dy in names.ACTION_DELTAS:
                if (cell[0]-dx,cell[1]-dy) in free:continue
                probe=cell;distance=0
                while True:
                    probe=(probe[0]+dx,probe[1]+dy)
                    if probe not in free or probe in {g['cell'] for g in goals}:break
                    distance+=1
                if distance>=2:options.append((cell,(dx,dy)))
        if not options:return None
        cell,delta=rng.choice(options);spots.remove(cell)
        launchers.append({'cell':cell,'delta':delta})
    spec=dict(format=FORMAT,generator_version=GENERATOR_VERSION,difficulty=difficulty,
              difficulty_version=DIFFICULTY_VERSION,reference_level=difficulty,
              reference_optimal_actions=p['reference_actions'],quality_version=3,
              quality_profile='learning' if difficulty==1 else 'challenge',
              curriculum_version=DIFFICULTY_VERSION,size=64,
              walls=sorted({(x,y) for x in range(12) for y in range(12)}-free),
              start=start,start_triple=initial,goals=goals,cyclers=cyclers,rails=rails,
              launchers=launchers,refills=sorted(refills),step_counter=42,step_cost=p['cost'],
              fog=p['fog'],topology='connected_irregular',rail_mode='reference' if rails else 'none',
              reference_calibration='aggregate statistics; no official geometry or routes copied')
    spec.update(structural_metrics(spec))
    spec['changed_kinds']=list(spec['changed_kinds'])
    return None if profile_errors(spec,require_proof=False) else spec


def patroller_contacts(layout, before, after, action, outcome):
    """Cyclers exercised by entry and landing, at their actual engine ticks."""
    dx,dy=names.ACTION_DELTAS[action]
    target=(before[0][0]+dx,before[0][1]+dy)
    visits=[(target,layout.next_tick(before[7]))] if layout.free(target) else []
    if outcome=='launched':visits.append((after[0],after[7]))
    contacts=set()
    for cell,tick in visits:
        for number,patroller in enumerate(layout.patrollers):
            own=tick if tick<patroller['tail'] else patroller['tail']+(tick-patroller['tail'])%patroller['period']
            if patroller['cells'][own]==cell:contacts.add(number)
    return contacts


def verify(spec, search_limit=None, min_slack=None):
    from .env import Ls20Scenario
    from .layout import extract
    from .plan import Oracle, simulate
    from .extended_curriculum import ContractMismatch, gameplay_hash
    d=spec['difficulty'];p=PROFILES[d];context=d-1
    limit=p['search_limit'] if search_limit is None else min(search_limit,p['search_limit'])
    policy_floor=8 if d==1 else 0
    floor=max(policy_floor,0 if min_slack is None else min_slack)
    if profile_errors(spec,require_proof=False):return None,'profile_structure'
    env=Ls20Scenario(build_level(spec),context)
    try:layout=extract(env)
    except ValueError:return None,'inexact_layout'
    oracle=Oracle(layout,limit=limit,engine='fast')
    if oracle.truncated:return None,'search_truncated'
    if not oracle.solvable:return None,'unsolvable'
    if not p['actions'][0]<=oracle.optimal_actions<=p['actions'][1]:return None,'reference_action_length'
    solution=oracle.solution(seed=spec['seed'])
    state=oracle.start;minimum_slack=state[6]//layout.step_cost
    used=Counter();moving_contacts=set();launcher_contacts=set()
    for action in solution:
        before=state;index=names.ACTION_IDS.index(action)
        state,outcome=simulate(layout,state,index,oracle.refills)
        minimum_slack=min(minimum_slack,route_budget_slack(layout,before,state,action=index,outcome=outcome))
        dx,dy=names.ACTION_DELTAS[index];target=(before[0][0]+dx,before[0][1]+dy)
        moving_contacts.update(patroller_contacts(layout,before,state,index,outcome))
        if outcome=='launched':
            entry=target if layout.free(target) else before[0]
            for number,pad in enumerate(layout.launchers):
                if entry in pad['triggers'] and pad['distance']>0:launcher_contacts.add(number);break
        used[outcome]+=1
        used['refills_consumed']+=(state[5]^before[5]).bit_count()
        used['goals_cleared']+=(state[4]^before[4]).bit_count()
        result=env.perform(action)
        if outcome!='won' and (result.finished or env.lives()!=3 or oracle.state_of(env)!=state):
            raise ContractMismatch(f'reference seed={spec["seed"]} tier={d}: engine/planner disagreement')
    if not result.won or env.lives()!=3 or env.levels_completed!=1:
        raise ContractMismatch('reference generated winning route failed real engine replay')
    if minimum_slack<floor:return None,'route_budget_floor'
    used.update(used_patroller_count=len(moving_contacts),used_launcher_count=len(launcher_contacts))
    used=dict(used)
    used.update(won=True,moving_cycler=bool(moving_contacts),launcher=bool(launcher_contacts),refill=bool(used['refills_consumed']))
    fingerprint,partition=geometry_d4_partition(spec)
    proof=dict(seed=spec['seed'],difficulty=d,difficulty_version=DIFFICULTY_VERSION,
               context_index=context,context_engine_verified=True,search_truncated=False,
               optimal_actions=oracle.optimal_actions,context_optimal_actions=oracle.optimal_actions,
               engine_win=True,replay_lives=env.lives(),levels_completed=env.levels_completed,
               search_limit=limit,reachable_states=oracle._reachable,oracle_backend=oracle.engine)
    row={**spec,'solution':solution,'optimal_actions':oracle.optimal_actions,'context_solution':solution,
         'context_optimal_actions':oracle.optimal_actions,'context_index':context,'training_context_index':context,
         'verification_level_index':context,'verification_match_hint':layout.match_hint,
         'context_engine_verified':True,'engine_verified':True,'engine_win':True,'verification_lives':3,
         'replay_lives':3,'levels_completed':1,'search_truncated':False,'search_limit':limit,
         'reachable_states':oracle._reachable,'oracle_backend':oracle.engine,'proof':proof,
         'minimum_slack_moves':minimum_slack,'slack_moves':state[6]//layout.step_cost,'budget_floor':policy_floor,
         'geometry_sha256':fingerprint,'geometry_d4_sha256':fingerprint,'geometry_split':partition,
         'geometry_version':'dihedral-v1','solution_mechanics':dict(used),'patroller_count':len(layout.patrollers),
         'tick_period':layout.tick_period,'changing_attributes':len(p['changed_kinds']),
         'distractor_count':0,'nonrequired_distractor_count':0,'non_required_distractor_count':0}
    errors=profile_errors(row)
    if errors:return None,'; '.join(errors)
    row['gameplay_sha256']=gameplay_hash(row)
    return row,None


def generate_level(seed,difficulty=1,attempts=400,min_slack=None,search_limit=None,*,split=None,record_rejection=None):
    if type(difficulty) is not int or difficulty not in DIFFICULTIES:raise ValueError('difficulty must be 1..7')
    if attempts<1 or (search_limit is not None and not 0<search_limit<=32000000) or (min_slack is not None and min_slack<0):
        raise ValueError('positive attempts/search limit <=32000000 and nonnegative slack required')
    if split is None:split='validation' if 1000000<=seed<2000000 or seed>=8000000 else 'train'
    if split not in ('train','validation'):raise ValueError('split must be train or validation')
    rng=random.Random(f'{DIFFICULTY_VERSION}:{seed}:{difficulty}')
    exclusions=Counter()
    for attempt in range(1,attempts+1):
        # Cheap geometry gates and holdout redraws do not spend an oracle search.
        spec=None
        for _ in range(64):
            spec=draft(rng,difficulty)
            if spec is None:exclusions['invalid_geometry']+=1;continue
            if geometry_d4_partition(spec)[1]!=split:exclusions['geometry_split']+=1;spec=None;continue
            break
        if spec is None:reason='geometry_attempts_exhausted';accepted=None
        else:
            spec.update(seed=seed,generation_attempt=attempt,split=split,source='generated_only')
            accepted,reason=verify(spec,search_limit,min_slack)
        if accepted:
            accepted['generation_exclusions']=dict(exclusions)
            return accepted
        exclusions[reason]+=1
        if record_rejection:record_rejection(dict(seed=seed,difficulty=difficulty,attempt=attempt,reason=reason))
    raise RuntimeError(f'no reference-profile level seed={seed} difficulty={difficulty} after {attempts} attempts: {dict(exclusions)}')
