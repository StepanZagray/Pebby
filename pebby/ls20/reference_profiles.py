"""Aggregate LS20 reference contracts; no official geometry or routes are sampled.

Ranges are explicit calibration tolerances around one reference per tier, not
estimated population confidence intervals. See the characterization artifact.
"""
from collections import Counter

DIFFICULTY_VERSION = 'ls20-reference-v1'
DIFFICULTIES = tuple(range(1, 8))
KINDS = ('shape', 'color', 'rotation')
# actions, free cells, corridor fraction, changed kinds, goals, launchers,
# refills, step cost, moving rail (kind, shape, length), minimum used pads/tanks.
_VALUES = (
    (13, (32,44), (.25,.45), ('rotation',), 1,0,0,1, (), 0,0),
    (45, (49,65), (.30,.51), ('rotation',), 1,0,2,2, (), 0,2),
    (39, (57,73), (.37,.58), ('color','rotation'), 1,2,2,2, (), 2,1),
    (43, (59,75), (.34,.56), ('shape','color'), 1,8,2,1, (), 5,1),
    (44, (61,77), (.40,.62), KINDS, 1,8,3,2, (('rotation','line',3),), 4,1),
    (72, (61,77), (.30,.52), KINDS, 2,2,3,1,
     (('rotation','line',5),('shape','line',5),('color','ring',8)), 1,1),
    (53, (58,74), (.35,.56), KINDS, 1,3,6,2, (('rotation','line',6),), 1,2),
)
ACTION_RANGES = ((10,17),(36,54),(31,47),(34,52),(35,53),(58,86),(42,64))
SEARCH_LIMITS = (600000,600000,1000000,2000000,4000000,24000000,32000000)
PROFILES = {i: dict(reference_level=i, reference_actions=v[0],
                   actions=ACTION_RANGES[i-1], free_cells=v[1], corridor_fraction=v[2],
                   changed_kinds=v[3], goals=v[4], launchers=v[5], refills=v[6],
                   cost=v[7], rails=v[8], used_launchers=v[9], consumed_refills=v[10],
                   fog=i==7, context_index=i-1, search_limit=SEARCH_LIMITS[i-1])
            for i,v in enumerate(_VALUES,1)}


def structural_metrics(spec):
    free = {(x,y) for x in range(12) for y in range(12)} - {tuple(p) for p in spec['walls']}
    degree = Counter(sum((x+dx,y+dy) in free for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)))
                     for x,y in free)
    changed = tuple(k for i,k in enumerate(KINDS)
                    if any(g['triple'][i] != spec['start_triple'][i] for g in spec['goals']))
    return dict(free_cells=len(free), corridor_fraction=(degree[0]+degree[1]+degree[2])/max(1,len(free)),
                bbox_width=max(x for x,y in free)-min(x for x,y in free)+1,
                bbox_height=max(y for x,y in free)-min(y for x,y in free)+1,
                changed_kinds=changed)


def profile_errors(spec, *, require_proof=True):
    errors=[]
    d=spec.get('difficulty')
    if type(d) is not int or d not in PROFILES:
        return ['difficulty must be an integer in 1..7']
    p=PROFILES[d]
    if spec.get('difficulty_version') != DIFFICULTY_VERSION:
        errors.append('missing calibrated difficulty version')
    try:
        m=structural_metrics(spec)
        for key in ('free_cells','corridor_fraction'):
            if not p[key][0] <= m[key] <= p[key][1]: errors.append(f'{key} outside reference range')
        if m['changed_kinds'] != p['changed_kinds']: errors.append('changed attribute kinds differ from reference')
        low,high=(7,9) if d==1 else (9,10)
        if not all(low<=m[k]<=high for k in ('bbox_width','bbox_height')):
            errors.append('room extent differs from reference')
        for key in ('goals','launchers','refills'):
            if len(spec.get(key,[])) != p[key]: errors.append(f'{key} count differs from reference')
        if spec.get('step_cost') != p['cost'] or spec.get('step_counter') != 42:
            errors.append('budget rules differ from reference')
        if spec.get('fog') is not p['fog']: errors.append('fog differs from reference')
        if len({tuple(g['triple']) for g in spec['goals']}) != p['goals']:
            errors.append('goal targets must be distinct')
        if any(g['triple']==spec['start_triple'] for g in spec['goals']):
            errors.append('spawn already matches a goal')
        if Counter(c['kind'] for c in spec['cyclers']) != Counter(p['changed_kinds']):
            errors.append('cycler composition differs from reference')
        rails=spec.get('rails',[])
        if sorted(len(r['cells']) for r in rails) != sorted(r[2] for r in p['rails']):
            errors.append('moving rail lengths differ from reference')
        for kind,shape,length in p['rails']:
            matching=[r for r in rails if any(c['kind']==kind and tuple(c['cell']) in
                      {tuple(cell) for cell in r['cells']} for c in spec['cyclers'])]
            if len(matching)!=1: errors.append(f'missing moving {kind} rail');continue
            cells={tuple(c) for c in matching[0]['cells']}
            xs,ys={x for x,y in cells},{y for x,y in cells}
            if shape=='line' and not (len(xs)==1 or len(ys)==1):errors.append('bent reference line rail')
            if shape=='ring' and (len(xs)!=3 or len(ys)!=3 or (min(xs)+1,min(ys)+1) in cells):
                errors.append('reference color rail must be a 3x3 ring')
        if require_proof:
            policy_floor=8 if d==1 else 0
            if spec.get('budget_floor') != policy_floor or spec.get('minimum_slack_moves',-1)<policy_floor:
                errors.append('reference budget floor is missing or violated')
            if not p['actions'][0]<=spec.get('optimal_actions',-1)<=p['actions'][1]:
                errors.append('optimal action length outside reference range')
            if spec.get('training_context_index') != d-1:errors.append('context differs from reference level')
            used=spec.get('solution_mechanics',{})
            if used.get('used_launcher_count',0)<p['used_launchers']:errors.append('too few launcher pads exercised')
            if used.get('refills_consumed',0)<p['consumed_refills']:errors.append('too few refill tanks consumed')
            if used.get('used_patroller_count',0)<len(p['rails']):errors.append('not all reference moving kinds exercised')
    except (KeyError,TypeError,ValueError,IndexError) as error:
        errors.append(f'malformed reference spec: {error}')
    return errors
