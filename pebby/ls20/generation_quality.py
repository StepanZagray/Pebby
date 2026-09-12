"""Budget quality shared by generated training curricula.

This is headroom on the verified route, not a guarantee that arbitrary mistakes
(especially those changing a moving cycler's phase) remain recoverable.
"""

import hashlib
import json


def route_budget_slack(layout, before, after, *, action=None, outcome=None):
    """Minimum remaining charged moves on either side of a transition.

    Include the pre-transition budget: stepping onto a refill resets it, which
    otherwise hides an exhausted approach behind a generous final budget.
    """
    minimum = min(before[6] // layout.step_cost, after[6] // layout.step_cost)
    if outcome == 'launched' and action is not None:
        from . import names
        dx, dy = names.ACTION_DELTAS[action]
        target = (before[0][0] + dx, before[0][1] + dy)
        refills = tuple(sorted(layout.refills))
        entry_refilled = (target in layout.refills
                          and not before[5] & (1 << refills.index(target)))
        # Launches charge before landing. A landing refill can hide this
        # intermediate budget; a refill at the entry tile instead skips charge.
        if not entry_refilled:
            minimum = min(minimum, (before[6] - layout.step_cost) // layout.step_cost)
    return minimum


def budget_floor(spec):
    """Keep forgiving lessons distinct from explicitly labeled challenge rows."""
    profile = spec.get('quality_profile', 'learning')
    if profile not in ('learning', 'challenge'):
        raise ValueError(f'unknown quality profile: {profile}')
    return 8 if profile == 'learning' else 0


def geometry_partition(spec):
    """Partition translation-normalized playable geometry, independently of seeds.

    Equal-sized buckets leave enough candidates for both splits. This holds out
    wall geometry in new v2 banks, not entire topology families or historical v1
    data. Translating a room cannot move it into the other split.
    """
    free = {(x, y) for x in range(12) for y in range(12)} - {tuple(p) for p in spec['walls']}
    left = min(x for x, _ in free)
    top = min(y for _, y in free)
    normalized = sorted((x - left, y - top) for x, y in free)
    fingerprint = hashlib.sha256(json.dumps(normalized, separators=(',', ':')).encode()).hexdigest()
    return fingerprint, 'validation' if int(fingerprint[:2], 16) % 2 else 'train'


def geometry_d4_hash(spec):
    """Canonical free geometry under translation, rotations and reflections."""
    free = {(x,y) for x in range(12) for y in range(12)} - {tuple(p) for p in spec['walls']}
    if not free:
        raise ValueError('empty playable geometry')
    variants=[]
    for swap in (False,True):
        for sx in (-1,1):
            for sy in (-1,1):
                points=[(sx*(y if swap else x),sy*(x if swap else y)) for x,y in free]
                left,top=min(x for x,y in points),min(y for x,y in points)
                variants.append(sorted((x-left,y-top) for x,y in points))
    return hashlib.sha256(json.dumps(min(variants),separators=(',',':')).encode()).hexdigest()


def geometry_d4_partition(spec):
    fingerprint=geometry_d4_hash(spec)
    return fingerprint, 'validation' if int(fingerprint[:2],16)%2 else 'train'
