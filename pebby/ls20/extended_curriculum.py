"""Shared proof failure and puzzle identity contracts for LS20 generation.

The historical extended-curriculum trainer and generator live in Git history.
The gameplay identity is preserved for existing multi-game corpus records.
"""
import hashlib
import json


class ContractMismatch(RuntimeError):
    """A planner/engine disagreement is an investigation blocker, never a rejected draft."""


def gameplay_hash(spec):
    # Preserve the canonicalization used by existing corpus certificates.
    gameplay = {'walls': sorted(spec['walls']), 'start': spec['start'],
                'start_triple': spec['start_triple'], 'goals': sorted(spec['goals'], key=lambda x: x['cell']),
                'cyclers': sorted(spec['cyclers'], key=lambda x: (x['cell'], x['kind'])),
                'rails': sorted([sorted(r['cells']) for r in spec.get('rails', [])]),
                'launchers': sorted(spec.get('launchers', []), key=lambda x: x['cell']),
                'refills': sorted(spec['refills']), 'step_counter': spec['step_counter'],
                'step_cost': spec['step_cost'], 'fog': spec['fog']}
    if 'difficulty_version' in spec:
        from .provenance import generated_context
        gameplay['context_index'] = generated_context(spec)
    return hashlib.sha256(json.dumps(gameplay, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
