"""Bounded learned-field beam search. Infrastructure, not a calibrated controller.

Completion is rewarded only before the next lost life. Calibrated outcomes
must partition each transition into win/loss/continue. Models receive public
fields/actions only; no teacher or engine interface is accepted.
"""
import math
import torch
from pebby.agent.structured_distance import discounted_value


@torch.no_grad()
def search(current_field, world_model, distance_head, outcome_calibrator, *, depth=4, beam=4, gamma=.99, chunk_size=128):
    """Return selected actions and per-root best traces/expected-return terms.

    world_model(fields, actions) -> (next_fields, raw_event_logits).
    outcome_calibrator(raw) -> tensor[N,3], columns loss, win, continue.
    Root alternatives always retain separate beams. No module modes or weights
    are mutated; callers must supply deterministic evaluation-mode modules.
    """
    if type(depth)!=int or not 1<=depth<=4 or beam!=4 or type(beam)!=int:raise ValueError('depth1..4 and beam4 required')
    if type(chunk_size)!=int or not 1<=chunk_size<=128:raise ValueError('chunk_size1..128 required')
    if isinstance(gamma,bool) or not isinstance(gamma,(int,float)) or not math.isfinite(gamma) or not 0<gamma<=1:raise ValueError('gamma in (0,1] required')
    if not isinstance(current_field,torch.Tensor) or current_field.ndim!=3 or current_field.shape[1:]!=(148,96) or len(current_field)==0 or not current_field.is_floating_point() or not torch.isfinite(current_field).all():raise ValueError('finite public fields[B,148,96] required')
    for module in (world_model,distance_head,outcome_calibrator):
        if isinstance(module,torch.nn.Module) and module.training:raise ValueError('evaluation mode required')
    batches=[];counts=[]
    for initial in current_field:
        # tuples: field, accrued reward, surviving probability, action trace.
        groups=[[ (initial,0.,1.,()) ] for _ in range(4)];count=0
        for horizon in range(1,depth+1):
            candidates=[]
            for root,group in enumerate(groups):
                for field,reward,survival,trace in group:
                    for action in ([root] if horizon==1 else range(4)):
                        candidates.append((root,field,reward,survival,trace+(action,)))
            expanded=[[] for _ in range(4)]
            for start in range(0,len(candidates),chunk_size):
                part=candidates[start:start+chunk_size]
                fields=torch.stack([x[1] for x in part]);actions=torch.tensor([x[4][-1] for x in part],device=fields.device,dtype=torch.long)
                following,events=world_model(fields,actions)
                if following.shape!=fields.shape or following.device!=fields.device or not following.is_floating_point() or not torch.isfinite(following).all():raise ValueError('invalid predicted fields')
                probabilities=outcome_calibrator(events)
                if not isinstance(probabilities,torch.Tensor) or probabilities.shape!=(len(part),3) or probabilities.device!=fields.device or not probabilities.is_floating_point() or not torch.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any() or not torch.allclose(probabilities.sum(-1),torch.ones(len(part),device=fields.device),atol=1e-6,rtol=0):raise ValueError('outcomes must form coherent loss/win/continue probabilities')
                values=discounted_value(distance_head(following),gamma)
                if values.shape!=(len(part),):raise ValueError('distance batch mismatch')
                for i,(root,_,reward,survival,trace) in enumerate(part):
                    loss,win,cont=map(float,probabilities[i]);newreward=reward+gamma**horizon*survival*win;newsurvival=survival*cont
                    score=newreward+gamma**horizon*newsurvival*float(values[i])
                    expanded[root].append((following[i],newreward,newsurvival,trace,score))
                count+=len(part)
            # Python stable sorting plus lexicographic trace gives explicit ties.
            best=[sorted(group,key=lambda x:(-x[4],x[3]))[:beam] for group in expanded]
            groups=[[(f,r,s,t) for f,r,s,t,_ in group] for group in best]
        roots=[]
        for root,group in enumerate(best):
            _,r,s,t,score=group[0];roots.append({'root_action':root,'actions':list(t),'score':score,'reward':r,'survival':s,'bootstrap':score-r})
        chosen=min(roots,key=lambda x:(-x['score'],x['root_action']))['root_action']
        batches.append({'action':chosen,'roots':roots});counts.append(count)
    return {'actions':torch.tensor([x['action'] for x in batches],device=current_field.device,dtype=torch.long),'results':batches,'transition_counts':counts,'depth':depth,'beam_per_root':beam,'gamma':gamma}
