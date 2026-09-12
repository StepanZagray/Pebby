"""One-step field fidelity objective and additive diagnostics, not a planning gate.

Appearance supervision distills frozen probabilities, not engine semantic truth.
Visibility/goal weights come only from detached target fields. Changed cells use
an observed probability difference threshold, not a privileged change label.
"""
import torch
from torch.nn import functional as F

from .structured_transition import steps_targets

CHANGED_THRESHOLD = .05
EVENT_NAMES = ('lost_life', 'terminal', 'won')
GROUPS = {'core': (0,48), 'appearance': (48,70), 'carried': (70,84),
          'visibility': (84,85), 'padding': (85,96)}
DEFAULT_WEIGHTS = {**{f'field_{name}':1. for name in GROUPS}, 'field_changed':1.,
                   **{f'readout_{name}':1. for name in ('player','glyph','steps','lives','roles','goal')},
                   'events':1.}


def _integer(value, shape, name, low, high):
    if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype not in
            (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)):
        raise ValueError(f'{name} must be integer {shape}')
    result=value.long()
    if bool(((result<low)|(result>high)).any()):raise ValueError(f'{name} outside {low}..{high}')
    return result


def _validate(fields,next_fields,labels,scale):
    if (not isinstance(fields,torch.Tensor) or fields.ndim!=3 or tuple(fields.shape[1:])!=(148,96)
            or not fields.is_floating_point() or not len(fields)):
        raise ValueError('fields must be nonempty floating [B,148,96]')
    if (not isinstance(next_fields,torch.Tensor) or next_fields.shape!=fields.shape or
            not next_fields.is_floating_point() or next_fields.device!=fields.device):
        raise ValueError('next_fields must match floating current fields')
    if not bool(torch.isfinite(fields).all() and torch.isfinite(next_fields).all()):raise ValueError('fields must be finite')
    for field in (fields,next_fields):
        probabilities=field[:,:144,48:85]
        if bool(((probabilities<0)|(probabilities>1)).any()):raise ValueError('observed field probability channels must be in0..1')
    if (not isinstance(scale,torch.Tensor) or scale.shape!=(48,) or not scale.is_floating_point()
            or not bool(torch.isfinite(scale).all()) or bool((scale<.1).any())):
        raise ValueError('scale must be finite training std[48] with floor.1')
    scale=scale.detach().to(device=fields.device,dtype=torch.float32)
    batch=len(fields);checked={}
    for prefix in ('','next_'):
        for name,shape,low,high in (('player_cell',(batch,2),0,11),('triple',(batch,3),0,5),
                                   ('steps',(batch,),-3,42),('lives',(batch,),0,3)):
            key=prefix+name
            if key not in labels:raise ValueError(f'missing label {key}')
            value=_integer(labels[key],shape,key,low,high)
            if name=='triple' and bool((value[:,1:]>3).any()):raise ValueError('color/rotation labels must be in0..3')
            if value.device!=fields.device:raise ValueError('labels and fields must share a device')
            checked[key]=value
    for name in EVENT_NAMES:
        if name not in labels:raise ValueError(f'missing label {name}')
        value=labels[name]
        if not isinstance(value,torch.Tensor):raise ValueError(f'{name} must be a tensor')
        if value.dtype==torch.bool:value=value.long()
        checked[name]=_integer(value,(batch,),name,0,1)
        if value.device!=fields.device:raise ValueError('labels and fields must share a device')
    if bool((checked['won'].bool() & ~checked['terminal'].bool()).any()):raise ValueError('won must imply terminal')
    return checked,scale


def _player(cell):
    return cell[:,1]*12+cell[:,0]


def _changed(fields,next_fields):
    return (next_fields.detach()[:,:144,48:70]-fields.detach()[:,:144,48:70]).abs().amax(-1)>CHANGED_THRESHOLD


def _weighted_mean(values,weights):
    # Sum keeps a differentiable finite zero when target visibility/goal mass is0.
    return (values*weights).sum()/weights.sum().clamp_min(1e-8)


def _readout_losses(readout,target,labels,prefix):
    result={'player':F.cross_entropy(readout['player_logits'],_player(labels[prefix+'player_cell'])),
            'steps':F.cross_entropy(readout['steps_logits'],steps_targets(labels[prefix+'steps'])),
            'lives':F.cross_entropy(readout['lives_logits'],labels[prefix+'lives'])}
    result['glyph']=sum(F.cross_entropy(readout[f'carried_{name}_logits'],labels[prefix+'triple'][:,column])
                         for column,name in enumerate(('shape','color','rotation')))/3
    target=target.detach().float();visibility=target[:,:144,84]
    result['roles']=_weighted_mean(F.binary_cross_entropy_with_logits(readout['role_logits'].float(),
                                       target[:,:144,48:56],reduction='none').mean(-1),visibility)
    goal_weights=visibility*target[:,:144,49]
    goal_loss=0
    for name,start,stop in (('shape',56,62),('color',62,66),('rotation',66,70)):
        probabilities=target[:,:144,start:stop]
        mass=probabilities.sum(-1,keepdim=True)
        # Zero vectors on unsupported cells carry no attribute supervision.
        probabilities=probabilities/mass.clamp_min(1e-8)
        values=-(probabilities*F.log_softmax(readout[f'goal_{name}_logits'].float(),-1)).sum(-1)
        goal_loss=goal_loss+_weighted_mean(values,goal_weights*(mass.squeeze(-1)>0))/3
    result['goal']=goal_loss
    return result


def objective(model,fields,next_fields,actions,labels,scale,*,weights=None,pos_weight=None):
    """Chosen action branch per level; scale is TRAIN-derived std with floor.1.

    Optional fixed training event pos_weight[3] uses lost_life/terminal/won order
    and is capped at20. All observed fields and targets are detached internally.
    Returned readouts are current/actual/predicted; output is model.forward's dict.
    """
    labels,scale=_validate(fields,next_fields,labels,scale)
    fields=fields.detach();target=next_fields.detach()
    effective=dict(DEFAULT_WEIGHTS)
    if weights is not None:
        if set(weights)-set(effective):raise ValueError('unknown objective weight')
        effective.update(weights)
    if any(not isinstance(v,(int,float)) or not torch.isfinite(torch.tensor(v)) or v<0 for v in effective.values()):
        raise ValueError('weights must be finite and nonnegative')
    if pos_weight is None:positive=torch.ones(3,device=fields.device)
    else:
        positive=torch.as_tensor(pos_weight,device=fields.device,dtype=torch.float32).detach()
        if positive.shape!=(3,) or not bool(torch.isfinite(positive).all()) or bool((positive<=0).any()):
            raise ValueError('pos_weight must be finite positive[3]')
        positive=positive.clamp(max=20)
    output=model(fields,actions)
    readouts={'predicted':output['readout'],'actual':model.readout(target),'current':model.readout(fields)}
    predicted=output['field'].float();reference=target.float();losses={}
    for name,(start,stop) in GROUPS.items():
        error=predicted[...,start:stop]-reference[...,start:stop]
        if name=='core':error=error/scale
        losses['field_'+name]=error.square().mean()
    changed=_changed(fields,target)
    changed_error=(predicted[:,:144,48:70]-reference[:,:144,48:70]).square().mean(-1)
    losses['field_changed']=_weighted_mean(changed_error,changed.float())
    state_losses=[_readout_losses(readouts[name],t,labels,prefix) for name,t,prefix in
                  (('predicted',target,'next_'),('actual',target,'next_'),('current',fields,''))]
    for name in state_losses[0]:losses['readout_'+name]=sum(item[name] for item in state_losses)/3
    event_logits=torch.stack([output['events'][name+'_logits'] for name in EVENT_NAMES],-1).float()
    event_targets=torch.stack([labels[name] for name in EVENT_NAMES],-1).float()
    losses['events']=F.binary_cross_entropy_with_logits(event_logits,event_targets,pos_weight=positive)
    total=sum(effective[name]*loss for name,loss in losses.items())
    return dict(total=total,losses=losses,output=output,readouts=readouts,
                effective_pos_weight=positive.detach())


@torch.no_grad()
def diagnostics(result,fields,next_fields,labels,scale):
    """JSON-friendly additive sums/counts; aggregate before dividing, never average ratios.

    Event confusion entries are literal counts (their denominator is sample count).
    false_safe covers life loss or terminal failure, not unlabelled unreachable states.
    Changed appearance/goal metrics are frozen-teacher agreement, NOT gold fidelity.
    """
    labels,scale=_validate(fields,next_fields,labels,scale)
    sums={};counts={};batch=len(fields)
    def add(name,values,mask=None):
        values=values.detach().float()
        if mask is not None:values=values[mask]
        sums[name]=float(values.sum());counts[name]=int(values.numel())
    moved=(labels['player_cell']!=labels['next_player_cell']).any(-1)
    reset=labels['lost_life'].bool()
    for source,prefix in (('predicted','next_'),('actual','next_'),('current',''),('copy','next_')):
        readout=result['readouts']['current' if source=='copy' else source]
        correct=readout['player_logits'].argmax(-1)==_player(labels[prefix+'player_cell'])
        add(source+'_player_accuracy',correct)
        if source!='current':
            for name,mask in (('moved',moved),('stationary',~moved),('reset',reset)):
                add(source+'_'+name+'_player_accuracy',correct,mask)
        add(source+'_steps_accuracy',readout['steps_logits'].argmax(-1)==steps_targets(labels[prefix+'steps']))
        add(source+'_lives_accuracy',readout['lives_logits'].argmax(-1)==labels[prefix+'lives'])
        correct=torch.stack([readout[f'carried_{name}_logits'].argmax(-1)==labels[prefix+'triple'][:,column]
                             for column,name in enumerate(('shape','color','rotation'))],-1).all(-1)
        add(source+'_glyph_joint_accuracy',correct)
        teacher=fields if source=='current' else next_fields
        # These are thresholded frozen-teacher agreements, never gold semantics.
        visible=teacher[:,:144,84]>=.5
        goals=visible&(teacher[:,:144,49]>=.5)
        correct=torch.stack([readout[f'goal_{name}_logits'].argmax(-1)==teacher[:,:144,start:stop].argmax(-1)
                             for name,start,stop in (('shape',56,62),('color',62,66),('rotation',66,70))],-1).all(-1)
        add(source+'_teacher_goal_joint_agreement',correct,goals)
        role_agreement=((readout['role_logits']>=0)==(teacher[:,:144,48:56]>=.5)).all(-1)
        add(source+'_teacher_visible_role_exact_agreement',role_agreement,visible)
    for name in EVENT_NAMES:
        truth=labels[name].bool();prediction=result['output']['events'][name+'_logits']>=0
        for label,values in (('tp',truth&prediction),('fp',~truth&prediction),('fn',truth&~prediction),
                             ('tn',~truth&~prediction),('positive',truth)):
            add('event_'+name+'_'+label,values)
    truth_unsafe=reset|(labels['terminal'].bool()&~labels['won'].bool())
    e=result['output']['events'];predicted_unsafe=(e['lost_life_logits']>=0)|((e['terminal_logits']>=0)&(e['won_logits']<0))
    add('event_unsafe_false_safe_rate',~predicted_unsafe,truth_unsafe)
    add('lost_life_false_safe_rate',e['lost_life_logits']<0,reset)
    add('next_steps_underflow',labels['next_steps']<0)
    changed=_changed(fields,next_fields)
    for source,predicted in (('predicted',result['output']['field']),('copy',fields)):
        for group,(start,stop) in GROUPS.items():
            error=predicted[...,start:stop].float()-next_fields[...,start:stop].float()
            if group=='core':error=error/scale
            add(source+'_field_'+group+'_mse',error.square())
        error=(predicted[:,:144,48:70].float()-next_fields[:,:144,48:70].float()).square().mean(-1)
        add(source+'_changed_appearance_mse',error,changed)
    add('changed_appearance_cell_fraction',changed)
    probabilities=result['output']['field'][... ,48:85]
    add('predicted_probability_out_of_range_fraction',(probabilities<0)|(probabilities>1))
    for name,target in (('current',fields),('next',next_fields)):
        add(name+'_teacher_visibility_mass',target[:,:144,84])
        add(name+'_teacher_visible_goal_mass',target[:,:144,84]*target[:,:144,49])
    return dict(sums=sums,counts=counts)
