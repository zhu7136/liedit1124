"""Hold-only knee contact allowance/upright-support rewards; actor unchanged."""
import torch

HOLD_START=186
KNEES=['left_knee_link','right_knee_link']
SUPPORT=KNEES+['left_ankle_roll_link','right_ankle_roll_link']

def hold_mask(env):
    return env.command_manager.get_term('motion').time_steps>=HOLD_START

def gated_undesired_contacts(env,threshold,sensor_cfg):
    sensor=env.scene.sensors[sensor_cfg.name]
    force=sensor.data.net_forces_w_history[:,:,sensor_cfg.body_ids,:].norm(dim=-1).amax(dim=1)
    names=[sensor.body_names[i] for i in sensor_cfg.body_ids]
    knee=torch.tensor([n in KNEES for n in names],device=force.device,dtype=torch.bool)
    allowed=hold_mask(env)[:,None]&knee[None,:]
    return ((force>threshold)&~allowed).sum(dim=-1).float()

def torso_tilt(env,asset_cfg):
    q=env.scene[asset_cfg.name].data.body_quat_w[:,asset_cfg.body_ids[0]]
    cos_angle=(1-2*(q[:,1].square()+q[:,2].square())).clamp(-1,1)
    return torch.acos(cos_angle)

def hold_upright(env,asset_cfg,std=.15):
    angle=torso_tilt(env,asset_cfg)
    return hold_mask(env).float()*torch.exp(-angle.square()/std**2)

def support_flags(env,sensor_cfg,threshold=5.):
    f=env.scene.sensors[sensor_cfg.name].data.net_forces_w[:,sensor_cfg.body_ids,:]
    return f.norm(dim=-1)>threshold

def hold_support_contacts(env,sensor_cfg,threshold=5.):
    # Reward all four links contacting, not mere large force on one leg.
    return hold_mask(env).float()*support_flags(env,sensor_cfg,threshold).all(dim=-1).float()

def configure_hold_support(cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    cfg.rewards.undesired_contacts.func=gated_undesired_contacts
    cfg.rewards.hold_torso_upright=RewardTermCfg(func=hold_upright,weight=.5,
        params={'asset_cfg':SceneEntityCfg('robot',body_names=['waist_pitch_link'],preserve_order=True),'std':.15})
    cfg.rewards.hold_four_support=RewardTermCfg(func=hold_support_contacts,weight=.5,
        params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names=SUPPORT,preserve_order=True),'threshold':5.})

def hold_stats(env):
    raw=env.unwrapped;mask=hold_mask(raw);count=mask.sum().clamp(min=1)
    torso=raw.reward_manager.get_term_cfg('hold_torso_upright').params['asset_cfg']
    support=raw.reward_manager.get_term_cfg('hold_four_support').params['sensor_cfg']
    tilt=torso_tilt(raw,torso);flags=support_flags(raw,support)
    return {'hold/environment_count':int(mask.sum().item()),
        'hold/torso_tilt_deg':float((torch.rad2deg(tilt)*mask).sum().item()/count.item()),
        'hold/four_support_fraction':float((flags.all(dim=-1)&mask).sum().item()/count.item())}