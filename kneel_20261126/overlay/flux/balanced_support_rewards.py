"""Hold-only ground load balancing. No actor/critic input dimension changes."""
import torch
SUPPORT_NAMES=['zuolun_link','youlun_link','left_ankle_roll_link','right_ankle_roll_link']
def measurements(env):
    cache=getattr(env,'_balanced_support_ids',None)
    robot=env.scene['robot'];sensor=env.scene.sensors['contact_forces']
    if cache is None:
        sensor_ids,names=sensor.find_bodies(SUPPORT_NAMES,preserve_order=True)
        assert names==SUPPORT_NAMES
        ids,_=robot.find_bodies(SUPPORT_NAMES,preserve_order=True)
        torso=robot.find_bodies('waist_pitch_link')[0][0]
        env._balanced_support_ids=(sensor_ids,ids,torso);cache=env._balanced_support_ids
    sensor_ids,ids,torso=cache
    # Self collisions disabled; these are ground contact forces, world Z upward.
    f=torch.clamp(sensor.data.net_forces_w[:,sensor_ids,2],min=0.)
    weight=robot.data.default_mass.sum(dim=-1).to(device=f.device,dtype=f.dtype)*9.81
    assert weight.device==f.device
    return f,weight,ids,torso
def gate(env):
    cmd=env.command_manager.get_term('motion')
    return torch.clamp((cmd.time_steps.float()-434.)/50.,0.,1.)
def force_balance(env):
    f,w,_,_=measurements(env);den=w.clamp(min=1.)
    knee=(f[:,0]-f[:,1])/den;foot=(f[:,2]-f[:,3])/den
    leg=(f[:,0]+f[:,2]-f[:,1]-f[:,3])/den
    return gate(env)*(4*knee.square()+4*foot.square()+2*leg.square())
def four_support(env):
    f,w,_,_=measurements(env)
    # Equal zero loads are NOT accepted as balance: each contact needs >=5% weight.
    coverage=torch.clamp(f/(.05*w[:,None]).clamp(min=1.),min=0.,max=1.)
    return gate(env)*coverage.prod(dim=-1)
def equal_load(env):
    f,w,_,_=measurements(env)
    return gate(env)*((f/w[:,None].clamp(min=1.)-.25).square().sum(dim=-1))
def force_smooth(env):
    f,w,_,_=measurements(env)
    prev=getattr(env,'_balanced_prev_forces',None)
    if prev is None:prev=f.detach().clone()
    penalty=((f-prev)/w[:,None].clamp(min=1.)).square().sum(dim=-1)
    # Never penalize a discontinuity caused by an episode reset as policy motion.
    reset=getattr(env,'reset_buf',None)
    if reset is not None:penalty=torch.where(reset.to(device=f.device,dtype=torch.bool),torch.zeros_like(penalty),penalty)
    env._balanced_prev_forces=f.detach().clone()
    return gate(env)*penalty
def hold_motion(env):
    from isaaclab.utils.math import quat_apply_inverse
    robot=env.scene['robot'];f,w,ids,torso=measurements(env)
    lin=robot.data.body_lin_vel_w[:,ids,:].square().sum(dim=(-1,-2))
    ang=robot.data.body_ang_vel_w[:,torso,:].square().sum(dim=-1)
    # foot local +X is forward. Raise the rear sole edge so the toe remains support.
    from isaaclab.utils.math import quat_apply
    fids=ids[2:];local=torch.tensor([-.112,0.,-.0635],device=env.device).expand(env.num_envs,2,3)
    heel_z=(robot.data.body_pos_w[:,fids,:]+quat_apply(robot.data.body_quat_w[:,fids,:],local))[...,2]
    heel_z=heel_z-env.scene.env_origins[:,2:3]
    heel_penalty=torch.clamp(.015-heel_z,min=0.).square().sum(dim=-1)*1000
    return gate(env)*(lin+ang+heel_penalty)
def hold_action_change(env):
    return gate(env)*(env.action_manager.action-env.action_manager.prev_action).square().sum(dim=-1)
def diagnostics(env):
    f,w,_,_=measurements(env);g=gate(env)>.99
    if not torch.any(g):return {'hold_env_count':0}
    a=f[g];wg=w[g];l=a[:,0]+a[:,2];r=a[:,1]+a[:,3];v={'hold_env_count':int(g.sum()),'four_contact_fraction':float((a>5.).all(dim=-1).float().mean()),'four_contact_fraction_over_5pct_weight':float((a>.05*wg[:,None]).all(dim=-1).float().mean()),'mean_abs_leg_imbalance':float((abs(l-r)/(l+r).clamp(min=1.)).mean()),'equal_load_error':float(((a/wg[:,None].clamp(min=1.)-.25).square().sum(dim=-1)).mean()),'mean_abs_knee_imbalance_bw':float((abs(a[:,0]-a[:,1])/wg.clamp(min=1.)).mean()),'mean_abs_foot_imbalance_bw':float((abs(a[:,2]-a[:,3])/wg.clamp(min=1.)).mean())}
    for i,n in enumerate(SUPPORT_NAMES):
        v[n+'_mean_Fz_N']=float(a[:,i].mean())
        v[n+'_mean_weight_fraction']=float((a[:,i]/wg.clamp(min=1.)).mean())
    return v