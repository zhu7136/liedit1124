"""Instrument the previously verified four-case replay, without changing its controller."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import opening_base as base
HERE=Path(__file__).resolve().parent
def replay_and_finish(env):
    raw=env.unwrapped;robot=raw.scene['robot'];sensor=raw.scene['contact_forces']
    feet=[raw.scene['force_left'],raw.scene['force_right']]
    from pair_filter import verify_filter_authoring,variant
    base.event('pair_filter_verified',**verify_filter_authoring(raw))
    pairs=[[raw.scene['pair_'+label+'_'+side] for side in ['left','right']] for label in ['actor','pd']]
    for group in pairs:
        for s in group:assert tuple(s.data.force_matrix_w.shape)==(1,1,1,3),s.data.force_matrix_w.shape
    names=list(robot.body_names);footids=[names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
    sensorids=[list(sensor.body_names).index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
    assert len(base.CASES)==4 and abs(float(raw.physics_dt)-.005)<1e-8
    for s in feet:assert s.num_bodies==1 and s.data.force_matrix_w.shape[1:]==(1,1,3)
    selected=[0,3]  # Keep the two frame0 controller comparisons; other cases are unchanged repeat controls.
    state=dict(inside=False,control=0,substep=0,active=np.ones(4,dtype=bool))
    data={k:[] for k in ['time_s','active','joint_pos','joint_vel','joint_target','torque_computed_estimate_nm','torque_applied_estimate_nm','foot_ground_normal_force_w','foot_ground_friction_force_w','foot_ground_average_contact_pos_w','foot_net_normal_force_w','body_net_normal_force_norm_n','explicit_foot_knee_normal_force_w','root_state_w','com_w']}
    masses=robot.root_physx_view.get_masses().to(raw.device)
    original_step=raw.step;original_update=raw.scene.update;original_save=base.np.savez_compressed
    def asarray(x):return x.detach().cpu().numpy()[selected].copy()
    def update(dt,**kwargs):
        result=original_update(dt,**kwargs)
        if not state['inside']:return result
        state['substep']+=1;t=state['substep']*float(raw.physics_dt)
        # Access forces after scene timestamps advance, triggering current-physics-step sensor refresh.
        normal=torch.stack([s.data.force_matrix_w[:,0,0] for s in feet],dim=1)
        friction=torch.stack([s.data.friction_forces_w[:,0,0] for s in feet],dim=1)
        points=torch.stack([s.data.contact_pos_w[:,0,0] for s in feet],dim=1)-raw.scene.env_origins[:,None,:]
        net=sensor.data.net_forces_w
        com=(robot.data.body_com_pos_w*masses[...,None]).sum(1)/masses.sum(1)[:,None]-raw.scene.env_origins
        root=robot.data.root_state_w.clone();root[:,:3]-=raw.scene.env_origins
        values=dict(time_s=t,active=state['active'][selected].copy(),joint_pos=asarray(robot.data.joint_pos),joint_vel=asarray(robot.data.joint_vel),joint_target=asarray(robot.data.joint_pos_target),torque_computed_estimate_nm=asarray(robot.data.computed_torque),torque_applied_estimate_nm=asarray(robot.data.applied_torque),foot_ground_normal_force_w=asarray(normal),foot_ground_friction_force_w=asarray(friction),foot_ground_average_contact_pos_w=asarray(points),foot_net_normal_force_w=asarray(net[:,sensorids]),body_net_normal_force_norm_n=asarray(net.norm(dim=-1)),explicit_foot_knee_normal_force_w=torch.stack([torch.stack([s.data.force_matrix_w[0,0,0] for s in group],dim=0) for group in pairs],dim=0).detach().cpu().numpy().copy(),root_state_w=asarray(root),com_w=asarray(com))
        for k,v in values.items():data[k].append(v)
        return result
    def step(actions):
        state['inside']=True;state['control']+=1
        try:out=original_step(actions)
        finally:state['inside']=False
        done=(out[2]|out[3]).detach().cpu().numpy();state['active']&=~done
        return out
    def save(stream,**arrays):
        for k,v in data.items():arrays['force_'+k]=np.asarray(v,dtype=bool if k=='active' else (np.float32 if k in ['time_s','root_state_w','com_w','foot_ground_average_contact_pos_w'] else np.float16))
        arrays['force_case_names']=np.asarray([base.CASES[i] for i in selected])
        arrays['force_joint_effort_limits_nm']=asarray(robot.data.joint_effort_limits)
        arrays['force_joint_stiffness']=asarray(robot.data.joint_stiffness)
        arrays['force_joint_damping']=asarray(robot.data.joint_damping)
        arrays['force_sensor_body_names']=np.asarray(sensor.body_names)
        arrays['force_physics_dt_s']=np.asarray(float(raw.physics_dt))
        return original_save(stream,**arrays)
    from pxr import UsdPhysics
    import omni.usd
    ground=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse() if str(p.GetPath()).startswith('/World/ground') and p.HasAPI(UsdPhysics.CollisionAPI)]
    base.event('force_instrumentation',collision_variant=variant(),explicit_pairs='env0 actor and env3 PD foot-versus-ipsilateral-knee; exact prim paths',physics_dt_s=float(raw.physics_dt),cases=[base.CASES[i] for i in selected],foot_sensor_body_names=[list(s.body_names) for s in feet],filter_patterns=[s.cfg.filter_prim_paths_expr for s in feet],ground_collision_prims=ground,filtered_force_kind='normal only; friction recorded separately',contact_positions='unweighted mean of contact points, NOT center of pressure',torque_kind='Isaac Lab implicit PD approximate effort, NOT exact solver motor torque',telemetry_precision='joint/force/torque float16 for bounded log transport; time/pose/COM/contact-position float32',joint_names=list(robot.joint_names),sensor_body_names=list(sensor.body_names),body_masses=masses.cpu().tolist(),automatic_retries=0,training_updates=0)
    raw.step=step;raw.scene.update=update;base.np.savez_compressed=save
    try:base.replay_and_finish(env)
    finally:raw.step=original_step;raw.scene.update=original_update;base.np.savez_compressed=original_save
    normal=np.asarray(data['foot_ground_normal_force_w']);net=np.asarray(data['foot_net_normal_force_w']);active=np.asarray(data['active'])
    base.event('force_capture_complete',physics_samples=len(data['time_s']),per_case_active_samples=active.sum(0).tolist(),filtered_peak_vertical_n=np.max(normal[...,2],axis=(0,2)).tolist(),unfiltered_peak_vertical_n=np.max(net[...,2],axis=(0,2)).tolist(),nonfinite_contact_points_are_expected_when_not_in_contact=True,training_updates=0)