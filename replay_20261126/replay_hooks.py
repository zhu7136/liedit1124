"""Two initializations, unchanged frozen actor/native physics, first-termination capture."""
import base64, hashlib, io, json, math, types
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
POLICY_SHA='8fe02e377239f879e59ff1c52a0b72ae07d082c6ed4f72ccde7de483d81f73a2'
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def replay_and_finish(env):
    from isaaclab.utils.math import quat_apply
    raw=env.unwrapped;robot=raw.scene['robot'];cmd=raw.command_manager.get_term('motion')
    assert raw.num_envs==2 and tuple(raw.observation_manager.group_obs_dim['policy'])==(154,) and raw.action_manager.total_action_dim==29
    assert list(raw.observation_manager.active_terms['policy'])==['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
    assert raw.cfg.events.push_robot is None and not raw.cfg.observations.policy.enable_corruption
    assert hashlib.sha256(Path(raw.cfg.commands.motion.motion_file).read_bytes()).hexdigest()=='12ceddb49bf5e8a795dc69f8132853d86655ec7126524cea6d166be0714c3714'
    assert hashlib.sha256((HERE/'input-policy.pt').read_bytes()).hexdigest()==POLICY_SHA
    actor=torch.jit.load(str(HERE/'input-policy.pt'),map_location=raw.device).eval()
    for param in actor.parameters():param.requires_grad_(False)
    dt=float(raw.step_dt);total=int(cmd.motion.time_step_total)
    assert total==1334 and abs(dt-.02)<1e-7
    original_reset=raw._reset_idx
    reset_calls=[]
    def suppress_reset(ids):
        reset_calls.append(ids.detach().cpu().tolist())
        # State is deliberately left untouched; caller captures the first terminal state.
    raw._reset_idx=suppress_reset
    def forbid_resampling(self,ids):
        assert len(ids)==0,'Unexpected reference wrap/resampling; replay must end before it'
    # No subsequent command resample may teleport either robot.
    cmd._resample_command=types.MethodType(forbid_resampling,cmd)
    cmd.time_steps[:]=0
    cmd.time_left[:]=1.e6
    root=robot.data.default_root_state.clone()
    root[:,:3]+=raw.scene.env_origins
    q=robot.data.default_joint_pos.clone();qd=robot.data.default_joint_vel.clone()
    root[0,:3]=cmd.motion._body_pos_w[0,0]+raw.scene.env_origins[0]
    root[0,3:7]=cmd.motion._body_quat_w[0,0]
    root[0,7:10]=cmd.motion._body_lin_vel_w[0,0]
    root[0,10:13]=cmd.motion._body_ang_vel_w[0,0]
    q[0,cmd.joint_indexes]=cmd.motion.joint_pos[0]
    qd[0,cmd.joint_indexes]=cmd.motion.joint_vel[0]
    root[1,7:]=0;qd[1,:]=0
    ids=torch.arange(2,device=raw.device,dtype=torch.long)
    robot.reset(ids)
    robot.write_joint_state_to_sim(q,qd,env_ids=ids)
    robot.write_root_state_to_sim(root,env_ids=ids)
    robot.set_joint_position_target(q)
    raw.scene.write_data_to_sim()
    # Forward kinematics only: no physics settling steps or reference-frame skipping.
    raw.sim.forward();raw.scene.update(dt)
    raw.action_manager.reset(ids)
    raw.episode_length_buf[:]=0
    raw.termination_manager.reset(ids)
    # Refresh relative reference bodies exactly as native command update, but net frame advance=0.
    cmd.time_steps[:]=-1
    cmd._update_command()
    assert bool((cmd.time_steps==0).all())
    masses=robot.root_physx_view.get_masses().to(raw.device)
    local_com=robot.root_physx_view.get_coms().to(raw.device)[...,:3]
    assert masses.shape[0]==2 and masses.shape[1]==cmd.motion._body_pos_w.shape[1]
    origins=raw.scene.env_origins
    def com_values():
        actual_com=robot.data.body_com_pos_w
        f=cmd.time_steps
        ref_com=cmd.motion._body_pos_w[f]+origins[:,None,:]+quat_apply(cmd.motion._body_quat_w[f],local_com)
        return ((actual_com*masses[...,None]).sum(1)/masses.sum(1)[:,None]-origins,
                (ref_com*masses[...,None]).sum(1)/masses.sum(1)[:,None]-origins)
    names=['reference_frame0','default_stance_frame0']
    columns=['time_s','reference_frame','com_x_m','com_y_m','com_z_m','reference_com_x_m','reference_com_y_m','reference_com_z_m','base_z_m','tilt_deg','joint_rmse_rad','anchor_position_error_m']
    active=[True,True];metrics=[[],[]];poses=[[],[]];results=[None,None]
    initial_q=robot.data.joint_pos.clone()
    initial_errors=(initial_q[:,cmd.joint_indexes]-cmd.motion.joint_pos[0]).square().mean(-1).sqrt()
    initial_root_errors=(robot.data.root_pos_w[0]-root[0,:3]).norm()
    assert torch.equal(masses[0],masses[1]) and torch.equal(local_com[0],local_com[1])
    assert raw.cfg.events.physics_material is None and raw.cfg.events.base_com is None
    event('replay_initialized',cases=names,source_task='TASK_20261126_542',policy_sha256=POLICY_SHA,actor_obs_dim=154,action_dim=29,step_dt=dt,reference_frames=total,num_envs=2,seed=42,training_updates=0,automatic_retries=0,noise=False,pushes=False,startup_randomization_retained=False,identical_nominal_physics_both_cases=True,automatic_environment_reset=False,initial_joint_rmse_rad=initial_errors.cpu().tolist(),reference_initial_root_error_m=float(initial_root_errors),body_names=list(robot.body_names),joint_names=list(robot.joint_names),controlled_joint_indices=cmd.joint_indexes.cpu().tolist(),body_masses=masses.cpu().tolist(),body_local_com=local_com.cpu().tolist(),default_joint_pos=robot.data.default_joint_pos.cpu().tolist(),initial_root_states=root.cpu().tolist())
    assert float(initial_errors[0])<1e-5 and float(initial_root_errors)<1e-5
    def capture(k,final=False):
        c,rc=com_values();f=cmd.time_steps
        quat=robot.data.root_quat_w
        tilt=torch.acos(torch.clamp(1-2*(quat[:,1].square()+quat[:,2].square()),-1,1))*180/math.pi
        rmse=(robot.data.joint_pos[:,cmd.joint_indexes]-cmd.motion.joint_pos[f]).square().mean(-1).sqrt()
        anchor=(cmd.robot_anchor_pos_w-cmd.anchor_pos_w).norm(dim=-1)
        for i in range(2):
            if not active[i]:continue
            row=[k*dt,int(f[i]),*c[i].tolist(),*rc[i].tolist(),float(robot.data.root_pos_w[i,2]-origins[i,2]),float(tilt[i]),float(rmse[i]),float(anchor[i])]
            assert np.isfinite(row).all()
            metrics[i].append(row)
            if k%2==0 or final:
                poses[i].append(np.concatenate(([k*dt,int(f[i])],(robot.data.root_pos_w[i]-origins[i]).cpu().numpy(),robot.data.root_quat_w[i].cpu().numpy(),robot.data.joint_pos[i].cpu().numpy())).astype(np.float32))
    def reasons(i):
        return [name for name in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(name)[i])]
    def finish_case(i,status,k,why):
        a=np.asarray(metrics[i]);results[i]=dict(case=names[i],status=status,time_s=round(k*dt,6),last_reference_frame=int(cmd.time_steps[i]),reference_total_frames=total,termination_terms=why,initial_com_z_m=float(a[0,4]),last_com_z_m=float(a[-1,4]),minimum_com_z_m=float(a[:,4].min()),max_tilt_deg=float(a[:,9].max()),last_tilt_deg=float(a[-1,9]),last_joint_rmse_rad=float(a[-1,10]),last_anchor_error_m=float(a[-1,11]),samples=len(a),training_updates=0)
        event('replay_case_result',**results[i]);active[i]=False
    capture(0)
    obs=raw.observation_manager.compute()['policy']
    with torch.inference_mode():
        for k in range(1,total):
            assert bool((cmd.time_steps==k-1).all()),'Reference frame progression changed'
            actions=actor(obs)
            assert actions.shape==(2,29) and bool(torch.isfinite(actions).all())
            obs_dict,_,terminated,truncated,_=raw.step(actions)
            done=(terminated|truncated).cpu().tolist()
            capture(k,final=any(done) or k==total-1)
            for i in range(2):
                if active[i] and done[i]:finish_case(i,'first_termination',k,reasons(i))
                elif active[i] and k==total-1:finish_case(i,'reference_completed',k,[])
            if not any(active):break
            obs=obs_dict['policy']
    assert all(r is not None for r in results)
    assert hashlib.sha256((HERE/'input-policy.pt').read_bytes()).hexdigest()==POLICY_SHA
    arrays={}
    for i,name in enumerate(names):
        arrays[name+'_metrics']=np.asarray(metrics[i],dtype=np.float32)
        arrays[name+'_poses']=np.asarray(poses[i],dtype=np.float32)
    arrays['metric_columns']=np.asarray(columns)
    arrays['joint_names']=np.asarray(robot.joint_names)
    metadata=dict(source_task='TASK_20261126_542',source_policy_sha256=POLICY_SHA,training_updates=0,policy_unchanged=True,results=results,metrics_columns=columns,render_note='Actual closed-loop native Isaac Lab trajectory, not reference animation',reset_attempts_intercepted=len(reset_calls),no_auto_reset_performed=True)
    arrays['metadata_json']=np.asarray(json.dumps(metadata))
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue()
    assert len(data)<=2*1024**2
    (HERE/'replay-trajectories.npz').write_bytes(data)
    encoded=base64.b64encode(data).decode();chunks=[encoded[j:j+1200] for j in range(0,len(encoded),1200)];digest=hashlib.sha256(data).hexdigest()
    event('replay_data_begin',bytes=len(data),sha256=digest,chunks=len(chunks))
    for j,chunk in enumerate(chunks):event('replay_data_chunk',index=j,data_b64=chunk)
    event('replay_data_complete',bytes=len(data),sha256=digest,chunks=len(chunks),training_updates=0)
    metadata.update(trajectory_sha256=digest,trajectory_bytes=len(data))
    (HERE/'replay-completion.json').write_text(json.dumps(metadata,indent=2))
    event('replay_complete',**metadata)