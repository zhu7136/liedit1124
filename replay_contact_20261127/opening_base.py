"""Four 3s native opening diagnostics; frozen policy, no learning/reset/resampling."""
import base64,hashlib,io,json,math,types
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
POLICY_SHA='8fe02e377239f879e59ff1c52a0b72ae07d082c6ed4f72ccde7de483d81f73a2'
CASES=['actor_frame0','actor_frame5','actor_frame50','reference_pd_frame0']
STARTS=[0,5,50,0]
def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)
def replay_and_finish(env):
    from isaaclab.utils.math import quat_apply
    raw=env.unwrapped;robot=raw.scene['robot'];cmd=raw.command_manager.get_term('motion');term=raw.action_manager.get_term('joint_pos')
    n=len(CASES);assert raw.num_envs==n and tuple(raw.observation_manager.group_obs_dim['policy'])==(154,) and raw.action_manager.total_action_dim==29
    assert list(raw.observation_manager.active_terms['policy'])==['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
    assert not raw.cfg.observations.policy.enable_corruption and all(getattr(raw.cfg.events,k) is None for k in ['push_robot','physics_material','base_com','add_joint_default_pos'])
    assert hashlib.sha256(Path(raw.cfg.commands.motion.motion_file).read_bytes()).hexdigest()==__import__('os').environ['REPLAY_REFERENCE_SHA']
    assert hashlib.sha256((HERE/'input-policy.pt').read_bytes()).hexdigest()==POLICY_SHA
    actor=torch.jit.load(str(HERE/'input-policy.pt'),map_location=raw.device).eval()
    for param in actor.parameters():param.requires_grad_(False)
    dt=float(raw.step_dt);assert abs(dt-.02)<1e-7 and cmd.motion.time_step_total==1334
    starts=torch.tensor(STARTS,device=raw.device,dtype=torch.long)
    reset_calls=[]
    def no_reset(ids):reset_calls.append(ids.detach().cpu().tolist())
    raw._reset_idx=no_reset
    def no_resample(self,ids):assert len(ids)==0
    cmd._resample_command=types.MethodType(no_resample,cmd)
    cmd.time_steps[:]=starts;cmd.time_left[:]=1.e6
    root=robot.data.default_root_state.clone();root[:,:3]=cmd.motion._body_pos_w[starts,0]+raw.scene.env_origins
    root[:,3:7]=cmd.motion._body_quat_w[starts,0];root[:,7:10]=cmd.motion._body_lin_vel_w[starts,0];root[:,10:13]=cmd.motion._body_ang_vel_w[starts,0]
    q=robot.data.default_joint_pos.clone();qd=robot.data.default_joint_vel.clone();q[:,cmd.joint_indexes]=cmd.motion.joint_pos[starts];qd[:,cmd.joint_indexes]=cmd.motion.joint_vel[starts]
    ids=torch.arange(n,device=raw.device,dtype=torch.long);robot.reset(ids)
    robot.write_joint_state_to_sim(q,qd,env_ids=ids);robot.write_root_state_to_sim(root,env_ids=ids);robot.set_joint_position_target(q)
    raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(dt);raw.action_manager.reset(ids);raw.episode_length_buf[:]=0;raw.termination_manager.reset(ids)
    cmd.time_steps[:]=starts-1;cmd._update_command();assert torch.equal(cmd.time_steps,starts)
    masses=robot.root_physx_view.get_masses().to(raw.device);local_com=robot.root_physx_view.get_coms().to(raw.device)[...,:3]
    mapping=json.loads((Path.cwd()/'flux/motion_order.json').read_text())['body_mapping'];refids=torch.tensor([mapping[name] for name in robot.body_names],device=raw.device,dtype=torch.long)
    assert masses.shape==(n,len(robot.body_names)) and torch.equal(refids[cmd.body_indexes],cmd.body_indexes)
    qerr=(robot.data.joint_pos[:,cmd.joint_indexes]-cmd.motion.joint_pos[starts]).square().mean(-1).sqrt()
    verr=(robot.data.joint_vel[:,cmd.joint_indexes]-cmd.motion.joint_vel[starts]).square().mean(-1).sqrt()
    perr=(robot.data.root_pos_w-root[:,:3]).norm(dim=-1)
    assert float(torch.max(qerr))<1e-5 and float(torch.max(verr))<1e-5 and float(torch.max(perr))<1e-5
    event('opening_initialized',cases=CASES,start_frames=STARTS,initial_joint_rmse_rad=qerr.cpu().tolist(),initial_joint_vel_rmse_rad_s=verr.cpu().tolist(),initial_root_errors_m=perr.cpu().tolist(),initial_root_states=root.cpu().tolist(),body_names=list(robot.body_names),joint_names=list(robot.joint_names),body_masses=masses.cpu().tolist(),body_local_com=local_com.cpu().tolist(),action_scale=term._scale.detach().cpu().tolist() if torch.is_tensor(term._scale) else term._scale,action_offset=term._offset.detach().cpu().tolist() if torch.is_tensor(term._offset) else term._offset,training_updates=0,horizon_seconds=3.,pd_baseline='Next reference joint position, native implicit PD gains/effort limits; NOT an ideal dynamics-feasibility oracle',automatic_environment_reset=False)
    columns=['time_s','reference_frame','com_x_m','com_y_m','com_z_m','reference_com_x_m','reference_com_y_m','reference_com_z_m','base_z_m','tilt_deg','joint_rmse_rad','anchor_position_error_m']
    active=[True]*n;metrics=[[] for _ in range(n)];poses=[[] for _ in range(n)];results=[None]*n
    def capture(k,final=False):
        f=cmd.time_steps;origins=raw.scene.env_origins
        assert int(f.max())<=200, "Diagnostic prefix bound exceeded"
        actual=(robot.data.body_com_pos_w*masses[...,None]).sum(1)/masses.sum(1)[:,None]-origins
        rp=cmd.motion._body_pos_w[f][:,refids];rq=cmd.motion._body_quat_w[f][:,refids]
        ref=((rp+quat_apply(rq,local_com))*masses[...,None]).sum(1)/masses.sum(1)[:,None]
        quat=robot.data.root_quat_w;tilt=torch.acos(torch.clamp(1-2*(quat[:,1].square()+quat[:,2].square()),-1,1))*180/math.pi
        rmse=(robot.data.joint_pos[:,cmd.joint_indexes]-cmd.motion.joint_pos[f]).square().mean(-1).sqrt();anchor=(cmd.robot_anchor_pos_w-cmd.anchor_pos_w).norm(dim=-1)
        for i in range(n):
            if not active[i]:continue
            row=[k*dt,int(f[i]),*actual[i].tolist(),*ref[i].tolist(),float(robot.data.root_pos_w[i,2]-origins[i,2]),float(tilt[i]),float(rmse[i]),float(anchor[i])];assert np.isfinite(row).all();metrics[i].append(row)
            if k%2==0 or final:poses[i].append(np.concatenate(([k*dt,int(f[i])],(robot.data.root_pos_w[i]-origins[i]).cpu().numpy(),robot.data.root_quat_w[i].cpu().numpy(),robot.data.joint_pos[i].cpu().numpy())).astype(np.float32))
    capture(0);obs=raw.observation_manager.compute()['policy']
    with torch.inference_mode():
        for k in range(1,151):
            assert torch.equal(cmd.time_steps,starts+k-1)
            actions=actor(obs);assert actions.shape==(n,29) and bool(torch.isfinite(actions).all())
            pd_actions=(cmd.motion.joint_pos[cmd.time_steps+1]-term._offset)/term._scale;actions[3]=pd_actions[3]
            if k<=3:event('opening_action',step=k,actions=actions.cpu().tolist(),observation=obs.cpu().tolist())
            obsd,_,terminated,truncated,_=raw.step(actions);done=(terminated|truncated).cpu().tolist();capture(k,final=any(done) or k==150)
            for i in range(n):
                if not active[i] or not(done[i] or k==150):continue
                why=[name for name in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(name)[i])]
                a=np.asarray(metrics[i]);results[i]=dict(case=CASES[i],start_frame=STARTS[i],status='first_termination' if done[i] else 'survived_3s_censored_not_full_motion',time_s=round(k*dt,6),last_reference_frame=int(cmd.time_steps[i]),termination_evaluated_reference_frame=int(cmd.time_steps[i])-1 if done[i] else None,termination_terms=why,initial_com_z_m=float(a[0,4]),last_com_z_m=float(a[-1,4]),minimum_com_z_m=float(a[:,4].min()),max_tilt_deg=float(a[:,9].max()),last_tilt_deg=float(a[-1,9]),last_joint_rmse_rad=float(a[-1,10]),last_anchor_error_m=float(a[-1,11]),samples=len(a),training_updates=0)
                event('replay_case_result',**results[i]);active[i]=False
            if not any(active):break
            obs=obsd['policy']
    assert all(r is not None for r in results) and hashlib.sha256((HERE/'input-policy.pt').read_bytes()).hexdigest()==POLICY_SHA
    arrays={'metric_columns':np.asarray(columns),'joint_names':np.asarray(robot.joint_names)}
    for i,name in enumerate(CASES):arrays[name+'_metrics']=np.asarray(metrics[i],dtype=np.float32);arrays[name+'_poses']=np.asarray(poses[i],dtype=np.float32)
    metadata=dict(source_task='TASK_20261126_542',source_policy_sha256=POLICY_SHA,training_updates=0,policy_unchanged=True,results=results,metrics_columns=columns,reset_attempts_intercepted=len(reset_calls),no_auto_reset_performed=True,horizon_seconds=3.,baseline_is_not_feasibility_oracle=True)
    arrays['metadata_json']=np.asarray(json.dumps(metadata));stream=io.BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue();assert len(data)<=2*1024**2
    (HERE/'replay-trajectories.npz').write_bytes(data);encoded=base64.b64encode(data).decode();chunks=[encoded[j:j+1200] for j in range(0,len(encoded),1200)];digest=hashlib.sha256(data).hexdigest()
    event('replay_data_begin',bytes=len(data),sha256=digest,chunks=len(chunks))
    for j,chunk in enumerate(chunks):event('replay_data_chunk',index=j,data_b64=chunk)
    event('replay_data_complete',bytes=len(data),sha256=digest,chunks=len(chunks),training_updates=0)
    metadata.update(trajectory_sha256=digest,trajectory_bytes=len(data));(HERE/'replay-completion.json').write_text(json.dumps(metadata,indent=2));event('replay_complete',**metadata)