"""Original GPU exporter, bounded config-only log transport, no PPO updates."""
import base64,hashlib,json,math,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
def event(name,**kw):print(json.dumps({'event':name,**kw},allow_nan=False),flush=True)
def export_and_finish(env):
    raw=env.unwrapped;robot=raw.scene['robot'];work=Path.cwd()
    expected=['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
    assert list(raw.observation_manager.active_terms['policy'])==expected
    assert tuple(raw.observation_manager.group_obs_dim['policy'])==(154,) and raw.action_manager.total_action_dim==29
    assert raw.num_envs==4096 and raw.cfg.episode_length_s==30.0 and raw.cfg.seed==42
    assert raw.cfg.observations.policy.motion_anchor_pos_b is None and raw.cfg.observations.policy.base_lin_vel is None
    assert hashlib.sha256(Path(raw.cfg.commands.motion.motion_file).read_bytes()).hexdigest()=='12ceddb49bf5e8a795dc69f8132853d86655ec7126524cea6d166be0714c3714'
    order=json.loads((work/'flux/motion_order.json').read_text());_,joints=robot.find_joints(raw.cfg.commands.motion.joint_names,preserve_order=True);assert joints==order['joint_names']
    from limx_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
    export_deploy_cfg(raw,str(HERE/'result'))
    target=HERE/'result/params/deploy.yaml';data=target.read_bytes();assert 100<len(data)<=65536
    digest=hashlib.sha256(data).hexdigest();encoded=base64.b64encode(data).decode();chunks=[encoded[i:i+1200] for i in range(0,len(encoded),1200)]
    dims=[list(c.func(raw,**c.params).shape[1:]) for c in raw.observation_manager._group_obs_term_cfgs['policy']]
    proof={'source_task':'TASK_20261126_542','source_commit':'325e3ba397440e6e183b69391b8966f3710cd078','actor_obs_dim':154,'action_dim':29,'observation_terms':expected,'observation_dims':dims,'robot_joint_names':list(robot.data.joint_names),'action_joint_names':joints,'sdk_joint_names':list(raw.cfg.scene.robot.joint_sdk_names),'step_dt':float(raw.cfg.sim.dt*raw.cfg.decimation),'sha256':digest,'bytes':len(data),'chunks':len(chunks),'num_envs':4096,'seed':42,'training_updates':0,'reexported_not_original':True,'hardware_approved':False}
    # Emit start/chunks/end only after the original YAML and runtime checks succeed.
    event('deploy_yaml_begin',**proof)
    for i,chunk in enumerate(chunks):event('deploy_yaml_chunk',index=i,data_b64=chunk)
    event('deploy_yaml_complete',sha256=digest,bytes=len(data),chunks=len(chunks),training_updates=0)
    (HERE/'export-completion.json').write_text(json.dumps(proof,indent=2))