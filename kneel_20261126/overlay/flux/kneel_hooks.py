"""Fresh No-State-Estimation auditing, bounded PPO, and checkpoint provenance."""
import hashlib,json,math,os,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/kneel_training'
MOTION_SHA='bbb0a091d83d108b2a81c17a2ba20d20dc69ed92c88aa0fff0fdd6537a8e9a51'
def event(kind,**kw):
    row={'event':kind,'elapsed_seconds':round(time.monotonic()-float(os.environ['KNEEL_STARTED']),3),**kw}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)

def audit_environment(env):
    import torch
    raw=env.unwrapped;robot=raw.scene['robot'];command=raw.command_manager.get_term('motion')
    order=json.loads((ROOT/'flux/motion_order.json').read_text())
    assert raw.num_envs==4096 and raw.action_manager.total_action_dim==29
    assert tuple(raw.observation_manager.group_obs_dim['policy'])==(154,)
    assert tuple(raw.observation_manager.group_obs_dim['critic'])==(286,)
    assert raw.cfg.episode_length_s==30.0
    assert all(n in robot.body_names for n in ['zuolun_link','youlun_link'])
    assert list(command.motion._body_indexes)==[order['body_mapping'][n] for n in command.cfg.body_names]
    assert hashlib.sha256(Path(raw.cfg.commands.motion.motion_file).read_bytes()).hexdigest()==MOTION_SHA
    assert raw.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert command.motion.joint_vel[434:].abs().max()<1e-6
    assert set(['hold_force_balance','hold_four_support','hold_force_smooth','hold_motion','final_hold_action_rate_l2']).issubset(raw.reward_manager.active_terms)
    from balanced_support_rewards import measurements
    f,w,ids,torso=measurements(raw);assert f.shape==(4096,4) and torch.isfinite(w).all()
    assert (w>400).all() and (w<700).all()
    event('environment_constructed',num_envs=4096,tracked_bodies=14,recorded_bodies=42,actor_obs_dim=154,critic_obs_dim=286,action_dim=29,episode_length_s=30,body_order_verified_by_name=True,motion_sha256=MOTION_SHA,reward_terms=list(raw.reward_manager.active_terms),parent_task='TASK_20261128_005',optimizer_reset=True,physics_knee_wheels_present=True,self_collisions_disabled=True,hold_start_frame=434)

class BudgetReached(Exception):pass

def run_bounded_training(runner,iterations,env,log_dir):
    import torch
    deadline=float(os.environ['KNEEL_SOFT_DEADLINE']);start=time.monotonic();updates=0
    assert runner.current_learning_iteration==0 and getattr(runner,'_normal23000_warm_start_verified',False), 'Strict normal23000 weights must be loaded before PPO'
    if deadline-start<30:raise RuntimeError('Startup consumed training budget')
    from upright_preflight import native_preflight
    native_preflight(runner,env,event,OUT)
    original=runner.log
    def bounded_log(locs,*args,**kwargs):
        nonlocal updates
        original(locs,*args,**kwargs);updates+=1
        row={'iteration':int(locs['it']),'completed_updates':updates,'training_seconds':round(time.monotonic()-start,3)}
        for key,value in locs.get('loss_dict',{}).items():row['loss/'+key]=float(value)
        for source,dest in [('collection_time','collection_seconds'),('learn_time','learning_seconds')]:
            row[dest]=float(locs[source])
        for source,dest in [('rewbuffer','mean_reward'),('lenbuffer','mean_episode_length')]:
            values=list(locs.get(source,[]))
            if values:row[dest]=float(sum(values)/len(values))
        from balanced_support_rewards import diagnostics
        for key,value in diagnostics(env.unwrapped).items():row['support/'+key]=value
        for name,tensor in env.unwrapped.command_manager.get_term('motion').metrics.items():
            row['motion/'+name]=float(tensor.mean().item())
        if any(not math.isfinite(v) for v in row.values()):raise RuntimeError('Non-finite PPO metric')
        if updates==1 or updates%25==0 or time.monotonic()>=deadline:
            if any(not torch.isfinite(p).all() for p in runner.alg.policy.parameters()):raise RuntimeError('Non-finite policy parameter')
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if updates==1 or updates%25==0:event('training_update_completed',**row)
        if time.monotonic()>=deadline:raise BudgetReached()
    runner.log=bounded_log
    event('training_begin',fresh_training=False,source_iteration=999,optimizer_reset=True,soft_remaining_seconds=round(deadline-start,3),automatic_retries=0)
    reason='iteration_safety_ceiling'
    try:runner.learn(num_learning_iterations=iterations,init_at_random_ep_len=True)
    except BudgetReached:reason='worker_soft_deadline'
    finally:runner.log=original
    if updates<1:raise RuntimeError('No completed PPO updates')
    checkpoint=OUT/'final_checkpoint.pt';runner.save(str(checkpoint))
    assert checkpoint.stat().st_size>10000
    if runner.writer is not None:runner.writer.flush();runner.writer.close()
    for name in ['env.yaml','agent.yaml','deploy.yaml']:
        candidates=[Path(log_dir)/'params'/name,Path(log_dir)/name]
        for source in candidates:
            if source.is_file():shutil.copy2(source,OUT/name);break
    # Embed actual native configuration in the final checkpoint, so YAML is recoverable
    # from the platform's confirmed model download path, not only chart archives.
    native_deploy=OUT/'deploy.yaml'
    assert native_deploy.is_file(), 'Actual same-run deploy.yaml required for delivery'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    metadata={'format':'hud04_native_deploy_metadata_v1','deploy_yaml_utf8':native_deploy.read_text(),
              'deploy_yaml_sha256':hashlib.sha256(native_deploy.read_bytes()).hexdigest(),
              'motion_sha256':MOTION_SHA,'completed_updates':updates,'source_iteration':999,
              'actor_obs_dim':154,'critic_obs_dim':286,'actions':29,
              'code_commit':os.environ['KNEEL_COMMIT'],'yaml_origin':'actual same-run native export',
              'preflight_passed':True,'no_mp4':True,'hardware_approved':False}
    payload['deployment_metadata']=metadata
    torch.save(payload,checkpoint)
    torch.save(metadata,OUT/'native_deploy_metadata.pt')
    check=torch.load(checkpoint,map_location='cpu',weights_only=False)['deployment_metadata']
    assert hashlib.sha256(check['deploy_yaml_utf8'].encode()).hexdigest()==metadata['deploy_yaml_sha256']
    event('native_deploy_metadata_embedded',deploy_yaml_sha256=metadata['deploy_yaml_sha256'],completed_updates=updates)
    # Retain every 100-update checkpoint in the archived output for selection.
    saved_dir=OUT/'checkpoints';saved_dir.mkdir(exist_ok=True)
    for saved in sorted(Path(log_dir).glob('model_*.pt')):
        shutil.copy2(saved,saved_dir/saved.name)
    # Export failure does not invalidate an already-saved native training checkpoint.
    try:
        from isaaclab_rl.rsl_rl import export_policy_as_jit,export_policy_as_onnx
        policy=runner.alg.policy;normalizer=getattr(policy,'actor_obs_normalizer',None)
        export_policy_as_jit(policy,normalizer=normalizer,path=str(OUT),filename='policy.pt')
        export_policy_as_onnx(policy,normalizer=normalizer,path=str(OUT),filename='policy.onnx')
        saved=torch.jit.load(str(OUT/'policy.pt'),map_location='cpu').eval()
        with torch.inference_mode():result=saved(torch.zeros(1,154))
        assert result.shape==(1,29) and torch.isfinite(result).all()
        event('policy_export_verified',input_dim=154,output_dim=29)
    except Exception as exc:event('policy_export_failed',error_type=type(exc).__name__)
    record={'completed_updates':updates,'last_iteration':int(runner.current_learning_iteration),'training_seconds':round(time.monotonic()-start,3),'stop_reason':reason,'motion_sha256':MOTION_SHA,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'checkpoint_bytes':checkpoint.stat().st_size,'num_envs':4096,'fresh_training':False,'source_iteration':999,'optimizer_reset':True,'all_self_collisions_disabled':True,'motion_artifact':'balanced-four-support-reference-v3','actor_obs_dim':154,'action_dim':29,'episode_length_s':30,'dynamics_validated':False,'hardware_approved':False,'automatic_retries':0,'state_estimation_inputs_removed':True}
    (OUT/'completion.json').write_text(json.dumps(record,indent=2,allow_nan=False));event('checkpoint_saved',**record)