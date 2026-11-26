"""Resumed No-State-Estimation auditing, bounded PPO, and checkpoint provenance."""
import hashlib,json,math,os,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/kneel_training'
MOTION_SHA='12ceddb49bf5e8a795dc69f8132853d86655ec7126524cea6d166be0714c3714'
def event(kind,**kw):
    row={'event':kind,'elapsed_seconds':round(time.monotonic()-float(os.environ['KNEEL_STARTED']),3),**kw}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)

def audit_environment(env):
    import torch
    raw=env.unwrapped;robot=raw.scene['robot'];cfg=raw.cfg.commands.motion
    order=json.loads((ROOT/'flux/motion_order.json').read_text())
    indexes,names=robot.find_bodies(cfg.body_names,preserve_order=True)
    for name,index in zip(names,indexes):
        if order['body_mapping'][name]!=index:raise RuntimeError('Reference/robot body ordering mismatch: '+name)
    _,joints=robot.find_joints(cfg.joint_names,preserve_order=True)
    if joints!=order['joint_names']:raise RuntimeError('29-joint ordering mismatch')
    assert len(names)==14 and raw.num_envs==4096
    assert tuple(raw.observation_manager.group_obs_dim['policy'])==(154,)
    expected=['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
    assert list(raw.observation_manager.active_terms['policy'])==expected,'Must use original No-State-Estimation task observations'
    assert raw.cfg.observations.policy.motion_anchor_pos_b is None
    assert raw.cfg.observations.policy.base_lin_vel is None
    assert raw.action_manager.total_action_dim==29 and raw.cfg.episode_length_s==30.0
    from isaaclab.envs.mdp import undesired_contacts
    assert raw.cfg.rewards.undesired_contacts.func is undesired_contacts
    assert not hasattr(raw.cfg.rewards,'hold_torso_upright') and not hasattr(raw.cfg.rewards,'hold_four_support')
    assert hashlib.sha256(Path(cfg.motion_file).read_bytes()).hexdigest()==MOTION_SHA
    command=raw.command_manager.get_term('motion')
    assert command.motion.time_step_total==1334 and command.motion.joint_pos.shape[1]==29
    assert command.motion._body_pos_w.shape[1]==42
    assert torch.isfinite(command.motion.joint_pos).all()
    event('environment_constructed',num_envs=4096,tracked_bodies=14,recorded_bodies=42,actor_obs_dim=154,action_dim=29,episode_length_s=30,body_order_verified=True,motion_sha256=MOTION_SHA,reward_terms=list(raw.reward_manager.active_terms),baseline_rewards_preserved=True,state_estimation_inputs_removed=True)

class BudgetReached(Exception):pass

def run_bounded_training(runner,iterations,env,log_dir):
    import torch
    deadline=float(os.environ['KNEEL_SOFT_DEADLINE']);start=time.monotonic();updates=0
    from kneel_resume import verify_restored_runner, SOURCE_UPDATES
    resume_record=verify_restored_runner(runner)
    if deadline-start<30:raise RuntimeError('Startup consumed training budget')
    original=runner.log
    def bounded_log(locs,*args,**kwargs):
        nonlocal updates
        original(locs,*args,**kwargs);updates+=1
        row={'iteration':int(locs['it']),'completed_updates':updates,'cumulative_completed_updates':SOURCE_UPDATES+updates,'training_seconds':round(time.monotonic()-start,3)}
        for key,value in locs.get('loss_dict',{}).items():row['loss/'+key]=float(value)
        for source,dest in [('collection_time','collection_seconds'),('learn_time','learning_seconds')]:
            row[dest]=float(locs[source])
        for source,dest in [('rewbuffer','mean_reward'),('lenbuffer','mean_episode_length')]:
            values=list(locs.get(source,[]))
            if values:row[dest]=float(sum(values)/len(values))
        for name,tensor in env.unwrapped.command_manager.get_term('motion').metrics.items():
            row['motion/'+name]=float(tensor.mean().item())
        if any(not math.isfinite(v) for v in row.values()):raise RuntimeError('Non-finite PPO metric')
        if updates==1 or updates%25==0 or time.monotonic()>=deadline:
            if any(not torch.isfinite(p).all() for p in runner.alg.policy.parameters()):raise RuntimeError('Non-finite policy parameter')
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if updates==1 or updates%25==0:event('training_update_completed',**row)
        if time.monotonic()>=deadline:raise BudgetReached()
    runner.log=bounded_log
    event('training_begin',fresh_training=False,soft_remaining_seconds=round(deadline-start,3),automatic_retries=0)
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
    record={'completed_updates':updates,'cumulative_completed_updates':SOURCE_UPDATES+updates,'last_iteration':int(runner.current_learning_iteration),'training_seconds':round(time.monotonic()-start,3),'stop_reason':reason,'motion_sha256':MOTION_SHA,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'checkpoint_bytes':checkpoint.stat().st_size,'num_envs':4096,'fresh_training':False,'resume':resume_record,'actor_obs_dim':154,'action_dim':29,'episode_length_s':30,'dynamics_validated':False,'hardware_approved':False,'automatic_retries':0,'state_estimation_inputs_removed':True}
    (OUT/'completion.json').write_text(json.dumps(record,indent=2,allow_nan=False));event('checkpoint_saved',**record)