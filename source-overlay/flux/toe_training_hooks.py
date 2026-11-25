"""Verified task418 continuation; task300 rewards/new motion remain unchanged."""
import hashlib,json,math,os,time,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/toe_training'
SOFT_DEADLINE=float(os.environ.get('TOE_SOFT_DEADLINE','0'))
SHA='d0cb1f807c9ae38820368dff57c666d33b47581d351886e5969b6cb8c332fd5a'
CHECKPOINT_SHA='6fcf5fa00b7aac83e3def8fb35be4e5a395a4621c1704a09aec9b1b76a3d398c'
PARENT_UPDATES=2094

def restore_verified_checkpoint(runner,path):
    import torch
    path=Path(path)
    if path.stat().st_size!=6724913 or hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:
        raise RuntimeError('Task418 resume checkpoint size/hash mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['iter']!=2093 or not checkpoint['optimizer_state_dict']['state']:
        raise RuntimeError('Unexpected parent iteration or missing Adam state')
    runner.load(str(path),load_optimizer=True,map_location=runner.device)
    def identical(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b
    if not identical(checkpoint['model_state_dict'],runner.alg.policy.state_dict()):
        raise RuntimeError('Actor/critic/noise restoration verification failed')
    if not identical(checkpoint['optimizer_state_dict'],runner.alg.optimizer.state_dict()):
        raise RuntimeError('Adam restoration verification failed')
    if runner.current_learning_iteration!=2093:raise RuntimeError('Native iteration restore failed')
    rates={float(g['lr']) for g in runner.alg.optimizer.param_groups}
    if len(rates)!=1:raise RuntimeError('Unexpected optimizer learning rates')
    runner.alg.learning_rate=rates.pop()
    runner.current_learning_iteration=PARENT_UPDATES
    runner._toe_resume_verified=True
    event('resume_verified',parent_task_id='TASK_20261125_418',checkpoint_sha256=CHECKPOINT_SHA,saved_iteration=2093,next_iteration=PARENT_UPDATES,parent_completed_updates=PARENT_UPDATES,optimizer_restored=True,model_tensors_restored=True,adaptive_learning_rate=runner.alg.learning_rate,environment_state_restored=False,rng_state_restored=False)

def cuda_snapshot(stage):
    import torch
    free,total=torch.cuda.mem_get_info(0)
    event('cuda_memory',stage=stage,free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))

def event(event_type,**kw):
    row={'event':event_type,'elapsed_seconds':round(time.monotonic()-float(os.environ.get('TOE_WORKER_STARTED',str(time.monotonic()))),3),**kw}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)

def audit_environment(env):
    raw=env.unwrapped; robot=raw.scene['robot']
    manifest=json.loads((ROOT/'flux/motion_order.json').read_text())
    cfg=raw.cfg.commands.motion
    body_indexes,names=robot.find_bodies(cfg.body_names,preserve_order=True)
    for name,index in zip(names,body_indexes):
        if manifest['body_mapping'].get(name)!=index:
            raise RuntimeError('Motion/robot body ordering mismatch: '+name)
    _,joint_names=robot.find_joints(cfg.joint_names,preserve_order=True)
    if joint_names!=manifest['joint_names']:
        raise RuntimeError('Motion joint ordering differs from expected 29-joint ordering')
    if raw.num_envs!=4096: raise RuntimeError('Refusing environment count other than approved 4096')
    expected_terms=['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
    if list(raw.observation_manager.active_terms['policy'])!=expected_terms:
        raise RuntimeError('Policy observations must match No-State-Estimation gamepad interface')
    if tuple(raw.observation_manager.group_obs_dim['policy'])!=(154,):
        raise RuntimeError('Actor must have exactly154 observation values')
    if raw.action_manager.total_action_dim!=29:
        raise RuntimeError('Policy must output29 controlled-joint actions')
    if raw.cfg.episode_length_s!=25.0:
        raise RuntimeError('Long reference requires the approved25s episode configuration')
    from isaaclab.envs.mdp import undesired_contacts
    assert raw.cfg.rewards.undesired_contacts.func is undesired_contacts
    assert not hasattr(raw.cfg.rewards,'hold_torso_upright')
    assert not hasattr(raw.cfg.rewards,'hold_four_support')
    assert 'hold_support' not in __import__('sys').modules
    motion=Path(cfg.motion_file)
    if hashlib.sha256(motion.read_bytes()).hexdigest()!=SHA:raise RuntimeError('Motion input hash changed')
    event('environment_constructed',num_envs=raw.num_envs,tracked_bodies=len(names),motion_sha256=SHA,joint_count=len(joint_names),body_order_verified=True,actor_obs_dim=154,action_dim=29,episode_length_s=25.0)
    event('reward_baseline_verified',baseline_task='TASK_20261124_300',upright_reward=False,four_support_reward=False,knee_contact_exemption=False,active_reward_terms=list(raw.reward_manager.active_terms))
    cuda_snapshot('environment_constructed')

class BudgetReached(Exception):pass

def run_bounded_training(runner,iterations,env):
    import torch
    if SOFT_DEADLINE<=time.monotonic()+10:raise RuntimeError('Insufficient remaining worker budget for training')
    original=runner.log;updates=0;start=time.monotonic()
    if runner.current_learning_iteration!=PARENT_UPDATES or not getattr(runner,'_toe_resume_verified',False):
        raise RuntimeError('Refusing fresh/unverified training; must continue task418 at2094')
    def bounded_log(locs,*args,**kwargs):
        nonlocal updates
        original(locs,*args,**kwargs)
        updates+=1
        if updates==1:cuda_snapshot('first_ppo_update')
        metrics={'iteration':int(locs['it']),'completed_updates':updates,'training_seconds':round(time.monotonic()-start,3)}
        # Existing runner host-side timings only; no extra GPU synchronization.
        for src,dst in [('collection_time','collection_seconds'),('learn_time','learning_seconds')]:
            value=float(locs[src])
            if not math.isfinite(value) or value<0:raise RuntimeError('Invalid timing: '+src)
            metrics[dst]=value
        elapsed=metrics['collection_seconds']+metrics['learning_seconds']
        metrics['steps_per_second']=4096*runner.num_steps_per_env/max(elapsed,1e-9)
        for key,value in locs.get('loss_dict',{}).items():
            v=float(value)
            if not math.isfinite(v):raise RuntimeError('Non-finite training loss: '+key)
            metrics['loss/'+key]=v
        for src,dst in [('rewbuffer','mean_reward'),('lenbuffer','mean_episode_length')]:
            values=list(locs.get(src,[]))
            if values:
                v=sum(values)/len(values)
                if not math.isfinite(v):raise RuntimeError('Non-finite '+dst)
                metrics[dst]=v
        command=env.unwrapped.command_manager.get_term('motion')
        for name,tensor in command.metrics.items():
            value=float(tensor.mean().item())
            if not math.isfinite(value):raise RuntimeError('Non-finite motion metric: '+name)
            metrics['motion/'+name]=value
        if updates==1 or updates%25==0 or time.monotonic()>=SOFT_DEADLINE:
            for parameter in runner.alg.policy.parameters():
                if not torch.isfinite(parameter).all():raise RuntimeError('Non-finite policy parameter')
        with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        if updates==1 or updates%25==0:event('training_update_completed',**metrics)
        if time.monotonic()>=SOFT_DEADLINE:raise BudgetReached()
    runner.log=bounded_log
    event('training_begin',iteration_safety_ceiling=iterations,soft_remaining_seconds=round(SOFT_DEADLINE-time.monotonic(),3),automatic_retries=0)
    reason='iteration_safety_ceiling'
    try:runner.learn(num_learning_iterations=iterations,init_at_random_ep_len=True)
    except BudgetReached:reason='worker_soft_deadline'
    if updates<1:raise RuntimeError('No completed PPO updates')
    destination=OUT/'final_checkpoint.pt'
    runner.save(str(destination))
    export_deployment(runner)
    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()
    record={'completed_updates':updates,'cumulative_updates':PARENT_UPDATES+updates,'parent_task_id':'TASK_20261125_418','parent_checkpoint_sha256':CHECKPOINT_SHA,'last_iteration':int(runner.current_learning_iteration),'training_seconds':round(time.monotonic()-start,3),'stop_reason':reason,'motion_sha256':SHA,'checkpoint_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'checkpoint_bytes':destination.stat().st_size,'num_envs':4096,'fresh_training':False,'optimizer_restored':True,'environment_state_restored':False,'rng_state_restored':False,'actor_obs_dim':154,'action_dim':29,'episode_length_s':25.0,'dynamics_validated':False,'hardware_approved':False}
    if record['checkpoint_bytes']<10000:raise RuntimeError('Empty final checkpoint')
    (OUT/'completion.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    event('checkpoint_saved',**record)

def export_deployment(runner):
    import torch
    import yaml
    from isaaclab_rl.rsl_rl import export_policy_as_jit,export_policy_as_onnx
    policy=runner.alg.policy
    normalizer=getattr(policy,'actor_obs_normalizer',None)
    export_policy_as_jit(policy,normalizer=normalizer,path=str(OUT),filename='policy.pt')
    export_policy_as_onnx(policy,normalizer=normalizer,path=str(OUT),filename='policy.onnx')
    source=OUT/'deploy.yaml'
    cfg=yaml.safe_load(source.read_text())
    if sum(len(v['scale']) for v in cfg['observations'].values())!=154:
        raise RuntimeError('Deployment observation metadata does not match154 inputs')
    if len(cfg['actions']['joint_pos']['joint_names'])!=29:
        raise RuntimeError('Deployment action metadata does not match29 joints')
    proof=json.loads((OUT/'deploy_verification.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==proof['controller_deploy_sha256']
    assert proof['motion_sha256']==SHA
    saved=torch.jit.load(str(OUT/'policy.pt'),map_location='cpu').eval()
    with torch.inference_mode():
        result=saved(torch.zeros(1,154))
    if tuple(result.shape)!=(1,29) or not torch.isfinite(result).all():
        raise RuntimeError('Exported TorchScript policy failed154->29 CPU smoke test')
    for name in ['policy.pt','policy.onnx','deploy.yaml']:
        p=OUT/name
        event('deployment_file_saved',name=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
