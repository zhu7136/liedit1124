"""Budget and provenance instrumentation; no reward/dynamics modifications."""
import hashlib,json,math,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/toe_training'
SOFT_DEADLINE=float(os.environ.get('TOE_SOFT_DEADLINE','0'))
SHA='68e6d580354a86caeed1b94e7869475fb3220c1fa4dc95894057301aed62fb35'
CHECKPOINT_SHA='84946a9c531e60b38a926874506c855fe95747a955f63cb1ac2b3c1f19a969a5'
PARENT_UPDATES=936

def restore_verified_checkpoint(runner,path):
    import torch
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CHECKPOINT_SHA:
        raise RuntimeError('Resume checkpoint SHA256 mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['iter']!=935 or len(checkpoint['optimizer_state_dict']['state'])!=17:
        raise RuntimeError('Unexpected checkpoint iteration or Adam state count')
    runner.load(str(path),load_optimizer=True,map_location=runner.device)
    def identical(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b
    if not identical(checkpoint['model_state_dict'],runner.alg.policy.state_dict()):
        raise RuntimeError('Actor/critic/noise restoration verification failed')
    if not identical(checkpoint['optimizer_state_dict'],runner.alg.optimizer.state_dict()):
        raise RuntimeError('Adam optimizer restoration verification failed')
    if runner.current_learning_iteration!=935:
        raise RuntimeError('Native RSL-RL load did not restore iteration')
    rates={float(g['lr']) for g in runner.alg.optimizer.param_groups}
    if len(rates)!=1:raise RuntimeError('Unexpected optimizer learning rates')
    runner.alg.learning_rate=rates.pop()
    # Native save stores the last completed zero-based iteration, not the next.
    runner.current_learning_iteration=PARENT_UPDATES
    runner._toe_resume_verified=True
    event('resume_verified',checkpoint_sha256=CHECKPOINT_SHA,saved_iteration=935,next_iteration=PARENT_UPDATES,
          parent_completed_updates=PARENT_UPDATES,optimizer_restored=True,model_tensors_restored=True,
          adaptive_learning_rate=runner.alg.learning_rate,environment_state_restored=False,rng_state_restored=False)

def cuda_snapshot(stage):
    import torch
    free,total=torch.cuda.mem_get_info(0)
    event('cuda_memory',stage=stage,free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))

def event(name,**kw):
    row={'event':name,'elapsed_seconds':round(time.monotonic()-float(os.environ.get('TOE_WORKER_STARTED',str(time.monotonic()))),3),**kw}
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
    motion=Path(cfg.motion_file)
    if hashlib.sha256(motion.read_bytes()).hexdigest()!=SHA:raise RuntimeError('Motion input hash changed')
    event('environment_constructed',num_envs=raw.num_envs,tracked_bodies=len(names),motion_sha256=SHA,joint_count=len(joint_names),body_order_verified=True)
    cuda_snapshot('environment_constructed')

class BudgetReached(Exception):pass

def run_bounded_training(runner,iterations,env):
    import torch
    if SOFT_DEADLINE<=time.monotonic()+10:raise RuntimeError('Insufficient remaining worker budget for training')
    original=runner.log;updates=0;start=time.monotonic()
    if runner.current_learning_iteration!=PARENT_UPDATES or not getattr(runner,'_toe_resume_verified',False):
        raise RuntimeError('Refusing fresh or unverified training: resume must start at iteration 936')
    def bounded_log(locs,*args,**kwargs):
        nonlocal updates
        original(locs,*args,**kwargs)
        updates+=1
        if updates==1:cuda_snapshot('first_ppo_update')
        metrics={'iteration':int(locs['it']),'completed_updates':updates,'cumulative_updates':PARENT_UPDATES+updates,'training_seconds':round(time.monotonic()-start,3)}
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
        if updates==1 or updates%25==0 or time.monotonic()>=SOFT_DEADLINE:event('training_update_completed',**metrics)
        if time.monotonic()>=SOFT_DEADLINE:raise BudgetReached()
    runner.log=bounded_log
    event('training_begin',iteration_safety_ceiling=iterations,start_iteration=PARENT_UPDATES,fresh_training=False,soft_remaining_seconds=round(SOFT_DEADLINE-time.monotonic(),3),automatic_retries=0)
    reason='iteration_safety_ceiling'
    try:runner.learn(num_learning_iterations=iterations,init_at_random_ep_len=True)
    except BudgetReached:reason='worker_soft_deadline'
    if updates<1:raise RuntimeError('No completed PPO updates')
    destination=OUT/'final_checkpoint.pt'
    runner.save(str(destination))
    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()
    record={'completed_updates':updates,'cumulative_updates':PARENT_UPDATES+updates,'parent_task_id':'TASK_20261124_187','parent_checkpoint_sha256':CHECKPOINT_SHA,'last_iteration':int(runner.current_learning_iteration),'training_seconds':round(time.monotonic()-start,3),'stop_reason':reason,'motion_sha256':SHA,'checkpoint_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'checkpoint_bytes':destination.stat().st_size,'num_envs':4096,'fresh_training':False,'optimizer_restored':True,'dynamics_validated':False,'hardware_approved':False}
    if record['checkpoint_bytes']<10000:raise RuntimeError('Empty final checkpoint')
    (OUT/'completion.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    event('checkpoint_saved',**record)
