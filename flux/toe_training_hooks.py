"""Budget and provenance instrumentation; no reward/dynamics modifications."""
import hashlib,json,math,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/toe_training'
SOFT_DEADLINE=float(os.environ.get('TOE_SOFT_DEADLINE','0'))
SHA='68e6d580354a86caeed1b94e7869475fb3220c1fa4dc95894057301aed62fb35'

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
    if raw.num_envs!=64: raise RuntimeError('Refusing environment count other than approved 64')
    motion=Path(cfg.motion_file)
    if hashlib.sha256(motion.read_bytes()).hexdigest()!=SHA:raise RuntimeError('Motion input hash changed')
    event('environment_constructed',num_envs=raw.num_envs,tracked_bodies=len(names),motion_sha256=SHA,joint_count=len(joint_names),body_order_verified=True)

class BudgetReached(Exception):pass

def run_bounded_training(runner,iterations,env):
    import torch
    if SOFT_DEADLINE<=time.monotonic()+10:raise RuntimeError('Insufficient remaining worker budget for training')
    original=runner.log;updates=0;start=time.monotonic()
    def bounded_log(locs,*args,**kwargs):
        nonlocal updates
        original(locs,*args,**kwargs)
        updates+=1
        metrics={'iteration':int(locs['it']),'completed_updates':updates,'training_seconds':round(time.monotonic()-start,3)}
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
    if runner.writer is not None:
        runner.writer.flush()
        runner.writer.close()
    record={'completed_updates':updates,'last_iteration':int(runner.current_learning_iteration),'training_seconds':round(time.monotonic()-start,3),'stop_reason':reason,'motion_sha256':SHA,'checkpoint_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'checkpoint_bytes':destination.stat().st_size,'num_envs':64,'dynamics_validated':False,'hardware_approved':False}
    if record['checkpoint_bytes']<10000:raise RuntimeError('Empty final checkpoint')
    (OUT/'completion.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    event('checkpoint_saved',**record)
