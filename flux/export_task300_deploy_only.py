"""ONE authorized <=600s worker: initialize task300 environment and export metadata only."""
import argparse,base64,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def event(event_type,**kw):print(json.dumps({'event':event_type,**kw},allow_nan=False),flush=True)
def send_file(path,name):
    raw=path.read_bytes();assert 0<len(raw)<180000
    encoded=base64.b64encode(raw).decode();chunks=[encoded[i:i+3000] for i in range(0,len(encoded),3000)]
    sha=hashlib.sha256(raw).hexdigest()
    for index,chunk in enumerate(chunks):event('deploy_file_chunk',name=name,index=index,total=len(chunks),bytes=len(raw),sha256=sha,base64=chunk)
    event('deploy_file_complete',name=name,bytes=len(raw),sha256=sha)

def child():
    os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    sys.path[:0]=[str(ROOT/'source/limx_rl_lab'),str(ROOT/'flux'),str(ROOT/'scripts/rsl_rl')]
    if importlib.util.find_spec('isaaclab') is None:
        for base in [Path('/workspace/isaaclab/source'),Path('/isaaclab/source'),Path('/opt/isaaclab/source')]:
            if (base/'isaaclab/isaaclab').is_dir():
                sys.path[:0]=[str(base/pkg) for pkg in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']];break
    # Reuse verified OFFLINE dependency installation, never bootstrap.worker()/training.
    spec=importlib.util.spec_from_file_location('task300_dependency_bootstrap',ROOT/'flux/bootstrap.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_inputs();module.prepare_dependencies()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True);app=launcher.app;env=None
    try:
        import gymnasium as gym
        import yaml
        import limx_rl_lab.tasks
        from limx_rl_lab.utils.parser_cfg import parse_env_cfg
        from limx_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
        from limx_rl_lab.tasks.beyondmimic.robots.limx.robot_cfg import OLI_PR_JOINT_NAMES
        cfg=parse_env_cfg('LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation',device='cuda:0',num_envs=1)
        cfg.seed=42
        motion=ROOT/'motions/gui00528900_toe_vertical_parallel_plus20s/motion.npz'
        assert hashlib.sha256(motion.read_bytes()).hexdigest()=='3d4bd08212cd96509af9ca46d4f9554b53004a3446dd4c64159c7555842c5fda'
        cfg.commands.motion.motion_file=str(motion)
        env=gym.make('LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation',cfg=cfg);raw=env.unwrapped
        terms=['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
        assert list(raw.observation_manager.active_terms['policy'])==terms
        assert tuple(raw.observation_manager.group_obs_dim['policy'])==(154,)
        assert raw.action_manager.total_action_dim==29 and raw.num_envs==1
        assert abs(raw.step_dt-0.02)<1e-9 and raw.cfg.episode_length_s==25.0
        out=ROOT/'outputs/task300_deploy_only';assert not out.exists();out.mkdir(parents=True)
        export_deploy_cfg(raw,str(out))
        native=out/'params/deploy.yaml'
        # Safe parsing: never instantiate Python objects from exported YAML tags.
        src=yaml.load(native.read_text(),Loader=yaml.BaseLoader)
        clean={'joint_ids_map':[int(x) for x in src['joint_ids_map']],'step_dt':float(src['step_dt'])}
        for key in ['stiffness','damping','default_joint_pos']:clean[key]=[float(x) for x in src[key]]
        action=src['actions']['joint_pos'];term=raw.action_manager._terms['joint_pos']
        ids=term._joint_ids
        if isinstance(ids,slice):ids=list(range(len(raw.scene['robot'].data.joint_names)))[ids]
        elif hasattr(ids,'tolist'):ids=ids.tolist()
        joints=[raw.scene['robot'].data.joint_names[i] for i in ids]
        assert joints==OLI_PR_JOINT_NAMES
        clean['actions']={'joint_pos':{'joint_names':joints,'joint_ids':[int(x) for x in ids],'scale':[float(x) for x in action['scale']]}}
        clean['observations']={}
        dims=[58,6,3,29,29,29]
        for name,dim in zip(terms,dims):
            item=src['observations'][name]
            scale=[float(x) for x in item['scale']];assert len(scale)==dim
            clip=item.get('clip')
            if clip in (None,'null','None','~'):clip=None
            else:clip=[float(x) for x in clip]
            history=int(item.get('history_length','1'));assert history==1
            clean['observations'][name]={'scale':scale,'clip':clip,'history_length':history}
        assert len(clean['joint_ids_map'])==31 and len(set(clean['joint_ids_map']))==31
        assert all(len(clean[k])==31 for k in ['default_joint_pos','stiffness','damping'])
        assert len(ids)==29 and len(action['scale'])==29
        destination=out/'deploy.yaml';destination.write_text(yaml.safe_dump(clean,sort_keys=False))
        assert yaml.safe_load(destination.read_text())==clean
        proof={'source_task':'TASK_20261124_300','source_commit':'dd2bf2387a5f4dcfd3eb8d9da37a44a251932259','regenerated':True,'training_updates':0,'sdk_commands_sent':0,'num_envs':1,'actor_obs_dim':154,'action_dim':29,'step_dt':raw.step_dt,'obs_terms':terms,'joint_names':joints,'original_deploy_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'controller_deploy_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'dynamics_validated':False,'hardware_approved':False}
        verification=out/'deploy_verification.json';verification.write_text(json.dumps(proof,indent=2))
        send_file(native,'deploy_native_regenerated.yaml')
        send_file(destination,'deploy.yaml')
        send_file(verification,'deploy_verification.json')
        event('deploy_export_succeeded',**proof)
    finally:
        if env is not None:env.close()
        app.close()

def deliver_from_parent():
    # This function runs without importing Isaac/Kit in the parent process.
    out=ROOT/'outputs/task300_deploy_only'
    proof=json.loads((out/'deploy_verification.json').read_text())
    assert proof['actor_obs_dim']==154 and proof['action_dim']==29
    assert proof['training_updates']==0 and proof['sdk_commands_sent']==0
    assert hashlib.sha256((out/'deploy.yaml').read_bytes()).hexdigest()==proof['controller_deploy_sha256']
    assert hashlib.sha256((out/'params/deploy.yaml').read_bytes()).hexdigest()==proof['original_deploy_sha256']
    files={'deploy_native_regenerated.yaml':(out/'params/deploy.yaml').read_bytes(),
           'deploy.yaml':(out/'deploy.yaml').read_bytes(),
           'deploy_verification.json':(out/'deploy_verification.json').read_bytes()}
    assert all(0<len(raw)<180000 for raw in files.values())
    # Secondary recovery path through the proven Flux PT-file uploader. This
    # is a data-only container, NOT policy weights, and is safely loadable.
    import torch
    container=out/'task300_deploy_delivery.pt'
    torch.save({'format':'task300_deploy_files_v1','files':files},container)
    event('deploy_backup_container_ready',name=container.name,bytes=container.stat().st_size,sha256=hashlib.sha256(container.read_bytes()).hexdigest())
    for name,path in [('deploy_native_regenerated.yaml',out/'params/deploy.yaml'),('deploy.yaml',out/'deploy.yaml'),('deploy_verification.json',out/'deploy_verification.json')]:
        send_file(path,name)
    event('deploy_export_succeeded',**proof)

def main():
    worker_started=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--commit');p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:return child()
    process=None
    def hard_budget(signum,frame):
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        event('deploy_worker_timeout',automatic_retries=0,hard_limit_seconds=600)
        os._exit(124)
    signal.signal(signal.SIGALRM,hard_budget);signal.alarm(600)
    assert a.commit
    actual=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    assert actual==a.commit
    event('deploy_only_worker_started',code_commit=actual,worker_limit_seconds=600,child_limit_seconds=570,automatic_retries=0,training=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--child']
    process=subprocess.Popen(command,cwd=str(ROOT),start_new_session=True)
    try:code=process.wait(timeout=570)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        event('deploy_worker_timeout',automatic_retries=0);raise SystemExit(124)
    if code==0:
        # Isaac may intercept child stdout or exit during app.close(). Recover
        # from files in the parent, outside Isaac's logging/shutdown hooks.
        deliver_from_parent()
        # Give Flux's native uploader a stability-scan window, never exceed
        # the approved600s worker window. No extra GPU work is performed.
        time.sleep(min(22,max(0,595-(time.monotonic()-worker_started))))
    event('deploy_only_worker_exit',code=code,automatic_retries=0)
    raise SystemExit(code)

if __name__=='__main__':main()