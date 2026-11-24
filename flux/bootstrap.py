"""One authorized 30-minute, one-GPU execution. No restart and no API keys."""
import argparse,base64,hashlib,importlib.metadata,importlib.util,json,os,runpy,signal,subprocess,sys,time,urllib.parse,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/toe_training'
WORKER_SECONDS=1800
SOFT_SECONDS=1620
CHILD_SECONDS=1740
STARTUP_SECONDS=420

def event(name,**kw):print(json.dumps({'event':name,**kw},allow_nan=False),flush=True)

def verify_inputs():
    manifest=json.loads((ROOT/'flux/source_manifest.json').read_text())
    for rel,wanted in manifest.items():
        p=ROOT/rel
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=wanted:raise RuntimeError('Source hash mismatch: '+rel)
    deps=json.loads((ROOT/'flux/dependencies.json').read_text())
    for entry in deps['fallback_wheels']:
        raw=(ROOT/entry['path']).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual!=entry['sha'] or len(raw)!=entry['bytes']:raise RuntimeError('Offline wheel hash mismatch')
    event('source_hashes_verified',files=len(manifest),fallback_wheels=len(deps['fallback_wheels']))

def prepare_dependencies():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Requires image CPython 3.11')
    # pip is isolated/offline/no-deps, writing only a project-owned overlay.
    deps=ROOT/'.flux-deps';deps.mkdir()
    mapping={'absl_py':'absl','cloudpickle':'cloudpickle','gitdb':'gitdb','gitpython':'git','grpcio':'grpc','importlib_metadata':'importlib_metadata','markdown':'markdown','markupsafe':'markupsafe','ml_dtypes':'ml_dtypes','onnx':'onnx','orjson':'orjson','prettytable':'prettytable','smmap':'smmap','tensorboard':'tensorboard','tensorboard_data_server':'tensorboard_data_server','tensordict':'tensordict','typing_extensions':'typing_extensions','werkzeug':'werkzeug','zipp':'zipp','wcwidth':'wcwidth','trimesh':'trimesh'}
    wheels=[]
    for wheel in sorted((ROOT/'flux/wheelhouse').glob('*.whl')):
        dist=wheel.name.split('-')[0]
        if dist=='rsl_rl_lib':
            try:present=importlib.metadata.version('rsl-rl-lib')=='3.1.2'
            except importlib.metadata.PackageNotFoundError:present=False
        else:
            module=mapping.get(dist)
            if module is None:raise RuntimeError('Unapproved offline dependency '+dist)
            present=importlib.util.find_spec(module) is not None
        if not present:wheels.append(str(wheel))
    if wheels:
        result=subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--target',str(deps),*wheels],capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError('Offline dependency install failed: '+result.stderr[-1500:])
    sys.path.insert(0,str(deps))
    os.environ['PYTHONPATH']=str(deps)+os.pathsep+os.environ.get('PYTHONPATH','')
    for module in ['torch','tensordict','numpy','gymnasium','hydra','argcomplete','prettytable','tensorboard','git','onnx','isaaclab','isaaclab_rl']:
        if importlib.util.find_spec(module) is None:raise RuntimeError('Image missing required import: '+module)
    if importlib.metadata.version('rsl-rl-lib')!='3.1.2':raise RuntimeError('RSL-RL version must be 3.1.2')
    import torch
    if not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Expected image Torch2.7.0 with CUDA; no CPU fallback')
    x=torch.ones(16,device='cuda:0');assert (x+x).sum().item()==32;torch.cuda.synchronize()
    event('runtime_preflight_verified',python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,rsl_rl='3.1.2',offline_installed=len(wheels))

def worker(started,resume_mount):
    os.environ.update(TOE_WORKER_STARTED=str(started),TOE_SOFT_DEADLINE=str(started+SOFT_SECONDS),CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    sys.path[:0]=[str(ROOT/'source/limx_rl_lab'),str(ROOT/'flux'),str(ROOT/'scripts/rsl_rl')]
    # Discover only documented IsaacLab source installation directories.
    if importlib.util.find_spec('isaaclab') is None:
        for base in [Path('/workspace/isaaclab/source'),Path('/isaaclab/source'),Path('/opt/isaaclab/source')]:
            if (base/'isaaclab/isaaclab').is_dir():
                sys.path[:0]=[str(base/pkg) for pkg in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]
                break
    verify_inputs()
    mount=Path(resume_mount)
    candidates=[mount] if mount.is_file() else list(mount.rglob('*.pt')) if mount.is_dir() else []
    if len(candidates)!=1 or hashlib.sha256(candidates[0].read_bytes()).hexdigest()!='84946a9c531e60b38a926874506c855fe95747a955f63cb1ac2b3c1f19a969a5':
        raise RuntimeError('Missing, ambiguous, or changed Flux checkpoint; no fresh fallback')
    resume_path=candidates[0].resolve()
    event('resume_input_verified',checkpoint_sha256=hashlib.sha256(resume_path.read_bytes()).hexdigest(),checkpoint_bytes=resume_path.stat().st_size,parent_task_id='TASK_20261124_187')
    prepare_dependencies();os.chdir(ROOT)
    sys.argv=[str(ROOT/'scripts/rsl_rl/train.py'),'--task','LimX-HU-D04-01-Flat-BeyondMimic','--resume','--checkpoint',str(resume_path),'--motion_file',str(ROOT/'motions/gui00528900_toe_vertical_parallel_candidate/motion.npz'),'--num_envs','4096','--max_iterations','100000','--seed','42','--run_name','toe_vertical_parallel_4096_resume_30min','--logger','tensorboard','--headless','--device','cuda:0']
    runpy.run_path(str(ROOT/'scripts/rsl_rl/train.py'),run_name='__main__')

def seal_and_archive(child_code,reason):
    record={'child_exit_code':child_code,'parent_stop_reason':reason,'worker_seconds':WORKER_SECONDS,'automatic_retries':0,'completion_present':(OUT/'completion.json').is_file()}
    (OUT/'parent_result.json').write_text(json.dumps(record,indent=2))
    files=[p for p in OUT.rglob('*') if p.is_file()]
    files += [p for p in (ROOT/'logs').rglob('*') if p.is_file() and p.suffix not in ['.pt','.onnx','.pkl']]
    if any(p.is_symlink() for p in files) or sum(p.stat().st_size for p in files)>120*1024**2:raise RuntimeError('Output bound exceeded')
    target=ROOT/'outputs/toe-training-output.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):archive.write(path,str(path.relative_to(ROOT)))
    event('output_ready_for_flux_archive',bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),credential_forwarding=False,**record)
    return child_code==0 and record['completion_present']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit');p.add_argument('--resume-mount',required=True);p.add_argument('--runtime-child',action='store_true');p.add_argument('--started',type=float)
    args=p.parse_args()
    if args.runtime_child:
        if args.started is None:raise RuntimeError('Missing monotonic parent start')
        worker(args.started,args.resume_mount);return
    started=time.monotonic();child=None
    def cutoff(*unused):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
        raise TimeoutError('Worker 1800-second limit reached')
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(WORKER_SECONDS)
    try:
        if not args.commit or OUT.exists():raise RuntimeError('Requires pinned commit, resume input and fresh output directory; no restart')
        actual=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        if actual!=args.commit:raise RuntimeError('Worker checkout commit differs from pinned commit')
        OUT.mkdir(parents=True)
        launcher=next((str(x) for x in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')] if x.is_file() and os.access(x,os.X_OK)),None)
        if launcher is None:
            if sys.version_info[:2]!=(3,11):raise RuntimeError('No documented IsaacSim Python launcher found')
            launcher=sys.executable
        event('worker_begin',commit=actual,worker_limit_seconds=WORKER_SECONDS,soft_stop_seconds=SOFT_SECONDS,startup_limit_seconds=STARTUP_SECONDS,num_envs=4096,automatic_retries=0,fresh_training=False,parent_task_id='TASK_20261124_187')
        child_log=OUT/'native.log'
        env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
        with child_log.open('w') as log:
            child=subprocess.Popen([launcher,str(Path(__file__).resolve()),'--runtime-child','--started',str(started),'--resume-mount',args.resume_mount],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            reason='child_exited';position=0
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=CHILD_SECONDS:reason='child_hard_deadline';break
                native_tail=child_log.open('rb')
                with native_tail:
                    native_tail.seek(0,2);size=native_tail.tell();native_tail.seek(max(0,size-65536))
                    native_text=native_tail.read().decode(errors='replace').lower()
                if any(marker in native_text for marker in ['out of memory','outofmemoryerror','cuda_error_out_of_memory','failed to allocate gpu memory','gpu memory allocation failed']):
                    event('gpu_memory_failure_detected',automatic_retries=0,environment_fallback=False)
                    reason='gpu_memory_failure';break
                event_file=OUT/'events.jsonl'
                initialized=False
                if event_file.exists():
                    text=event_file.read_text();initialized='environment_constructed' in text
                    if len(text)>position:
                        print(text[position:],end='',flush=True);position=len(text)
                if elapsed>=STARTUP_SECONDS and not initialized:reason='startup_watchdog';break
                time.sleep(2)
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        code=child.returncode
        event_file=OUT/'events.jsonl'
        if event_file.exists():
            text=event_file.read_text()
            if len(text)>position:print(text[position:],end='',flush=True)
        event('child_exit',code=code,reason=reason)
        if code!=0:
            tail=child_log.read_text(errors='replace')[-12000:]
            print('NATIVE_ERROR_TAIL\n'+tail,flush=True)
        if not seal_and_archive(code,reason):raise RuntimeError('Training did not finish with completed updates and final checkpoint')
    except BaseException as exc:
        event('worker_failed',error_type=type(exc).__name__)
        raise SystemExit(1)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        signal.alarm(0)

if __name__=='__main__':main()
