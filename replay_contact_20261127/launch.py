"""Bounded contact diagnostic: compact source, unchanged native assets/controller."""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,signal,stat,subprocess,sys,types,zipfile
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));importlib.invalidate_caches()
def event(name,**kw):print(json.dumps(dict(event=name,**kw)),flush=True)
event('compact_entry_import_begin',training_updates=0)
import force_program as force
event('compact_entry_import_complete',training_updates=0)
POLICY='8fe02e377239f879e59ff1c52a0b72ae07d082c6ed4f72ccde7de483d81f73a2'
def restore(work,manifest):
    archive=HERE/'workload.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256'];assert not work.exists()
    expected=manifest['workload_files']
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(expected) and set(z.namelist())==set(expected)
        assert sum(x.file_size for x in z.infolist())<100*1024**2
        for name,digest in expected.items():
            p=PurePosixPath(name);i=z.getinfo(name);assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and not stat.S_ISLNK(i.external_attr>>16)
            data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest
            target=work/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
    event('compact_source_verified',files=len(expected),archive_bytes=archive.stat().st_size,all_retained_original_files_sha_verified=True,omitted_wheel_bytes=manifest['omitted_wheel_bytes'],training_updates=0)
def image_dependencies(work):
    assert sys.version_info[:2]==(3,11)
    approved=json.loads((work/'flux/offline-wheel-manifest.json').read_text());wheel=work/'flux/wheelhouse/argcomplete-3.6.2-py3-none-any.whl';spec=approved[wheel.name]
    assert wheel.stat().st_size==spec['bytes'] and hashlib.sha256(wheel.read_bytes()).hexdigest()==spec['sha256']
    try:present=importlib.metadata.version('argcomplete')=='3.6.2'
    except importlib.metadata.PackageNotFoundError:present=False
    if not present:
        deps=work/'.flux-deps';deps.mkdir(exist_ok=True);lock=work/'.compact-argcomplete.lock';lock.write_text('argcomplete @ '+wheel.resolve().as_uri()+' --hash=sha256:'+spec['sha256']+'\n')
        p=subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-cache-dir','--no-deps','--require-hashes','--only-binary=:all:','--target',str(deps),'-r',str(lock)],capture_output=True,text=True,timeout=30)
        assert p.returncode==0,'Pinned offline argcomplete installation failed'
        sys.path.insert(0,str(deps));os.environ['PYTHONPATH']=str(deps)+os.pathsep+os.environ.get('PYTHONPATH','');importlib.invalidate_caches()
    required=['torch','tensordict','numpy','gymnasium','hydra','argcomplete','prettytable','tensorboard','git','onnx','isaaclab','isaaclab_rl']
    missing=[n for n in required if importlib.util.find_spec(n) is None]
    event('compact_dependency_audit',missing=missing,online_install=False,offline_wheels=1,omitted_wheels_expected_preinstalled=True)
    assert not missing,missing
    assert importlib.metadata.version('rsl-rl-lib')=='3.1.2' and importlib.metadata.version('argcomplete')=='3.6.2'
    import torch
    assert torch.__version__.startswith('2.7.0') and torch.cuda.is_available()
    x=torch.ones(16,device='cuda:0');assert (x+x).sum().item()==32;torch.cuda.synchronize()
    event('runtime_preflight_verified',python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,rsl_rl='3.1.2',offline_installed=0 if present else 1,training_updates=0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--policy-url');p.add_argument('--child',action='store_true');p.add_argument('--work',type=Path);a=p.parse_args()
    os.environ.update(PYTHONUNBUFFERED='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    if a.child:
        shim=types.ModuleType('offline_dependencies');shim.prepare_dependencies=lambda:image_dependencies(a.work);sys.modules['offline_dependencies']=shim
        force.base.child(a.work);return
    if not a.child:
        manifest_pre=json.loads((HERE/'manifest.json').read_text())
        for name,digest in manifest_pre['entry_files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    process=None
    def cutoff(*unused):
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        event('replay_hard_timeout',entry_limit_seconds=240);os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(240)
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=10).strip();assert actual==a.expected_commit
    manifest=json.loads((HERE/'manifest.json').read_text());assert manifest['training_updates']==0 and manifest['automatic_retries']==0
    for name,h in manifest['entry_files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    assert a.policy_url and urlsplit(a.policy_url).scheme=='https' and urlsplit(a.policy_url).hostname=='limx-gradmotion.oss-cn-beijing.aliyuncs.com'
    with urlopen(a.policy_url,timeout=20) as response:data=response.read(2*1024**2+1)
    a.policy_url=None;assert len(data)<=2*1024**2 and hashlib.sha256(data).hexdigest()==POLICY
    with (HERE/'input-policy.pt').open('xb') as f:f.write(data)
    event('replay_policy_verified',bytes=len(data),sha256=POLICY,public_git_policy=False)
    work=HERE/'verified-workload';restore(work,manifest)
    launcher=next((str(x) for x in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')] if x.is_file() and os.access(x,os.X_OK)),sys.executable)
    try:
        import time
        limit=time.monotonic()+205
        for collision_variant in ['original','contact']:
            event('collision_variant_begin',variant=collision_variant,training_updates=0)
            child_env=os.environ.copy();child_env['REPLAY_COLLISION_VARIANT']='filtered';child_env['REPLAY_REFERENCE_VARIANT']=collision_variant
            process=subprocess.Popen([launcher,str(Path(__file__).resolve()),'--expected-commit',actual,'--child','--work',str(work)],start_new_session=True,env=child_env)
            code=process.wait(timeout=min(95,max(1,limit-time.monotonic())));assert code==0,f'Replay child exited {code}'
            assert (HERE/'replay-completion.json').is_file() and hashlib.sha256((HERE/'input-policy.pt').read_bytes()).hexdigest()==POLICY
            for name in ['replay-completion.json','replay-trajectories.npz']:
                (HERE/(collision_variant+'-'+name)).write_bytes((HERE/name).read_bytes())
            event('collision_variant_complete',variant=collision_variant,training_updates=0)

        event('replay_worker_finished',training_updates=0,automatic_retries=0,policy_sha256_unchanged=POLICY,child_exit_code=code)
    finally:
        if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        signal.alarm(0)
if __name__=='__main__':main()