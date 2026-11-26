import hashlib,importlib.metadata,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def event(event_type,**kw):print(json.dumps({"event":event_type,**kw}),flush=True)

def prepare_dependencies():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Requires image CPython 3.11')
    # pip is isolated/offline/no-deps, writing only a project-owned overlay.
    deps=ROOT/'.flux-deps';deps.mkdir()
    mapping={'absl_py':'absl','cloudpickle':'cloudpickle','gitdb':'gitdb','gitpython':'git','grpcio':'grpc','importlib_metadata':'importlib_metadata','markdown':'markdown','markupsafe':'markupsafe','ml_dtypes':'ml_dtypes','onnx':'onnx','orjson':'orjson','prettytable':'prettytable','smmap':'smmap','tensorboard':'tensorboard','tensorboard_data_server':'tensorboard_data_server','tensordict':'tensordict','typing_extensions':'typing_extensions','werkzeug':'werkzeug','zipp':'zipp','wcwidth':'wcwidth','trimesh':'trimesh'}
    mapping['argcomplete']='argcomplete'
    approved=json.loads((ROOT/'flux/offline-wheel-manifest.json').read_text())
    actual_wheels=sorted((ROOT/'flux/wheelhouse').glob('*.whl'))
    if {w.name for w in actual_wheels}!=set(approved):raise RuntimeError('Offline wheel set differs from pinned manifest')
    wheels=[]
    for wheel in actual_wheels:
        spec=approved[wheel.name]
        if wheel.is_symlink() or wheel.stat().st_size!=spec['bytes'] or hashlib.sha256(wheel.read_bytes()).hexdigest()!=spec['sha256']:
            raise RuntimeError('Offline wheel SHA256 mismatch: '+wheel.name)
        dist=wheel.name.split('-')[0]
        if dist in ['rsl_rl_lib','argcomplete']:
            package,version=('rsl-rl-lib','3.1.2') if dist=='rsl_rl_lib' else ('argcomplete','3.6.2')
            try:present=importlib.metadata.version(package)==version
            except importlib.metadata.PackageNotFoundError:present=False
        else:
            module=mapping.get(dist)
            if module is None:raise RuntimeError('Unapproved offline dependency '+dist)
            present=importlib.util.find_spec(module) is not None
        if not present:wheels.append(str(wheel))
    if wheels:
        selected=ROOT/'.flux-selected-wheels.lock'
        selected.write_text(''.join(approved[Path(w).name]['distribution']+' @ '+Path(w).resolve().as_uri()+' --hash=sha256:'+approved[Path(w).name]['sha256']+'\n' for w in wheels))
        result=subprocess.run([sys.executable,'-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-cache-dir','--no-deps','--require-hashes','--only-binary=:all:','--target',str(deps),'-r',str(selected)],capture_output=True,text=True,timeout=120)
        if result.returncode:raise RuntimeError('Offline dependency install failed: '+result.stderr[-1500:])
    sys.path.insert(0,str(deps))
    os.environ['PYTHONPATH']=str(deps)+os.pathsep+os.environ.get('PYTHONPATH','')
    required=['torch','tensordict','numpy','gymnasium','hydra','argcomplete','prettytable','tensorboard','git','onnx','isaaclab','isaaclab_rl']
    missing=[module for module in required if importlib.util.find_spec(module) is None]
    event('dependency_import_audit',required=required,missing=missing,offline_wheels=len(actual_wheels))
    if missing:raise RuntimeError('Missing required imports after offline installation: '+','.join(missing))
    if importlib.metadata.version('rsl-rl-lib')!='3.1.2':raise RuntimeError('RSL-RL version must be 3.1.2')
    if importlib.metadata.version('argcomplete')!='3.6.2':raise RuntimeError('Argcomplete version must be 3.6.2')
    import torch
    if not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Expected image Torch2.7.0 with CUDA; no CPU fallback')
    x=torch.ones(16,device='cuda:0');assert (x+x).sum().item()==32;torch.cuda.synchronize()
    event('runtime_preflight_verified',python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,rsl_rl='3.1.2',offline_installed=len(wheels))
