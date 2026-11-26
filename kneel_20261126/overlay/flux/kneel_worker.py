"""One 4090D, resumed No-State-Estimation BeyondMimic; no credential forwarding."""
import argparse,hashlib,importlib.util,json,os,runpy,signal,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/kneel_training'
WORKER_SECONDS=2400;SOFT_SECONDS=2220;CHILD_SECONDS=2340;STARTUP_SECONDS=600
def event(kind,**kw):print(json.dumps({'event':kind,**kw}),flush=True)
def kill_child(child):
    if child is not None and child.poll() is None:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=5)
def worker(started):
    os.environ.update(KNEEL_STARTED=str(started),KNEEL_SOFT_DEADLINE=str(started+SOFT_SECONDS),CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    sys.path[:0]=[str(ROOT/'source/limx_rl_lab'),str(ROOT/'flux'),str(ROOT/'scripts/rsl_rl')]
    if importlib.util.find_spec('isaaclab') is None:
        for base in [Path('/workspace/isaaclab/source'),Path('/isaaclab/source'),Path('/opt/isaaclab/source')]:
            if (base/'isaaclab/isaaclab').is_dir():
                sys.path[:0]=[str(base/pkg) for pkg in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']];break
    manifest=json.loads((ROOT/'flux/source_manifest.json').read_text())
    for name,wanted in manifest.items():
        path=ROOT/name
        assert path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==wanted,'Source checksum: '+name
    from offline_dependencies import prepare_dependencies
    prepare_dependencies();os.chdir(ROOT)
    sys.argv=[str(ROOT/'scripts/rsl_rl/train.py'),'--task','LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation','--motion_file',str(ROOT/'motions/kneel_candidate/motion.npz'),'--num_envs','4096','--max_iterations','100000','--seed','42','--experiment_name','hud04_kneel_nose_20261126','--run_name','resume512_nose154_4096_40min','--resume','--logger','tensorboard','--headless','--device','cuda:0']
    runpy.run_path(str(ROOT/'scripts/rsl_rl/train.py'),run_name='__main__')
def archive_outputs(code,reason):
    record={'child_exit_code':code,'parent_stop_reason':reason,'worker_seconds':2400,'automatic_retries':0,'completion_present':(OUT/'completion.json').is_file(),'commit':os.environ['KNEEL_COMMIT'],'delivery_sha256':os.environ['KNEEL_DELIVERY_SHA256']}
    (OUT/'parent_result.json').write_text(json.dumps(record,indent=2))
    paths=[p for p in OUT.rglob('*') if p.is_file()]+[p for p in (ROOT/'logs').rglob('*') if p.is_file() and p.suffix not in ['.pt','.onnx','.pkl']]
    assert not any(p.is_symlink() for p in paths) and sum(p.stat().st_size for p in paths)<160*1024**2
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2));paths.append(OUT/'SHA256SUMS.json')
    target=ROOT/'outputs/kneel-training-output.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):z.write(path,str(path.relative_to(ROOT)))
    event('output_ready_for_flux_archive',bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),**record)
    return code==0 and record['completion_present']
def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-child',action='store_true');p.add_argument('--started',type=float);args=p.parse_args()
    if args.runtime_child:
        assert args.started is not None;worker(args.started);return
    started=float(os.environ['KNEEL_DELIVERY_STARTED']);child=None;position=0
    assert 0<=time.monotonic()-started<STARTUP_SECONDS
    def cutoff(*unused):
        kill_child(child);event('worker_hard_deadline',seconds=WORKER_SECONDS);os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(max(1,int(WORKER_SECONDS-(time.monotonic()-started))))
    try:
        assert not OUT.exists(),'No restart or output overwrite';OUT.mkdir(parents=True)
        launcher=next((str(x) for x in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')] if x.is_file() and os.access(x,os.X_OK)),None)
        if launcher is None:
            assert sys.version_info[:2]==(3,11),'No supported Isaac Python3.11 runtime';launcher=sys.executable
        event('worker_begin',worker_seconds=WORKER_SECONDS,soft_seconds=SOFT_SECONDS,startup_seconds=STARTUP_SECONDS,fresh_training=False,num_envs=4096,automatic_retries=0,commit=os.environ['KNEEL_COMMIT'],task='LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation')
        logpath=OUT/'native.log'
        with logpath.open('x') as log:
            child=subprocess.Popen([launcher,str(Path(__file__).resolve()),'--runtime-child','--started',str(started)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'),start_new_session=True)
            reason='child_exited'
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=CHILD_SECONDS:reason='child_hard_deadline';break
                with logpath.open('rb') as f:
                    f.seek(0,2);size=f.tell();f.seek(max(0,size-32768));tail=f.read().decode(errors='replace').lower()
                if any(marker in tail for marker in ['out of memory','outofmemoryerror','cuda_error_out_of_memory','gpu memory allocation failed']):reason='gpu_memory_failure';break
                events=OUT/'events.jsonl';initialized=False
                if events.exists():
                    text=events.read_text();initialized='environment_constructed' in text
                    if len(text)>position:print(text[position:],end='',flush=True);position=len(text)
                if elapsed>=STARTUP_SECONDS and not initialized:reason='startup_watchdog';break
                time.sleep(2)
            kill_child(child)
        code=child.returncode;events=OUT/'events.jsonl'
        if events.exists():print(events.read_text()[position:],end='',flush=True)
        event('child_exit',code=code,reason=reason)
        if code:
            with logpath.open('rb') as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-16000));print(f.read().decode(errors='replace'),flush=True)
        if not archive_outputs(code,reason):raise RuntimeError('No successful completed-update checkpoint; no automatic retry')
    finally:kill_child(child);signal.alarm(0)
if __name__=='__main__':main()