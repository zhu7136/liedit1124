"""Additive native-resume kneel delivery; does not run historical root launch or checkpoints."""
import argparse,hashlib,io,json,os,runpy,signal,stat,subprocess,sys,time,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];TRIAL=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def event(kind,**kw):print(json.dumps({'event':kind,**kw}),flush=True)
def restore(destination,expected_sha):
    raw=(TRIAL/'delivery.json').read_bytes();assert sha(raw)==expected_sha,'Delivery descriptor mismatch'
    d=json.loads(raw);assert d['format']=='kneel_git_shards_v1'
    assert not destination.exists(),'Never overwrite an existing workload'
    assert 0<len(d['chunks'])<=40 and d['source_archive_bytes']<=90*1024**2
    data=[]
    for index,c in enumerate(d['chunks']):
        assert c['index']==index and c['file']=='source.part-%03d'%index
        path=ROOT/'source-parts'/c['file'];assert not path.is_symlink()
        assert path.stat().st_size==c['bytes']<=3*1024**2
        part=path.read_bytes();assert sha(part)==c['sha256']
        assert hashlib.sha1(b'blob '+str(len(part)).encode()+b'\0'+part).hexdigest()==c['git_blob']
        data.append(part)
    payload=b''.join(data);assert len(payload)==d['source_archive_bytes'] and sha(payload)==d['source_archive_sha256']
    all_files={}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=z.namelist();assert len(names)==len(set(names))<256
        assert sum(i.file_size for i in z.infolist())<100*1024**2
        for name,wanted in d['base_source_files'].items():
            p=PurePosixPath(name);info=z.getinfo(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and not stat.S_ISLNK(info.external_attr>>16)
            content=z.read(name);assert sha(content)==wanted;all_files[name]=content
    overlay=TRIAL/'overlay';paths=list(overlay.rglob('*'));assert not any(p.is_symlink() for p in paths)
    assert {p.relative_to(overlay).as_posix() for p in paths if p.is_file()}==set(d['overlay_files'])
    for name,wanted in d['overlay_files'].items():
        p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
        content=(overlay/name).read_bytes();assert sha(content)==wanted;all_files[name]=content
    manifest_raw=all_files['flux/source_manifest.json'];assert sha(manifest_raw)==d['final_manifest_sha256']
    manifest=json.loads(manifest_raw);assert set(all_files)==set(manifest)|{'flux/source_manifest.json'}
    for name,wanted in manifest.items():assert sha(all_files[name])==wanted,'Final source mismatch: '+name
    assert manifest['motions/kneel_candidate/motion.npz']==d['motion_sha256']
    assert not any('resume_checkpoint' in n or 'toe_training_hooks' in n or 'hold_support' in n for n in all_files)
    for name,content in sorted(all_files.items()):
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(content)
    event('source_delivery_verified',files=len(all_files),chunks=len(d['chunks']),archive_sha256=sha(payload),motion_sha256=d['motion_sha256'],fresh_training=False,historical_entrypoints_included=False,worker_http_downloads=True,checkpoint_transport="explicit_private_HTTPS_with_SHA256")
    return d
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-manifest-sha256',required=True);parser.add_argument('--expected-commit');parser.add_argument('--checkpoint-url');parser.add_argument('--work-dir',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-only',action='store_true');mode.add_argument('--train',action='store_true');args=parser.parse_args();started=time.monotonic()
    def cutoff(*unused):event('source_delivery_deadline');os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(2400 if args.train else 120)
    if args.train:
        assert args.expected_commit and not args.work_dir and args.checkpoint_url
        os.environ['KNEEL_PRIVATE_CHECKPOINT_URL']=args.checkpoint_url
        args.checkpoint_url=None
        sys.argv=[sys.argv[0]]
        actual=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip();assert actual==args.expected_commit,'Worker Git revision mismatch'
        subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','HEAD','--','kneel_20261126','source-parts'],capture_output=True,check=True,timeout=15)
    destination=(args.work_dir or TRIAL/'verified-workload').resolve();d=restore(destination,args.expected_manifest_sha256)
    if args.verify_only:signal.alarm(0);return
    assert d['worker_seconds']==2400 and d['automatic_retries']==0
    os.environ.update(KNEEL_DELIVERY_STARTED=str(started),KNEEL_DELIVERY_SHA256=args.expected_manifest_sha256,KNEEL_COMMIT=args.expected_commit)
    entry=destination/'flux/kneel_worker.py';sys.argv=[str(entry)];runpy.run_path(str(entry),run_name='__main__')
if __name__=='__main__':main()