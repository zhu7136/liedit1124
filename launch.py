"""Pinned task418 continuation overlay; local shards and Flux checkpoint mount only."""
import argparse,hashlib,json,os,runpy,signal,stat,subprocess,time,zipfile
import sys
from pathlib import Path,PurePosixPath

MAX_ARCHIVE=90*1024**2
OVERLAY_PATHS={'flux/bootstrap.py','flux/dependencies.json','flux/source_manifest.json','flux/offline-wheel-manifest.json','flux/offline-requirements.lock','flux/wheelhouse/argcomplete-3.6.2-py3-none-any.whl','scripts/rsl_rl/train.py','flux/toe_training_hooks.py'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def event(event_type,**kw):print(json.dumps({'event':event_type,**kw}),flush=True)
def check_descriptor(d):
    assert d['format']=='hud04_delivery_v1' and d['repo']=='zhu7136/liedit1124'
    assert 0<d['archive_bytes']<=MAX_ARCHIVE and len(d['archive_sha256'])==64
    assert 0<len(d['chunks'])<=40
    assert sum(c['bytes'] for c in d['chunks'])==d['archive_bytes']
    for i,c in enumerate(d['chunks']):
        assert c['index']==i and c['file']=='source.part-%03d'%i and 0<c['bytes']<=3*1024**2
        assert len(c['sha256'])==64 and len(c['git_blob'])==40
        assert all(v in '0123456789abcdef' for v in c['sha256']+c['git_blob'])
    overlay=d['overlay'];assert overlay['format']=='hud04_offline_dependency_overlay_v1'
    assert len(overlay['files'])==len(OVERLAY_PATHS) and {e['path'] for e in overlay['files']}==OVERLAY_PATHS
    assert sum(e['bytes'] for e in overlay['files'])<300000
def local_parts(root,descriptor):
    directory=root/'source-parts'
    assert directory.is_dir() and not directory.is_symlink(),'Missing local Git shards; no network fallback'
    assert {p.name for p in directory.iterdir()}=={c['file'] for c in descriptor['chunks']},'Missing/extra local Git shards'
    paths=[]
    for c in descriptor['chunks']:
        path=directory/c['file'];assert path.is_file() and not path.is_symlink() and path.stat().st_size==c['bytes'],'Invalid local Git shard'
        with path.open('rb') as f:raw=f.read(c['bytes']+1)
        assert not raw.startswith(b'version https://git-lfs.github.com/spec/'),'Git LFS pointers are not payload'
        assert len(raw)==c['bytes'] and sha(raw)==c['sha256'],'Local shard SHA256 mismatch'
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==c['git_blob']
        event('source_chunk_verified',index=c['index'],bytes=len(raw),delivery='local_git_shard');paths.append(path)
    return paths
def extract_verified(archive,descriptor,destination):
    assert not destination.exists(),'Never overwrite an existing workload'
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        assert len(names)==descriptor['file_count'] and len(names)==len(set(names)) and len(names)<256
        assert sum(i.file_size for i in infos)<=100*1024**2
        for i in infos:
            p=PurePosixPath(i.filename);mode=i.external_attr>>16
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
            assert str(p)==i.filename and not i.is_dir() and not stat.S_ISLNK(mode) and i.file_size<=30*1024**2
        manifest_raw=z.read('flux/source_manifest.json');assert sha(manifest_raw)==descriptor['source_manifest_sha256']
        manifest=json.loads(manifest_raw);assert set(names)==set(manifest)|{'flux/source_manifest.json'}
        for n,wanted in manifest.items():assert sha(z.read(n))==wanted,'Internal file checksum: '+n
        destination.mkdir()
        for i in infos:
            path=destination/i.filename;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(z.read(i))
def apply_overlay(root,descriptor,destination):
    overlay=descriptor['overlay'];directory=root/'source-overlay'
    assert directory.is_dir() and not directory.is_symlink(),'Missing dependency overlay'
    all_paths=list(directory.rglob('*'));assert not any(p.is_symlink() for p in all_paths),'Overlay symlinks forbidden'
    assert {p.relative_to(directory).as_posix() for p in all_paths if p.is_file()}==OVERLAY_PATHS,'Missing/extra overlay files'
    payload={}
    for e in overlay['files']:
        p=directory/e['path'];assert p.stat().st_size==e['bytes']
        raw=p.read_bytes();assert sha(raw)==e['sha256'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['git_blob'],'Overlay hash mismatch'
        payload[e['path']]=raw
    manifest_raw=payload['flux/source_manifest.json'];assert sha(manifest_raw)==overlay['final_source_manifest_sha256']
    manifest=json.loads(manifest_raw)
    assert set(manifest)|{'flux/source_manifest.json'}=={p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}|set(payload)
    assert len(manifest)+1==overlay['final_file_count']
    # Verify every final byte BEFORE replacing files in this fresh extraction.
    for name,wanted in manifest.items():
        raw=payload[name] if name in payload else (destination/name).read_bytes()
        assert sha(raw)==wanted,'Final overlay source hash mismatch: '+name
    assert manifest['motions/gui00528900_knee_toe_upright_candidate/motion.npz']==descriptor['motion_sha256']
    for name,raw in payload.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    event('dependency_overlay_verified',files=len(payload),final_source_files=overlay['final_file_count'],worker_http_downloads=False)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-manifest-sha256',required=True);p.add_argument('--expected-loader-commit')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-only',action='store_true');mode.add_argument('--train',action='store_true')
    p.add_argument('--work-dir',type=Path);args=p.parse_args();started=time.monotonic()
    def cutoff(*unused):event('source_delivery_hard_timeout');os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(1800 if args.train else 120)
    root=Path(__file__).resolve().parent
    if args.train:
        assert args.expected_loader_commit and not args.work_dir,'Training requires pinned loader commit and default workload path'
        actual=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip();assert actual==args.expected_loader_commit,'Loader commit mismatch'
        subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--','launch.py','source.json','source-parts','source-overlay'],check=True,capture_output=True,timeout=15)
    raw=(root/'source.json').read_bytes();assert sha(raw)==args.expected_manifest_sha256,'Delivery descriptor mismatch'
    descriptor=json.loads(raw);check_descriptor(descriptor)
    event('source_delivery_begin',verify_only=args.verify_only,chunks=len(descriptor['chunks']),delivery='local_git_shards_with_dependency_overlay',worker_http_downloads=False)
    paths=local_parts(root,descriptor);work=(args.work_dir or root/'verified-source').resolve();work.mkdir(parents=True,exist_ok=True)
    assert not (work/'workload').exists(),'Do not restart/overwrite an existing workload'
    archive=work/'source.zip'
    with archive.open('xb') as out:
        for path in paths:out.write(path.read_bytes())
    contents=archive.read_bytes();assert len(contents)==descriptor['archive_bytes'] and sha(contents)==descriptor['archive_sha256'],'Full archive checksum mismatch'
    destination=work/'workload';extract_verified(archive,descriptor,destination);apply_overlay(root,descriptor,destination)
    event('source_delivery_verified',archive_sha256=sha(contents),files=descriptor['overlay']['final_file_count'],seconds=round(time.monotonic()-started,3),training_started=False,delivery='local_git_shards_with_dependency_overlay')
    if args.verify_only:signal.alarm(0);return
    os.environ['DELIVERY_STARTED_MONOTONIC']=str(started);os.environ['DELIVERY_MANIFEST_SHA256']=args.expected_manifest_sha256
    os.environ['TOE_RESUME_MOUNT']='/workspace/flux_resume'
    entry=destination/'flux/bootstrap.py';assert entry.is_file();sys.argv=[str(entry),'--commit',args.expected_loader_commit]
    runpy.run_path(str(entry),run_name='__main__')
if __name__=='__main__':main()