"""Small default-branch launcher. Downloads verified payload; CPU-only unless --train."""
import argparse,base64,concurrent.futures,hashlib,json,os,runpy,signal,stat,subprocess,sys,time,urllib.error,urllib.request,zipfile
from pathlib import Path,PurePosixPath

MAX_ARCHIVE=90*1024**2
def sha(raw):return hashlib.sha256(raw).hexdigest()
def event(event_type,**kw):print(json.dumps({'event':event_type,**kw}),flush=True)
def check_descriptor(d):
    assert d['format']=='hud04_delivery_v1' and d['repo']=='zhu7136/liedit1124'
    assert 0<d['archive_bytes']<=MAX_ARCHIVE and len(d['archive_sha256'])==64
    assert 0<len(d['chunks'])<=40
    assert sum(c['bytes'] for c in d['chunks'])==d['archive_bytes']
    for i,c in enumerate(d['chunks']):
        assert c['index']==i and 0<c['bytes']<=3*1024**2
        assert len(c['sha256'])==64 and len(c['git_blob'])==40
        assert all(v in '0123456789abcdef' for v in c['sha256']+c['git_blob'])

def fetch_chunk(repo,c,dest,deadline):
    target=dest/('part-%03d'%c['index'])
    if target.exists():
        raw=target.read_bytes()
        if len(raw)==c['bytes'] and sha(raw)==c['sha256']:return target
        raise RuntimeError('Cached part is corrupt; no automatic overwrite')
    url='https://api.github.com/repos/'+repo+'/git/blobs/'+c['git_blob']
    for attempt in range(2):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Source download deadline')
        try:
            request=urllib.request.Request(url,headers={'Accept':'application/vnd.github.raw+json','User-Agent':'hud04-source-delivery-v1'})
            with urllib.request.urlopen(request,timeout=min(30,remaining)) as response:
                assert response.geturl()==url,'No cross-domain redirects allowed'
                data=response.read(5*1024**2+1);assert len(data)<=5*1024**2
                if 'json' in response.headers.get('Content-Type',''):
                    body=json.loads(data);assert body['encoding']=='base64'
                    raw=base64.b64decode(body['content'].replace('\n',''),validate=True)
                else:raw=data
            break
        except urllib.error.HTTPError as exc:
            if exc.code<500 or attempt:raise
            time.sleep(1)
        except (urllib.error.URLError,TimeoutError):
            if attempt:raise
            time.sleep(1)
    assert len(raw)==c['bytes'] and sha(raw)==c['sha256'],'Part checksum mismatch'
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==c['git_blob']
    with target.open('xb') as f:f.write(raw)
    event('source_chunk_verified',index=c['index'],bytes=len(raw))
    return target

def extract_verified(archive,descriptor,destination):
    assert not destination.exists(),'Never overwrite an existing workload'
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        assert len(names)==descriptor['file_count'] and len(names)==len(set(names)) and len(names)<256
        assert sum(i.file_size for i in infos)<=100*1024**2
        for i in infos:
            p=PurePosixPath(i.filename);mode=i.external_attr>>16
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
            assert str(p)==i.filename and not i.is_dir() and not stat.S_ISLNK(mode)
            assert i.file_size<=30*1024**2
        manifest_raw=z.read('flux/source_manifest.json')
        assert sha(manifest_raw)==descriptor['source_manifest_sha256']
        manifest=json.loads(manifest_raw)
        assert set(names)==set(manifest)|{'flux/source_manifest.json'}
        for n,wanted in manifest.items():assert sha(z.read(n))==wanted,'Internal file checksum: '+n
        # Every entry and hash is validated before any file is extracted.
        destination.mkdir()
        for i in infos:
            path=destination/i.filename;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:f.write(z.read(i))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--expected-loader-commit')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-only',action='store_true');mode.add_argument('--train',action='store_true')
    p.add_argument('--work-dir',type=Path)
    args=p.parse_args();started=time.monotonic()
    def cutoff(*unused):
        event('source_delivery_hard_timeout');os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(3600 if args.train else 240)
    root=Path(__file__).resolve().parent
    if args.train:
        assert args.expected_loader_commit and not args.work_dir,'Training requires pinned loader commit and default workload path'
        actual=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        assert actual==args.expected_loader_commit,'Loader commit mismatch; refusing training'
        subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--','launch.py','source.json'],check=True,capture_output=True,timeout=10)
    raw=(root/'source.json').read_bytes();assert sha(raw)==args.expected_manifest_sha256,'Delivery descriptor mismatch'
    descriptor=json.loads(raw);check_descriptor(descriptor)
    work=(args.work_dir or root/'verified-source').resolve();work.mkdir(parents=True,exist_ok=True)
    assert not (work/'workload').exists(),'Do not restart/overwrite an existing workload'
    event('source_delivery_begin',verify_only=args.verify_only,archive_bytes=descriptor['archive_bytes'],chunks=len(descriptor['chunks']))
    parts=work/'parts';parts.mkdir(exist_ok=True);deadline=time.monotonic()+180
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fetch_chunk,descriptor['repo'],c,parts,deadline) for c in descriptor['chunks']]
        paths=[future.result() for future in futures]
    archive=work/'source.zip'
    with archive.open('xb') as out:
        for path in paths:out.write(path.read_bytes())
    contents=archive.read_bytes();assert len(contents)==descriptor['archive_bytes'] and sha(contents)==descriptor['archive_sha256'],'Full archive checksum mismatch'
    destination=work/'workload';extract_verified(archive,descriptor,destination)
    event('source_delivery_verified',archive_sha256=sha(contents),files=descriptor['file_count'],seconds=round(time.monotonic()-started,3),training_started=False)
    if args.verify_only:
        signal.alarm(0);return
    os.environ['DELIVERY_STARTED_MONOTONIC']=str(started)
    os.environ['DELIVERY_MANIFEST_SHA256']=args.expected_manifest_sha256
    entry=destination/'flux/bootstrap.py';assert entry.is_file()
    # Stay in the SDK's gm-run process so its output-collection/finalizers survive.
    # The already active alarm covers this transition; bootstrap sets remaining time.
    sys.argv=[str(entry),'--commit',args.expected_loader_commit]
    runpy.run_path(str(entry),run_name='__main__')
if __name__=='__main__':main()