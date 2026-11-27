"""Bounded private input transport; no credentials and no URL logging."""
import hashlib,io,json,os,time,urllib.request,zipfile
from pathlib import Path
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
def ensure_checkpoint():
    spec=json.loads((ROOT/'flux/hold_trial.json').read_text());dest=ROOT/'inputs/normal_model_23000.pt'
    assert not dest.exists(),'No stale checkpoint reuse'
    url=os.environ.pop('NORMAL23000_PRIVATE_INPUT_URL','');parts=urlsplit(url)
    assert parts.scheme=='https' and parts.hostname=='download.wetransfer.com' and not parts.username and not parts.password
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    raw=bytearray();deadline=time.monotonic()+90
    try:
        with urllib.request.build_opener(NoRedirect).open(url,timeout=25) as response:
            assert response.status==200
            while True:
                assert time.monotonic()<deadline
                block=response.read(256*1024)
                if not block:break
                raw.extend(block);assert len(raw)<=7909192
    except Exception as exc:
        raise RuntimeError('Private checkpoint input download failed: '+type(exc).__name__+'; URL suppressed') from None
    assert len(raw)==7909192 and hashlib.sha256(raw).hexdigest()=='e735f28035f81aa2870ff508306acb13bb6fc88884fd86fc98d517b6c6c3e68b'
    name='3332/logs/rsl_rl/limx_hu_d04_01_flat_beyondmimic/2026-06-02_12-19-08/model_23000.pt'
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert len(z.infolist())<100 and z.getinfo(name).file_size==6724543
        checkpoint=z.read(name)
    assert hashlib.sha256(checkpoint).hexdigest()==spec['checkpoint_sha256']
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(checkpoint)
    dest.chmod(0o600)
    print(json.dumps({'event':'normal23000_private_checkpoint_delivered','bytes':len(checkpoint),'sha256':spec['checkpoint_sha256'],'archive_sha256_verified':True,'public_weights':False}),flush=True)
    return dest