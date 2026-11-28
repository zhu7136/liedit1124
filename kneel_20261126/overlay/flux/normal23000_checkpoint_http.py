"""Private expiring TASK005 input; no secrets, redirects, URL logging, or public weights."""
import hashlib,json,os,time,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[1]
def ensure_checkpoint():
    spec=json.loads((ROOT/'flux/hold_trial.json').read_text());dest=ROOT/'inputs/normal_model_23000.pt'
    assert not dest.exists()
    url=os.environ.pop('NORMAL23000_PRIVATE_INPUT_URL','');p=urlsplit(url)
    assert p.scheme=='https' and p.hostname in ['limx-gradmotion.oss-cn-beijing.aliyuncs.com','argo-prod.oss-cn-beijing.aliyuncs.com'] and not p.username and not p.password
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
                raw.extend(block);assert len(raw)<=6730609
    except Exception as exc:raise RuntimeError('Private TASK005 checkpoint input failed: '+type(exc).__name__+'; URL suppressed') from None
    assert len(raw)==6730609 and hashlib.sha256(raw).hexdigest()==spec['checkpoint_sha256']
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(raw)
    dest.chmod(0o600)
    print(json.dumps({'event':'TASK005_private_checkpoint_delivered','bytes':len(raw),'sha256':spec['checkpoint_sha256'],'public_weights':False}),flush=True)
    return dest