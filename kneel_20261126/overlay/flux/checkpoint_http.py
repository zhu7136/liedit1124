"""Explicit bounded Flux-owned HTTPS checkpoint delivery; URL never logged or archived."""
import hashlib,os,time,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
SOURCE_SHA='234f686f8bad05b814921d2e60d5d8659ffa4c7ee7082f359d3de0d976a6f3fb'
SOURCE_BYTES=6724913
DEST=Path(__file__).resolve().parents[1]/'inputs/flux512/final_checkpoint.pt'
def ensure_checkpoint():
    if DEST.is_file():
        assert DEST.stat().st_size==SOURCE_BYTES and hashlib.sha256(DEST.read_bytes()).hexdigest()==SOURCE_SHA
        return DEST
    url=os.environ.pop('KNEEL_PRIVATE_CHECKPOINT_URL','');parsed=urlsplit(url)
    assert parsed.scheme=='https' and parsed.hostname=='limx-gradmotion.oss-cn-beijing.aliyuncs.com' and not parsed.username and not parsed.password,'Missing or unapproved checkpoint URL'
    assert parsed.path=='/upload%2F2026%2F11%2F26%2Ffinal_checkpoint_20261126190201A793.pt','Wrong checkpoint object'
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*a,**kw):return None
    deadline=time.monotonic()+60;raw=bytearray()
    try:
        with urllib.request.build_opener(NoRedirect).open(url,timeout=15) as response:
            assert response.status==200
            while True:
                assert time.monotonic()<deadline,'Checkpoint transfer deadline'
                block=response.read(256*1024)
                if not block:break
                raw.extend(block);assert len(raw)<=SOURCE_BYTES,'Oversized checkpoint'
    except Exception as exc:
        raise RuntimeError('Checkpoint download failed: '+type(exc).__name__+'; URL intentionally suppressed') from None
    assert len(raw)==SOURCE_BYTES and hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'Downloaded checkpoint SHA/size mismatch'
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with DEST.open('xb') as f:f.write(raw)
    DEST.chmod(0o600)
    print('{"event":"checkpoint_delivery_verified","source_task":"TASK_20261126_512","bytes":6724913,"sha256":"'+SOURCE_SHA+'","delivery":"explicit HTTPS, not assumed native mount"}',flush=True)
    return DEST