"""The sole experimental axis is a consistent reference asset, never actor weights."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
def configure_reference(cfg):
    manifest=json.loads((HERE/'manifest.json').read_text());variant=os.environ['REPLAY_REFERENCE_VARIANT'];assert variant in ['original','contact']
    if variant=='contact':cfg.commands.motion.motion_file=str(HERE/'contact_reference_diagnostic_only.npz')
    path=Path(cfg.commands.motion.motion_file);digest=hashlib.sha256(path.read_bytes()).hexdigest();expected=manifest['reference_sha256'][variant];assert digest==expected
    os.environ['REPLAY_REFERENCE_SHA']=expected
    print(json.dumps(dict(event='reference_asset_verified',reference_variant=variant,sha256=digest,diagnostic_prefix_limit_frame=200,all_other_physics_unchanged=True,training_updates=0)),flush=True)
