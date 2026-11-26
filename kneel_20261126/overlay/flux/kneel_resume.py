"""Fail-closed native RSL-RL resume of verified Flux512 NoSE154 checkpoint."""
import hashlib,json
from pathlib import Path
SOURCE_TASK='TASK_20261126_512'
SOURCE_SHA='234f686f8bad05b814921d2e60d5d8659ffa4c7ee7082f359d3de0d976a6f3fb'
SOURCE_BYTES=6724913
SOURCE_ITER=2021
SOURCE_UPDATES=2022
MOUNT=Path('/workspace/flux_resume512')
def verified_checkpoint_path():
    paths=[MOUNT] if MOUNT.is_file() else sorted(MOUNT.glob('*.pt')) if MOUNT.is_dir() else []
    assert len(paths)<=8,'Unexpected checkpoint mount layout'
    matches=[p for p in paths if not p.is_symlink() and p.stat().st_size==SOURCE_BYTES and hashlib.sha256(p.read_bytes()).hexdigest()==SOURCE_SHA]
    assert len(matches)==1,'Required Flux512 final checkpoint not mounted or SHA mismatch; no fresh fallback'
    return matches[0]
def verify_restored_runner(runner):
    import torch
    from kneel_hooks import event
    path=verified_checkpoint_path()
    saved=torch.load(str(path),map_location='cpu',weights_only=True)
    assert saved['iter']==SOURCE_ITER and runner.current_learning_iteration==SOURCE_ITER,'Resume iteration not restored'
    model=saved['model_state_dict'];actual=runner.alg.policy.state_dict()
    assert set(model)==set(actual) and all(torch.equal(actual[k].detach().cpu(),v.cpu()) for k,v in model.items()),'Actor/critic/normalizer restoration mismatch'
    assert any(k.startswith('actor.') and v.ndim==2 and v.shape[1]==154 for k,v in model.items()),'Not a154-input actor checkpoint'
    expected=saved['optimizer_state_dict'];got=runner.alg.optimizer.state_dict()
    assert expected['state'] and set(expected['state'])==set(got['state']),'Optimizer state not restored'
    assert expected['param_groups']==got['param_groups'],'Optimizer parameter groups mismatch'
    for key,row in expected['state'].items():
        assert set(row)==set(got['state'][key])
        for name,value in row.items():
            restored=got['state'][key][name]
            assert torch.equal(value.cpu(),restored.cpu()) if isinstance(value,torch.Tensor) else value==restored,'Optimizer tensor mismatch'
    rates={float(g['lr']) for g in got['param_groups']};assert len(rates)==1
    if hasattr(runner.alg,'learning_rate'):runner.alg.learning_rate=next(iter(rates))
    record={'source_task':SOURCE_TASK,'source_checkpoint_sha256':SOURCE_SHA,'source_iteration':SOURCE_ITER,'source_completed_updates':SOURCE_UPDATES,'actor_critic_weights_verified':True,'optimizer_state_verified':True,'normalizer_buffers_verified_with_model_state':True,'learning_rate':next(iter(rates)),'environment_state_restored':False,'rng_state_restored':False,'fresh_training':False}
    event('checkpoint_loaded',**record)
    return record