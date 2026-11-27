"""Final-hold-only action-rate penalty and strict normal23000 warm start."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
HOLD_START_FRAME=334
SOURCE_ITERATION=23000

def hold_action_rate_l2(env, command_name='motion', start_frame=HOLD_START_FRAME):
    command=env.command_manager.get_term(command_name)
    gate=(command.time_steps>=start_frame)&(command.joint_vel.abs().amax(dim=-1)<1e-6)
    penalty=(env.action_manager.action-env.action_manager.prev_action).square().sum(dim=-1)
    return gate.to(penalty.dtype)*penalty

def configure_hold_trial(env_cfg,agent_cfg):
    from isaaclab.managers import RewardTermCfg
    assert env_cfg.episode_length_s==10.0
    assert env_cfg.rewards.action_rate_l2.weight==-0.1
    env_cfg.rewards.final_hold_action_rate_l2=RewardTermCfg(func=hold_action_rate_l2,weight=-0.1,params={'command_name':'motion','start_frame':HOLD_START_FRAME})
    # The source checkpoint has no normalization buffers. Explicit False keeps
    # the same identity normalization; never create uninitialized normalizers.
    agent_cfg.policy.actor_obs_normalization=False
    agent_cfg.policy.critic_obs_normalization=False
    agent_cfg.save_interval=100

def strict_warm_start(runner,checkpoint_path):
    from kneel_hooks import event
    descriptor=json.loads((ROOT/'flux/hold_trial.json').read_text())
    p=Path(checkpoint_path)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==descriptor['checkpoint_sha256']
    checkpoint=torch.load(p,map_location='cpu',weights_only=False)
    assert checkpoint['iter']==SOURCE_ITERATION
    state=checkpoint['model_state_dict'];policy=runner.alg.policy
    current=policy.state_dict()
    assert set(current)==set(state),'No partial tensor initialization permitted'
    assert all(current[k].shape==v.shape and torch.isfinite(v).all() for k,v in state.items())
    assert state['actor.0.weight'].shape==(512,154)
    assert state['critic.0.weight'].shape==(512,286)
    assert state['std'].shape==(29,) and torch.all(state['std']>0)
    policy.load_state_dict(state,strict=True)
    assert all(torch.equal(policy.state_dict()[k].detach().cpu(),v) for k,v in state.items())
    assert len(runner.alg.optimizer.state)==0,'Changed reward: use a fresh optimizer'
    assert runner.current_learning_iteration==0
    # Check the actual GPU actor against a CPU reconstruction of the source.
    source=torch.nn.Sequential(torch.nn.Linear(154,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,29))
    source.load_state_dict({k[len('actor.'):]:v for k,v in state.items() if k.startswith('actor.')},strict=True)
    x=torch.linspace(-.5,.5,154).reshape(1,154)
    tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():
            expected=source(x);actual=policy.actor(x.to(next(policy.parameters()).device)).cpu()
    finally:torch.backends.cuda.matmul.allow_tf32=tf32
    assert torch.allclose(actual,expected,rtol=2e-4,atol=2e-5)
    runner._normal23000_warm_start_verified=True
    event('normal23000_warm_start_verified',source_iteration=SOURCE_ITERATION,checkpoint_sha256=descriptor['checkpoint_sha256'],actor_obs_dim=154,critic_obs_dim=286,actions=29,tensors_matched=len(state),all_tensors_exact=True,optimizer_reset=True,normalization='identity as source checkpoint',actor_output_max_abs_difference=float((actual-expected).abs().max()),fresh_training=False)