"""Final-hold-only action-rate penalty and strict normal23000 warm start."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
HOLD_START_FRAME=434
SOURCE_ITERATION=999

def hold_action_rate_l2(env, command_name='motion', start_frame=HOLD_START_FRAME):
    command=env.command_manager.get_term(command_name)
    gate=(command.time_steps>=start_frame)&(command.joint_vel.abs().amax(dim=-1)<1e-6)
    penalty=(env.action_manager.action-env.action_manager.prev_action).square().sum(dim=-1)
    return gate.to(penalty.dtype)*penalty

def configure_hold_trial(env_cfg,agent_cfg):
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from balanced_support_rewards import force_balance,four_support,equal_load,force_smooth,hold_motion,hold_action_change
    env_cfg.episode_length_s=30.0
    env_cfg.scene.robot.spawn.usd_path=str(ROOT/"source/limx_rl_lab/data/Robots/limx/HU_D04_description/usd/HU_D04_01_BalancedContacts.usda")
    env_cfg.scene.robot.actuators['passive_knee_wheels']=ImplicitActuatorCfg(joint_names_expr=['left_knee_wheel_joint','right_knee_wheel_joint'],effort_limit_sim=0.0,velocity_limit_sim=1000.0,stiffness=0.0,damping=0.0,armature=0.0)
    env_cfg.scene.robot.init_state.joint_pos['.*_knee_wheel_joint']=0.0
    env_cfg.rewards.hold_force_balance=RewardTermCfg(func=force_balance,weight=-4.0)
    env_cfg.rewards.hold_four_support=RewardTermCfg(func=four_support,weight=2.0)
    env_cfg.rewards.hold_equal_load=RewardTermCfg(func=equal_load,weight=-4.0)
    env_cfg.rewards.hold_force_smooth=RewardTermCfg(func=force_smooth,weight=-0.1)
    env_cfg.rewards.hold_motion=RewardTermCfg(func=hold_motion,weight=-0.5)
    env_cfg.rewards.final_hold_action_rate_l2=RewardTermCfg(func=hold_action_change,weight=-0.05)
    env_cfg.rewards.undesired_contacts.params['sensor_cfg']=SceneEntityCfg('contact_forces',body_names=[r'^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!zuolun_link$)(?!youlun_link$)(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)(?!left_hand_contact$)(?!right_hand_contact$).+$'])
    # First establish support under paired contact properties; retain observation noise,
    # disable random pushes and COM/material asymmetry for this first feasibility trial.
    env_cfg.events.physics_material=None;env_cfg.events.base_com=None;env_cfg.events.push_robot=None
    env_cfg.commands.motion.pose_range={};env_cfg.commands.motion.velocity_range={};env_cfg.commands.motion.joint_position_range=(0.0,0.0)
    agent_cfg.policy.actor_obs_normalization=False;agent_cfg.policy.critic_obs_normalization=False
    agent_cfg.algorithm.learning_rate=0.0001
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