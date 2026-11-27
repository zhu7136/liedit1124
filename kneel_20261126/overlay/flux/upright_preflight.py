"""Frozen-policy native-physics compatibility check, not a quality acceptance test."""
import json,time
import torch

def native_preflight(runner,env,event,out):
    raw=env.unwrapped;robot=raw.scene['robot'];cmd=raw.command_manager.get_term('motion')
    names=list(robot.body_names)
    assert 'zuolun_link' not in names and 'youlun_link' not in names, 'Unexpected MuJoCo-only knee wheels in native training asset'
    assert cmd.motion.joint_vel[434:].abs().max()<1e-6
    assert cmd.motion.joint_vel[335:434].abs().max()>1e-4
    assert cmd.motion._body_ang_vel_w[434:].abs().max()<1e-6
    def actor_obs(value):
        if isinstance(value,tuple):value=value[0]
        if not torch.is_tensor(value):value=value['policy']
        assert value.shape==(4096,154) and torch.isfinite(value).all()
        return value
    start=time.monotonic();obs=actor_obs(env.get_observations());done_events=0;contact_peak=0.
    event('upright_native_preflight_begin',physics_steps=100,optimizer_updates=0,quality_acceptance=False)
    with torch.inference_mode():
        for step in range(100):
            action=runner.alg.policy.actor(obs)
            assert action.shape==(4096,29) and torch.isfinite(action).all()
            result=env.step(action);obs=actor_obs(result[0]);done_events+=int(result[2].sum().item())
            assert torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all()
            assert torch.isfinite(robot.data.joint_vel).all()
            forces=raw.scene.sensors['contact_forces'].data.net_forces_w
            assert torch.isfinite(forces).all()
            contact_peak=max(contact_peak,float(forces.norm(dim=-1).max().item()))
    assert len(runner.alg.optimizer.state)==0, 'Preflight must not optimize'
    env.reset()
    result={'passed':True,'physics_steps':100,'simulated_seconds_per_environment':float(raw.step_dt*100),'num_envs':4096,'done_events':done_events,'peak_contact_force_N':contact_peak,'seconds':round(time.monotonic()-start,3),'native_knee_wheels_absent':True,'finite_state_and_contact_forces':True,'optimizer_updates':0,'quality_acceptance':False,'dynamics_stability_validated':False,'note':'Frozen baseline with normal training resets/randomization. Proves runtime compatibility only; it does not prove stable upright holding.'}
    (out/'native_preflight.json').write_text(json.dumps(result,indent=2))
    event('upright_native_preflight_passed',**result)
