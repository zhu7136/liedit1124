"""One authorized, bounded Isaac Lab replay; never trains or retries."""
import argparse, ast, hashlib, importlib.util, json, os, signal, subprocess, sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE='325e3ba397440e6e183b69391b8966f3710cd078'
DESCRIPTOR='3740a371e5b29e1e3f61a7cfbc632541f98fa6dc53e243e8ad21a3b7b3dca9f9'
POLICY='8fe02e377239f879e59ff1c52a0b72ae07d082c6ed4f72ccde7de483d81f73a2'
def event(name,**kw): print(json.dumps(dict(event=name,**kw)),flush=True)
def program(work):
    original=(work/'scripts/rsl_rl/train.py').read_text()
    marker='    from kneel_hooks import audit_environment\n'
    assert original.count(marker)==1
    prefix=original.split(marker)[0]
    target='    env = gym.make('
    assert prefix.count(target)==1
    prefix=prefix.replace(target,
        '    env_cfg.observations.policy.enable_corruption = False\n'
        '    env_cfg.events.push_robot = None\n'
        '    env_cfg.events.physics_material = None\n'
        '    env_cfg.events.base_com = None\n'
        '    env_cfg.events.add_joint_default_pos = None\n'+target)
    text=prefix+'    from replay_hooks import replay_and_finish\n    replay_and_finish(env)\n    env.close()\n\nif __name__ == "__main__":\n    main()\n    simulation_app.close()\n'
    tree=ast.parse(text)
    tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='load_init_checkpoint')]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert not(isinstance(node.func,ast.Attribute) and node.func.attr in ['learn','save','backward','step','run_bounded_training'])
            assert not(isinstance(node.func,ast.Name) and node.func.id=='OnPolicyRunner')
    return tree
def child(work):
    sys.path[:0]=[str(work/'source/limx_rl_lab'),str(work/'flux'),str(work/'scripts/rsl_rl'),str(HERE)]
    if importlib.util.find_spec('isaaclab') is None:
        for base in [Path('/workspace/isaaclab/source'),Path('/isaaclab/source'),Path('/opt/isaaclab/source')]:
            if (base/'isaaclab/isaaclab').is_dir():
                sys.path[:0]=[str(base/p) for p in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']];break
    from offline_dependencies import prepare_dependencies
    prepare_dependencies();os.chdir(work)
    tree=program(work)
    sys.argv=[str(work/'scripts/rsl_rl/train.py'),'--task','LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation','--motion_file',str(work/'motions/kneel_candidate/motion.npz'),'--num_envs','2','--max_iterations','0','--seed','42','--experiment_name','hud04_kneel_nose_20261126','--run_name','replay_only_542','--logger','tensorboard','--headless','--device','cuda:0']
    event('replay_before_simulator',training_updates=0,num_envs=2,random_pushes=False,observation_noise=False)
    exec(compile(tree,str(work/'scripts/rsl_rl/train.py'),'exec'),{'__name__':'__main__','__file__':str(work/'scripts/rsl_rl/train.py')})
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--policy-url');p.add_argument('--child',action='store_true');p.add_argument('--work',type=Path);a=p.parse_args()
    os.environ.update(PYTHONUNBUFFERED='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    if a.child: child(a.work);return
    process=None
    def cutoff(*unused):
        if process is not None and process.poll() is None:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        event('replay_hard_timeout',entry_limit_seconds=240);os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(240)
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=10).strip();assert actual==a.expected_commit
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,h in manifest['entry_files'].items(): assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    assert manifest['source_commit']==SOURCE and manifest['training_updates']==0 and manifest['automatic_retries']==0
    assert a.policy_url and urlsplit(a.policy_url).scheme=='https' and urlsplit(a.policy_url).hostname=='limx-gradmotion.oss-cn-beijing.aliyuncs.com'
    with urlopen(a.policy_url,timeout=20) as response: data=response.read(2*1024**2+1)
    a.policy_url=None
    assert len(data)<=2*1024**2 and hashlib.sha256(data).hexdigest()==POLICY
    with (HERE/'input-policy.pt').open('xb') as f: f.write(data)
    event('replay_policy_verified',bytes=len(data),sha256=POLICY,transport='original_platform_private_https',public_git_policy=False)
    spec=importlib.util.spec_from_file_location('verified542',ROOT/'kneel_20261126/launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    work=HERE/'verified-workload';m.restore(work,DESCRIPTOR)
    launcher=next((str(x) for x in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')] if x.is_file() and os.access(x,os.X_OK)),sys.executable)
    try:
        process=subprocess.Popen([launcher,str(Path(__file__).resolve()),'--expected-commit',actual,'--child','--work',str(work)],start_new_session=True)
        code=process.wait(timeout=205)
        assert code==0,f'Replay child exited {code}'
        assert (HERE/'replay-completion.json').is_file(),'No replay completion proof'
        assert hashlib.sha256((HERE/'input-policy.pt').read_bytes()).hexdigest()==POLICY
        event('replay_worker_finished',training_updates=0,automatic_retries=0,policy_sha256_unchanged=POLICY,child_exit_code=code)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        signal.alarm(0)
if __name__=='__main__': main()