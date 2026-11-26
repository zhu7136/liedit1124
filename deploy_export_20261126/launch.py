"""One export-only task, 240s entry limit reserving startup within 300s budget."""
import argparse,ast,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SOURCE_COMMIT='325e3ba397440e6e183b69391b8966f3710cd078'
DESCRIPTOR_SHA='3740a371e5b29e1e3f61a7cfbc632541f98fa6dc53e243e8ad21a3b7b3dca9f9'
def event(name,**kw):print(json.dumps({'event':name,**kw}),flush=True)
def child(work):
    sys.path[:0]=[str(work/'source/limx_rl_lab'),str(work/'flux'),str(work/'scripts/rsl_rl'),str(HERE)]
    if importlib.util.find_spec('isaaclab') is None:
        for base in [Path('/workspace/isaaclab/source'),Path('/isaaclab/source'),Path('/opt/isaaclab/source')]:
            if (base/'isaaclab/isaaclab').is_dir():
                sys.path[:0]=[str(base/p) for p in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']];break
    from offline_dependencies import prepare_dependencies
    prepare_dependencies();os.chdir(work)
    original=(work/'scripts/rsl_rl/train.py').read_text()
    marker='    from kneel_hooks import audit_environment\n'
    assert original.count(marker)==1
    prefix=original.split(marker)[0]
    # Original environment creation/configuration is retained byte-for-byte.
    program=prefix+'    from export_hooks import export_and_finish\n    export_and_finish(env)\n    env.close()\n\nif __name__ == "__main__":\n    main()\n    simulation_app.close()\n'
    tree=ast.parse(program)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='load_init_checkpoint')]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert not (isinstance(node.func,ast.Attribute) and node.func.attr in ['learn','load','save','run_bounded_training'])
            assert not (isinstance(node.func,ast.Name) and node.func.id=='OnPolicyRunner')
    sys.argv=[str(work/'scripts/rsl_rl/train.py'),'--task','LimX-HU-D04-01-Flat-BeyondMimic-No-State-Estimation','--motion_file',str(work/'motions/kneel_candidate/motion.npz'),'--num_envs','4096','--max_iterations','0','--seed','42','--experiment_name','hud04_kneel_nose_20261126','--run_name','export_only_task542','--logger','tensorboard','--headless','--device','cuda:0']
    event('export_only_before_simulator',training_updates=0,checkpoint_loading=False)
    exec(compile(tree,str(work/'scripts/rsl_rl/train.py'),'exec'),{'__name__':'__main__','__file__':str(work/'scripts/rsl_rl/train.py')})
def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--child',action='store_true');p.add_argument('--work',type=Path);args=p.parse_args()
    os.environ.update(PYTHONUNBUFFERED='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y')
    if args.child:child(args.work);return
    process=None
    def cutoff(*unused):
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        event('export_hard_timeout',worker_limit_seconds=240);os._exit(124)
    signal.signal(signal.SIGALRM,cutoff);signal.alarm(240)
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=10).strip();assert actual==args.expected_commit
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,h in manifest['entry_files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    assert manifest['source_commit']==SOURCE_COMMIT and manifest['original_descriptor_sha256']==DESCRIPTOR_SHA and manifest['worker_limit_seconds']==240 and manifest['training_updates']==0
    spec=importlib.util.spec_from_file_location('verified542_delivery',ROOT/'kneel_20261126/launch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    work=HERE/'verified-workload';m.restore(work,DESCRIPTOR_SHA)
    event('export_only_worker_begin',source_task='TASK_20261126_542',source_commit=SOURCE_COMMIT,worker_limit_seconds=240,num_envs=4096,training_updates=0,automatic_retries=0,export_commit=actual)
    launcher=next((str(x) for x in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')] if x.is_file() and os.access(x,os.X_OK)),sys.executable)
    try:
        process=subprocess.Popen([launcher,str(Path(__file__).resolve()),'--expected-commit',actual,'--child','--work',str(work)],start_new_session=True)
        code=process.wait(timeout=220)
        assert code==0,f'Export child exited {code}'
        assert (HERE/'export-completion.json').is_file(),'No successful export proof'
        event('export_only_worker_finished',training_updates=0,child_exit_code=code)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        signal.alarm(0)
if __name__=='__main__':main()