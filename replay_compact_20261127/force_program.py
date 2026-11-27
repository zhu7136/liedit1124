"""Reuse the proven bounded launcher and add two read-only filtered contact sensors."""
import ast,importlib,json,sys
from pathlib import Path
print(json.dumps(dict(event='force_entry_import_begin',training_updates=0)),flush=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
importlib.invalidate_caches()
import launch_base as base
print(json.dumps(dict(event='force_entry_import_complete',training_updates=0)),flush=True)
original_program=base.program
def program(work):
    tree=original_program(work)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env' for t in n.targets))
    setup=['from isaaclab.sensors import ContactSensorCfg']
    for side in ['left','right']:
        setup.append(f"env_cfg.scene.force_{side} = ContactSensorCfg(prim_path='{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link', update_period=0.0, history_length=1, track_contact_points=True, track_friction_forces=True, max_contact_data_count_per_prim=32, filter_prim_paths_expr=['/World/ground/.*'], debug_vis=False)")
    main.body[index:index]=ast.parse('\n'.join(setup)).body
    ast.fix_missing_locations(tree)
    return tree
base.program=program
base.__file__=__file__
if __name__=='__main__':base.main()