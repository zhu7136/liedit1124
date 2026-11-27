"""Only author two ipsilateral knee-foot filtered pairs before PhysX parsing."""
import json,os
def variant():
    value=os.environ.get('REPLAY_COLLISION_VARIANT');assert value in ['baseline','filtered'];return value
def configure_spawn_filter(env_cfg):
    assert env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
    if variant()=='baseline':return
    original=env_cfg.scene.robot.spawn.func
    def spawn(prim_path,cfg,*args,**kwargs):
        result=original(prim_path,cfg,*args,**kwargs)
        import omni.usd
        from pxr import UsdPhysics
        from isaaclab.sim import find_matching_prims
        stage=omni.usd.get_context().get_stage();roots=find_matching_prims(prim_path);assert roots
        for root in roots:
            for side in ['left','right']:
                knee=stage.GetPrimAtPath(str(root.GetPath())+'/'+side+'_knee_link');foot=stage.GetPrimAtPath(str(root.GetPath())+'/'+side+'_ankle_roll_link');assert knee and foot
                api=UsdPhysics.FilteredPairsAPI.Apply(knee);rel=api.GetFilteredPairsRel()
                if not rel:rel=api.CreateFilteredPairsRel()
                rel.AddTarget(foot.GetPath())
        print(json.dumps(dict(event='pair_filters_authored_before_physics',variant=variant(),robot_prims=[str(x.GetPath()) for x in roots],added_body_pairs_per_robot=2,all_self_collisions_disabled=False)),flush=True)
        return result
    env_cfg.scene.robot.spawn.func=spawn
def verify_filter_authoring(raw):
    import omni.usd
    from pxr import UsdPhysics
    stage=omni.usd.get_context().get_stage();rows=[]
    assert raw.cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True
    for i in range(raw.num_envs):
        for side in ['left','right']:
            root=f'/World/envs/env_{i}/Robot/';knee=stage.GetPrimAtPath(root+side+'_knee_link');assert knee
            targets=[str(x) for x in UsdPhysics.FilteredPairsAPI(knee).GetFilteredPairsRel().GetTargets()];foot=root+side+'_ankle_roll_link'
            assert (foot in targets)==(variant()=='filtered'),(variant(),i,side,targets)
            assert root+side+'_ankle_pitch_link' in targets,'Do not remove the original adjacent-body exclusion'
            rows.append(dict(env=i,side=side,knee_targets=targets,ipsilateral_foot_filtered=foot in targets))
    return dict(variant=variant(),all_self_collisions_disabled=False,pair_filter_authoring_verified=rows)