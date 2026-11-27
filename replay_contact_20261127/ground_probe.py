"""Read-only exact-path ground filters; original wildcard sensors remain intact."""
import json
GROUND_CANDIDATES=['/World/ground/terrain/GroundPlane/CollisionPlane','/World/ground/terrain/GroundPlane','/World/ground/terrain']
def configure_ground_probe(cfg):
    from isaaclab.sensors import ContactSensorCfg
    for side in ['left','right']:
        setattr(cfg.scene,'exact_ground_'+side,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+side+'_ankle_roll_link',update_period=0.0,history_length=1,track_contact_points=True,track_friction_forces=True,max_contact_data_count_per_prim=64,filter_prim_paths_expr=GROUND_CANDIDATES,debug_vis=False))
def inspect_ground_probe(raw):
    from pxr import UsdGeom,UsdPhysics
    import omni.usd
    stage=omni.usd.get_context().get_stage();rows=[]
    for path in GROUND_CANDIDATES:
        p=stage.GetPrimAtPath(path);assert p,path
        rows.append(dict(path=path,usd_type=p.GetTypeName(),collision_api=p.HasAPI(UsdPhysics.CollisionAPI),rigid_body_api=p.HasAPI(UsdPhysics.RigidBodyAPI),local_to_world_matrix=[list(row) for row in UsdGeom.XformCache().GetLocalToWorldTransform(p)]))
    sensors=[raw.scene['exact_ground_'+s] for s in ['left','right']]
    for s in sensors:assert tuple(s.data.force_matrix_w.shape)==(4,1,3,3),tuple(s.data.force_matrix_w.shape)
    return dict(candidate_paths=GROUND_CANDIDATES,candidate_stage_readback=rows,filter_counts=[s.contact_physx_view.filter_count for s in sensors],native_force_matrix_shapes=[list(s.data.force_matrix_w.shape) for s in sensors],original_wildcard_sensor_unchanged=True,measurement_only=True)