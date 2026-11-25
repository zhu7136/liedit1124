"""Safe native deploy conversion and redundant credential-free return channels."""
import base64,hashlib,json,shutil
from pathlib import Path
import yaml

TERMS=['command','motion_anchor_ori_b','base_ang_vel','joint_pos','joint_vel','actions']
DIMS=[58,6,3,29,29,29]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_metadata(source,raw,out):
    from limx_rl_lab.tasks.beyondmimic.robots.limx.robot_cfg import OLI_PR_JOINT_NAMES
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    src=yaml.load(source.read_text(),Loader=yaml.BaseLoader)
    cfg={'joint_ids_map':[int(v) for v in src['joint_ids_map']],'step_dt':float(src['step_dt'])}
    for name in ['stiffness','damping','default_joint_pos']:cfg[name]=[float(v) for v in src[name]]
    ids=raw.action_manager._terms['joint_pos']._joint_ids
    if isinstance(ids,slice):ids=list(range(len(raw.scene['robot'].data.joint_names)))[ids]
    elif hasattr(ids,'tolist'):ids=ids.tolist()
    names=[raw.scene['robot'].data.joint_names[i] for i in ids]
    assert names==OLI_PR_JOINT_NAMES and len(names)==29
    action=src['actions']['joint_pos']
    cfg['actions']={'joint_pos':{'joint_names':names,'joint_ids':[int(v) for v in ids],'scale':[float(v) for v in action['scale']]}}
    assert list(raw.observation_manager.active_terms['policy'])==TERMS
    assert tuple(raw.observation_manager.group_obs_dim['policy'])==(154,)
    cfg['observations']={}
    for name,dim in zip(TERMS,DIMS):
        item=src['observations'][name];scale=[float(v) for v in item['scale']];assert len(scale)==dim
        clip=item.get('clip');clip=None if clip in [None,'null','None','~'] else [float(v) for v in clip]
        history=int(item.get('history_length','1'));assert history==1
        cfg['observations'][name]={'scale':scale,'clip':clip,'history_length':history}
    assert len(cfg['joint_ids_map'])==31 and sorted(cfg['joint_ids_map'])==list(range(31))
    assert [cfg['joint_ids_map'][i] for i in ids]==list(range(29))
    assert len(action['scale'])==29 and all(len(cfg[n])==31 for n in ['stiffness','damping','default_joint_pos'])
    assert abs(cfg['step_dt']-.02)<1e-9
    target=out/'deploy.yaml';target.write_text(yaml.safe_dump(cfg,sort_keys=False));assert yaml.safe_load(target.read_text())==cfg
    shutil.copy2(source,out/'deploy_native.yaml')
    proof={'actor_obs_dim':154,'action_dim':29,'step_dt':cfg['step_dt'],'joint_names':names,
        'controller_deploy_sha256':sha(target),'native_deploy_sha256':sha(source),
        'motion_sha256':sha(raw.cfg.commands.motion.motion_file),'hardware_approved':False}
    (out/'deploy_verification.json').write_text(json.dumps(proof,indent=2))
    return proof

def send_file(path,name,event):
    raw=Path(path).read_bytes();assert 0<len(raw)<180000
    encoded=base64.b64encode(raw).decode();chunks=[encoded[i:i+3000] for i in range(0,len(encoded),3000)]
    digest=hashlib.sha256(raw).hexdigest()
    for i,chunk in enumerate(chunks):event('deploy_file_chunk',name=name,index=i,total=len(chunks),bytes=len(raw),sha256=digest,base64=chunk)
    event('deploy_file_complete',name=name,bytes=len(raw),sha256=digest)

def return_metadata(out,event):
    out=Path(out);proof=json.loads((out/'deploy_verification.json').read_text())
    assert sha(out/'deploy.yaml')==proof['controller_deploy_sha256']
    assert sha(out/'deploy_native.yaml')==proof['native_deploy_sha256']
    for name in ['deploy.yaml','deploy_native.yaml','deploy_verification.json']:send_file(out/name,name,event)
    event('deploy_return_verified',**proof)

def bundle_backup(out,motion,event):
    import torch
    out=Path(out)
    names=['deploy.yaml','deploy_native.yaml','deploy_verification.json','policy.pt','policy.onnx','completion.json']
    files={n:(out/n).read_bytes() for n in names if (out/n).is_file()}
    assert all(n in files for n in ['deploy.yaml','deploy_verification.json'])
    files['motion.npz']=Path(motion).read_bytes()
    manifest={n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in files.items()}
    target=out/'controller_bundle.pt'
    torch.save({'format':'hud04_controller_bundle_v1','files':files,'manifest':manifest},target)
    restored=torch.load(target,map_location='cpu',weights_only=True)
    assert restored['files']==files and restored['manifest']==manifest
    event('controller_backup_ready',name=target.name,bytes=target.stat().st_size,sha256=sha(target))