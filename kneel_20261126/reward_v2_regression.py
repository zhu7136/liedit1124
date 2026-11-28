"""Bounded CPU/meta regression, not native Isaac/CUDA or a training result."""
import importlib.util,json,types
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('balanced_rewards',P/'overlay/flux/balanced_support_rewards.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
class Scene:
    def __getitem__(self,name):return self.robot
def mock(device='cpu'):
    scene=Scene()
    robot=types.SimpleNamespace(data=types.SimpleNamespace(default_mass=torch.full((5,1),540./9.81)))
    robot.find_bodies=lambda n,preserve_order=True:(list(range(4)),r.SUPPORT_NAMES) if n==r.SUPPORT_NAMES else ([0],['waist_pitch_link'])
    scene.robot=robot
    f=torch.zeros(5,4,3,device=device)
    if device=='cpu':f[:,:,2]=torch.tensor([[135,135,135,135],[27,27,243,243],[0,131,263,146],[0,0,0,0],[135,135,135,135]])
    scene.sensors={'contact_forces':types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=f),find_bodies=lambda n,preserve_order=True:(list(range(4)),r.SUPPORT_NAMES))}
    cmd=types.SimpleNamespace(time_steps=torch.full((5,),500,device=device))
    return types.SimpleNamespace(scene=scene,command_manager=types.SimpleNamespace(get_term=lambda n:cmd),reset_buf=torch.zeros(5,device=device,dtype=torch.bool)),cmd
e,cmd=mock()
b=r.force_balance(e);c=r.four_support(e);q=r.equal_load(e)
assert b[0]==0 and c[0]==1 and q[0]==0
assert b[1]==0 and c[1]==1 and q[1]>.15
assert c[2]==0 and c[3]==0 and q[3]>.2
assert torch.isfinite(q).all()
assert torch.all(r.force_smooth(e)==0)
e.scene.sensors['contact_forces'].data.net_forces_w[:,:,2]+=10
assert torch.all(r.force_smooth(e)>0)
e.scene.sensors['contact_forces'].data.net_forces_w[:,:,2]+=10
e.reset_buf[:]=True
assert torch.all(r.force_smooth(e)==0)
d=r.diagnostics(e);assert d['hold_env_count']==5 and 'equal_load_error' in d
cmd.time_steps[:]=100
assert torch.all(r.equal_load(e)==0) and torch.all(r.force_balance(e)==0) and torch.all(r.four_support(e)==0)
e,_=mock('meta')
for fn in [r.force_balance,r.four_support,r.equal_load,r.force_smooth]:assert fn(e).device.type=='meta'
print(json.dumps({'passed':True,'test':'CPU/meta only','native_CUDA_verified':False,'checks':['device_alignment','equal_load_not_pair_balance','zero_and_single_knee_rejected','hold_only_gate','reset_force_jump_masked','finite_diagnostics'],'GPU_jobs':0}))