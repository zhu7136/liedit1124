"""Bounded diagnostic ankle balance law. No learning, direct torque API, or policy edits."""
import os
import numpy as np
SPEC=dict(kp_s2=9.,kd_s=6.,cop_offset_limit_m=.04,joint_offset_limit_rad=.18,joint_offset_rate_rad_s=1.,ramp_seconds=.10,velocity_filter_alpha=.30)
def law(error_xy,velocity_error_xy,height,weight,axes_w,stiffness,load_fraction):
    error=np.asarray(error_xy,dtype=float);velocity=np.asarray(velocity_error_xy,dtype=float)
    desired=error+np.clip(height,.3,1.2)/9.81*(SPEC['kp_s2']*error+SPEC['kd_s']*velocity)
    norm=np.linalg.norm(desired);cop=desired*min(1.,SPEC['cop_offset_limit_m']/max(norm,1e-12))
    ground_moment=np.array([cop[1]*weight,-cop[0]*weight,0.])
    # Motor torque on child is opposite the equivalent support moment on the robot.
    motor=-np.einsum('ij,j->i',np.asarray(axes_w),ground_moment)*np.repeat(load_fraction,2)
    raw=motor/np.asarray(stiffness);offset=np.clip(raw,-SPEC['joint_offset_limit_rad'],SPEC['joint_offset_limit_rad'])
    return offset,cop,desired
class Feedback:
    def __init__(self,raw,cmd):
        from isaaclab.utils.math import quat_apply
        self.raw=raw;self.cmd=cmd;self.rotate=quat_apply;self.enabled=os.environ['REPLAY_BALANCE_MODE']=='balance';assert os.environ['REPLAY_BALANCE_MODE'] in ['pd','balance']
        robot=raw.scene['robot'];self.robot=robot;self.mass=robot.root_physx_view.get_masses()[3].to(raw.device);self.local=robot.root_physx_view.get_coms()[3,...,:3].to(raw.device);self.weight=float(self.mass.sum())*9.81
        import json,torch
        from pathlib import Path
        mapping=json.loads((Path.cwd()/'flux/motion_order.json').read_text())['body_mapping'];self.refids=torch.tensor([mapping[n] for n in robot.body_names],device=raw.device,dtype=torch.long)
        joint_names=[robot.joint_names[int(i)] for i in cmd.joint_indexes];self.names=[side+'_ankle_'+axis+'_joint' for side in ['left','right'] for axis in ['pitch','roll']];self.action_ids=[joint_names.index(n) for n in self.names];self.joint_ids=[robot.joint_names.index(n) for n in self.names];self.body_ids=[robot.body_names.index(n.replace('_joint','_link')) for n in self.names]
        self.stiffness=robot.data.joint_stiffness[3,self.joint_ids].detach().cpu().numpy();assert np.all(self.stiffness==60.)
        self.prev=np.zeros(4);self.filtered_velocity=None;self.rows=[];self.step=0
        print(json.dumps(dict(event='balance_feedback_verified',enabled=self.enabled,spec=SPEC,controlled_env=3,controlled_joints=self.names,motor_stiffness=self.stiffness.tolist(),actor_envs_unchanged=True,training_updates=0)),flush=True)
    def apply(self,target):
        import torch
        self.step+=1;r=self.robot;c=self.cmd;f=int(c.time_steps[3]);mass=self.mass;local=self.local;rotate=self.rotate
        bodypos=r.data.body_pos_w[3]-self.raw.scene.env_origins[3];bodyquat=r.data.body_quat_w[3];lc=rotate(bodyquat,local);com=((bodypos+lc)*mass[:,None]).sum(0)/mass.sum()
        vel=((r.data.body_lin_vel_w[3]+torch.cross(r.data.body_ang_vel_w[3],lc,dim=-1))*mass[:,None]).sum(0)/mass.sum()
        rp=c.motion._body_pos_w[f,self.refids];rq=c.motion._body_quat_w[f,self.refids];rlc=rotate(rq,local);reference=((rp+rlc)*mass[:,None]).sum(0)/mass.sum()
        rv=((c.motion._body_lin_vel_w[f,self.refids]+torch.cross(c.motion._body_ang_vel_w[f,self.refids],rlc,dim=-1))*mass[:,None]).sum(0)/mass.sum()
        error=(com-reference).detach().cpu().numpy()[:2];v=(vel-rv).detach().cpu().numpy()[:2]
        self.filtered_velocity=v if self.filtered_velocity is None else (1-SPEC['velocity_filter_alpha'])*self.filtered_velocity+SPEC['velocity_filter_alpha']*v
        local_axes=torch.tensor([[0.,1.,0.],[1.,0.,0.]]*2,device=self.raw.device);axes=rotate(bodyquat[self.body_ids],local_axes).detach().cpu().numpy()
        forces=np.array([float(self.raw.scene['exact_ground_'+s].data.force_matrix_w[3,0,0,2]) for s in ['left','right']]);fraction=np.clip(forces/max(forces.sum(),1e-6),.2,.8) if forces.sum()>50 else np.array([.5,.5]);fraction/=fraction.sum()
        requested,cop,uncapped=law(error,self.filtered_velocity,float(com[2]),self.weight,axes,self.stiffness,fraction)
        ramp=min(1.,self.step*.02/SPEC['ramp_seconds']);desired=requested*ramp if self.enabled else np.zeros(4)
        bounded=self.prev+np.clip(desired-self.prev,-SPEC['joint_offset_rate_rad_s']*.02,SPEC['joint_offset_rate_rad_s']*.02);self.prev=bounded
        assert np.isfinite(bounded).all() and np.abs(bounded).max()<=SPEC['joint_offset_limit_rad']+1e-9
        result=target.clone();result[self.action_ids]+=torch.tensor(bounded,device=result.device,dtype=result.dtype)
        # Ankle hard limits, independently from controller's original actuator-effort limits.
        limits=[(-1.012,.611),(-.436,.436)]*2
        for i,(lo,hi) in zip(self.action_ids,limits):result[i]=result[i].clamp(lo,hi)
        actual=(result[self.action_ids]-target[self.action_ids]).detach().cpu().numpy();assert np.abs(actual).max()<=SPEC['joint_offset_limit_rad']+1e-6
        self.rows.append(np.r_[self.step*.02,f,error,self.filtered_velocity,cop,uncapped,bounded,actual,forces])
        return result
    def arrays(self):
        return dict(balance_metrics=np.asarray(self.rows,dtype=np.float32),balance_metric_columns=np.asarray(['control_time_s','reference_frame','com_error_x_m','com_error_y_m','velocity_error_x_m_s','velocity_error_y_m_s','bounded_cop_offset_x_m','bounded_cop_offset_y_m','raw_cop_offset_x_m','raw_cop_offset_y_m']+['requested_'+n for n in self.names]+['applied_'+n for n in self.names]+['left_ground_vertical_n','right_ground_vertical_n']),balance_enabled=np.asarray(self.enabled))