from controller import Controller
from quad_model import Model
from traj_planner import Traj_Planner
import numpy as np
import matplotlib.pyplot as plt
from nn_traj_planer import Traj_Planer
import time

def conf_u(u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u
        

controller=Controller()
quad=Model()
traj=Traj_Planer()


p0=[0,0,0,0]
v0=[0,0,0,0]

T=15
dt=.01
N=int(T/dt)
t=np.linspace(0,T,N)
Tc=0

gate_list=[[3,3,0],[-2,3,0],[-2,-2,0],[1,-4,0]]

state_list=np.zeros((N,12))
state_list[0,:]=[p0[0],p0[1],p0[2],v0[0],v0[1],v0[2],0,0,p0[3],0,0,v0[3]]
traj_list=np.zeros((N,8))
traj_list[0,:]=[p0[0],p0[1],p0[2],p0[3],v0[0],v0[1],v0[2],v0[3]]
quad.reset(state_list[0,:])

for gate in gate_list:

    vf=[0,0,0]
    traj.run_planner(p0,gate,v0,vf)

    i=0
    while True:
        if Tc>=15:
            break
        i+=1
        Tc+=dt
        pose_ref,vel_ref=traj.get_traj(t[i])

        target=[vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],pose_ref[3]]
        traj_all=[pose_ref[0],pose_ref[1],pose_ref[2],pose_ref[3],vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],vel_ref[3]]
        traj_list[i,:]=traj_all
        states=quad.x
        u=controller.run_controller(x=states[3:12],x_t=target)
        x=quad.run_model(u=conf_u(u))
        state_list[i,:]=x
        pose=x[0:3]
        err=np.sqrt(pow(pose[0]-gate[0],2)+pow(pose[1]-gate[1],2)+pow(pose[2]-gate[2],2))
        if err<1:
            break





