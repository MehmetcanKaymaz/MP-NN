from controller import Controller
from quad_model import Model
from traj_planner import Traj_Planner
import numpy as np
import random

def conf_u(u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u

def calculate_score(state_list,traj_list):
    pose_err=0
    vel_err=0
    psi_err=0
    N=len(state_list[:,0])
    for i in range(1,N):
        pose_err+=np.sqrt(pow(state_list[i,0]-traj_list[i,0],2)+pow(state_list[i,1]-traj_list[i,1],2)+pow(state_list[i,2]-traj_list[i,2],2))
        vel_err+=np.sqrt(pow(state_list[i,3]-traj_list[i,4],2)+pow(state_list[i,4]-traj_list[i,5],2)+pow(state_list[i,5]-traj_list[i,6],2))
        psi_err+=abs(state_list[i,8]-traj_list[i,3])

    total_err=(pose_err+vel_err+5*psi_err)/T

    return total_err
        

controller=Controller()
quad=Model()
traj=Traj_Planner()


epp=100000
x_lim=10
y_lim=10
z_lim=10
psi_lim=0

vx_lim=10
vy_lim=10
vz_lim=10



for i in range(epp):
    print("Episode {} is running".format(i))
    p0=[0,0,0,0]
    v0=[0,0,0,0]
    pf=[random.uniform(-x_lim,x_lim),random.uniform(-y_lim,y_lim),random.uniform(-z_lim,z_lim),random.uniform(-psi_lim,psi_lim)]
    vf=[random.uniform(-vx_lim,vx_lim),random.uniform(-vy_lim,vy_lim),random.uniform(-vz_lim,vz_lim),0]
    T=.0
    save_statu=False
    while True:
        T+=.25
        dt=.01
        N=int(T/dt)
        t=np.linspace(0,T,N)
        traj.find_traj(x_initial=p0,x_final=pf,v_initial=v0,v_final=vf,T=T)

        state_list=np.zeros((N,12))
        state_list[0,:]=[p0[0],p0[1],p0[2],v0[0],v0[1],v0[2],0,0,p0[3],0,0,v0[3]]
        traj_list=np.zeros((N,8))
        traj_list[0,:]=[p0[0],p0[1],p0[2],p0[3],v0[0],v0[1],v0[2],v0[3]]

        quad.reset(state_list[0,:])

        for i in range(1,N):
            vel_ref=traj.get_vel(t[i])
            pose_ref=traj.get_target(t[i])

            target=[vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],pose_ref[3]]
            traj_all=[pose_ref[0],pose_ref[1],pose_ref[2],pose_ref[3],vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],vel_ref[3]]
            traj_list[i,:]=traj_all
            states=quad.x
            u=controller.run_controller(x=states[3:12],x_t=target)
            x=quad.run_model(u=conf_u(u))
            state_list[i,:]=x


        score=calculate_score(state_list,traj_list)
        if score<200:
            traj.save_data(T)
            save_statu=True
            print("Optimal solution is fount at T:{}".format(T))
            break
        if T>19.5:
            traj.save_data(0)
            save_statu=False
            print("Optimal solution is not fount!!!")
            break

traj.data_to_picle()






