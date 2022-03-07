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

def calculate_score(state_list,traj_list):
    pose_err=0
    vel_err=0
    psi_err=0
    N=len(state_list[:,0])
    for i in range(1,N):
        pose_err+=np.sqrt(pow(state_list[i,0]-traj_list[i,0],2)+pow(state_list[i,1]-traj_list[i,1],2)+pow(state_list[i,2]-traj_list[i,2],2))
        vel_err+=np.sqrt(pow(state_list[i,3]-traj_list[i,4],2)+pow(state_list[i,4]-traj_list[i,5],2)+pow(state_list[i,5]-traj_list[i,6],2))
        psi_err+=abs(state_list[i,8]-traj_list[i,3])

    print("pose err:{}".format(pose_err))
    print("vel err:{}".format(vel_err))
    print("psi err:{}".format(psi_err))
    total_err=(pose_err+vel_err+5*psi_err)/T
    print("total_err:{}".format(total_err))
        

controller=Controller()
quad=Model()
traj=Traj_Planer()


p0=[0,0,0,0]
v0=[4,2,1,0]
pf=[5,5,5,0]
vf=[1,2,0,0]

statu=0
ti=time.time()
traj.run_planner(p0,pf,v0,vf)
tf=time.time()
deltat=tf-ti
print("Delta t : {}".format(deltat))
pose_ref,vel_ref,T,statu=traj.get_traj(0)
print("T:{}".format(T))
print("Statu:{}".format(statu))
dt=.01
N=int(T/dt)
t=np.linspace(0,T,N)



state_list=np.zeros((N,12))
state_list[0,:]=[p0[0],p0[1],p0[2],v0[0],v0[1],v0[2],0,0,p0[3],0,0,v0[3]]
traj_list=np.zeros((N,8))
traj_list[0,:]=[p0[0],p0[1],p0[2],p0[3],v0[0],v0[1],v0[2],v0[3]]

quad.reset(state_list[0,:])

for i in range(1,N):
    pose_ref,vel_ref,T,statu=traj.get_traj(t[i])

    target=[vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],pose_ref[3]]
    traj_all=[pose_ref[0],pose_ref[1],pose_ref[2],pose_ref[3],vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],vel_ref[3]]
    traj_list[i,:]=traj_all
    states=quad.x
    u=controller.run_controller(x=states[3:12],x_t=target)
    x=quad.run_model(u=conf_u(u))
    state_list[i,:]=x


calculate_score(state_list,traj_list)

print("Last states {}".format([state_list[N-1]]))

figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(t,traj_list[:,0],t,state_list[:,0])
axis[0, 0].legend(['traj','real'])
axis[0, 0].set_title("X")

axis[0, 1].plot(t,traj_list[:,1],t,state_list[:,1])
axis[0, 1].legend(['traj','real'])
axis[0, 1].set_title("Y")

axis[1, 0].plot(t,traj_list[:,2],t,state_list[:,2])
axis[1, 0].legend(['traj','real'])
axis[1, 0].set_title("Z")

axis[1, 1].plot(t,traj_list[:,3],t,state_list[:,8])
axis[1, 1].legend(['traj','real'])
axis[1, 1].set_title("Psi")


plt.show()

figure, axis = plt.subplots(2, 2)

axis[0, 0].plot(t,traj_list[:,4],t,state_list[:,3])
axis[0, 0].legend(['traj','real'])
axis[0, 0].set_title("Vx")

axis[0, 1].plot(t,traj_list[:,5],t,state_list[:,4])
axis[0, 1].legend(['traj','real'])
axis[0, 1].set_title("Vy")

axis[1, 0].plot(t,traj_list[:,6],t,state_list[:,5])
axis[1, 0].legend(['traj','real'])
axis[1, 0].set_title("Vz")

axis[1, 1].plot(t,traj_list[:,7],t,state_list[:,11])
axis[1, 1].set_title("r")
axis[1, 1].legend(['traj','real'])


plt.show()




