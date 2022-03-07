from controller import Controller
from quad_model import Model
from traj_planner import Traj_Planner
import numpy as np
import matplotlib.pyplot as plt
from nn_traj_planer import Traj_Planer
import time
from mpl_toolkits.mplot3d import Axes3D

def conf_u(u):
        for i in range(3):
            u[i]=(u[i]-5)/10

        return u
        

controller=Controller()
quad=Model()
traj=Traj_Planer()


p0=[0,0,0,0]
v0=[0,0,0,0]

T=20
dt=.01
N=int(T/dt)
t=np.linspace(0,T,N)
Tc=0

gate_list=np.array([[3,3,0,0],[0,5,0,0],[-2,3,0,0],[0,0,0,0]])
vel_list=[[5,5,0,0],[-5,0,0,0],[0,5,0,0],[0,0,0,0]]

state_list=[]
state_list.append([p0[0],p0[1],p0[2],v0[0],v0[1],v0[2],0,0,p0[3],0,0,v0[3]])
traj_list=[]
traj_list.append([p0[0],p0[1],p0[2],p0[3],v0[0],v0[1],v0[2],v0[3]])
quad.reset(state_list[0])
p0c=np.array(p0)
for i in range(4):

    vf=vel_list[i]

    target=gate_list[i]-p0c
    traj.run_planner(p0,target,v0,vf)
    pose_ref,vel_ref,Ts,statu=traj.get_traj(0)
    if int(statu)==1:
        print("no solution")
        break

    Ns=int(Ts/dt)
    ts=np.linspace(0,Ts,Ns)
    for k in range(N):
        pose_ref,vel_ref,T,statu=traj.get_traj(ts[k])

        target=[vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],pose_ref[3]]
        traj_all=[pose_ref[0],pose_ref[1],pose_ref[2],pose_ref[3],vel_ref[0]*np.cos(pose_ref[3])+vel_ref[1]*np.sin(pose_ref[3]),-vel_ref[0]*np.sin(pose_ref[3])+vel_ref[1]*np.cos(pose_ref[3]),vel_ref[2],vel_ref[3]]
        traj_list.append(traj_all)
        states=quad.x
        u=controller.run_controller(x=states[3:12],x_t=target)
        x=quad.run_model(u=conf_u(u))
        state_list.append(x)
        pose=x[0:3]
        p0c=np.array([pose[0],pose[1],pose[2],x[8]])
        err=np.sqrt(pow(pose[0]-gate_list[i][0],2)+pow(pose[1]-gate_list[i][1],2)+pow(pose[2]-gate_list[i][2],2))
        if err<1 or statu==1:
            break





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

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(traj_list[:,0], traj_list[:,1], traj_list[:,2], 'red')
ax.plot3D(state_list[:,0], state_list[:,1], state_list[:,2], 'blue')
ax.scatter3D(gate_list[:,0], gate_list[:,1], gate_list[:,2], c=gate_list[:,2], cmap='Greens')