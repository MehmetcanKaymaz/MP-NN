import numpy as np
import torch


class Traj_Planer:
    def __init__(self):

        self.inputs=np.zeros(7)
        self.c=np.zeros(10)
        
        self.net=torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
	    torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 12),
    )
        self.net.load_state_dict(torch.load("Traj-Models/checkpoint-3-1000.pth",map_location=torch.device('cpu')))
        self.net.eval() 

    def __conf_inputs(self,pose0,posef,vel0,velf):
        return [posef[0],posef[1],posef[2],posef[3],vel0[0],vel0[1],vel0[2],velf[0],velf[1],velf[3]]
    
    def __run_nn(self):
        self.c=self.net(torch.from_numpy(np.array(self.inputs,np.float32))).cpu().detach().numpy()#.unsqueeze(0)

    def run_planner(self,pose0,posef,vel0,velf):
        self.inputs=self.__conf_inputs(pose0,posef,vel0,velf)
        self.__run_nn()

    def get_traj(self,t):
        xd=self.c[0]*t+self.c[1]*pow(t,2)+self.c[2]*pow(t,3)
        yd=self.c[3]*t+self.c[4]*pow(t,2)+self.c[5]*pow(t,3)
        zd=self.c[6]*t+self.c[7]*pow(t,2)+self.c[8]*pow(t,3)
        psid=self.c[9]

        vxd=self.c[0]+2*self.c[1]*t+3*self.c[2]*pow(t,2)
        vyd=self.c[3]+2*self.c[4]*t+3*self.c[5]*pow(t,2)
        vzd=self.c[6]+2*self.c[7]*t+3*self.c[8]*pow(t,2)
        rd=0
        T=self.c[10]
        statu=self.c[11]

        pose_ref=[xd,yd,zd,psid]
        vel_ref=[vxd,vyd,vzd,rd]

        return pose_ref,vel_ref,T,statu



