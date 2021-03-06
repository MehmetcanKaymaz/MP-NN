import numpy as np
import _pickle as cPickle


class Traj_Planner:
    def __init__(self):
        self.T=5
        self.cx=np.zeros(4)
        self.cy=np.zeros(4)
        self.cz=np.zeros(4)
        self.cpsi=np.zeros(2)

        self.x_initiral=np.zeros(4)
        self.x_final=np.zeros(4)
        self.v_initial=np.zeros(4)
        self.v_final=np.zeros(4)

        self.xm=np.zeros(4)
        self.ym=np.zeros(4)
        self.zm=np.zeros(4)
        self.psim=np.zeros(4)

        self.datalist=[]

    def __calculate_A(self):
        T=self.T
        A=np.matrix([[1,0,0,0],[1,T,pow(T,2),pow(T,3)],[0,1,0,0],[0,1,2*T,3*pow(T,2)]])
        return A

    def __calculate_c(self,x):
        A=self.__calculate_A()
        inverse_A=np.linalg.inv(A)
        x_matrix=np.matrix([[x[0]],[x[1]],[x[2]],[x[3]]])
        #print("x_matrix:{}   ".format(x_matrix))
        c=np.array(np.matmul(inverse_A,x_matrix))
        c_a=np.zeros(4)
        for i in range(4):
            c_a[i]=c[i][0]
        return c_a
    def __calculate_xm(self):
        self.xm=np.array([self.x_initiral[0],self.x_final[0],self.v_initial[0],self.v_final[0]])
        self.ym=np.array([self.x_initiral[1],self.x_final[1],self.v_initial[1],self.v_final[1]])
        self.zm=np.array([self.x_initiral[2],self.x_final[2],self.v_initial[2],self.v_final[2]])
        self.psim=np.array([self.x_initiral[3],self.x_final[3],self.v_initial[3],self.v_final[3]])
        

    def find_traj(self,x_initial,x_final,v_initial,v_final,T):
        self.x_initiral=x_initial
        self.x_final=x_final
        self.v_initial=v_initial
        self.v_final=v_final
        self.T=T
        self.__calculate_xm()
        self.cx=self.__calculate_c(self.xm)
        self.cy=self.__calculate_c(self.ym)
        self.cz=self.__calculate_c(self.zm)
        #self.cpsi=self.__calculate_c(self.psim)
        self.cpsi=np.array([x_initial[3],(x_final[3]-x_initial[3])/self.T])
        #print("cx : {}".format(self.cx)) 
        #return self.T

        
    def __pose_err(self,x,xt):
        err=0
        for i in range(3):
            err+=pow(xt[i]-x[i],2)
        return np.sqrt(err)

    def __find_T(self,current_pose,final_pose,current_vel,final_vel):
        pose_err=self.__pose_err(current_pose,final_pose)
        vel_init=self.__pose_err(current_vel,[0,0,0])
        vel_final=self.__pose_err(final_vel,[0,0,0])
        mean_vel=(vel_final+vel_init)/2
        T=pose_err/mean_vel
        return T

        

    def __calculate_ref(self,c,t):
        x=c[0]+c[1]*t+c[2]*pow(t,2)+c[3]*pow(t,3)
        return x
    def __calculate_vel(self,c,t):
        x=c[1]+c[2]*2*t+c[3]*3*pow(t,2)
        return x

    def get_target(self,t):
        xt=self.__calculate_ref(self.cx,t)
        yt=self.__calculate_ref(self.cy,t)
        zt=self.__calculate_ref(self.cz,t)
        #psit=self.__calculate_ref(self.cpsi,t)
        psit=self.cpsi[0]+self.cpsi[1]*t
        return np.array([xt,yt,zt,psit])

    def get_vel(self,t):
        vxd=self.__calculate_vel(self.cx,t)
        vyd=self.__calculate_vel(self.cy,t)
        vzd=self.__calculate_vel(self.cz,t)
        #rd=self.__calculate_vel(self.cpsi,t)
        rd=self.cpsi[1]
        return np.array([vxd,vyd,vzd,rd])
    


    def get_traj(self,t):
        pos=self.get_target(t)
        vel=self.get_vel(t)
        return np.array([pos,vel])


    def save_data(self,statu,T):
        input_matrix=np.array([self.x_initiral,self.x_final,self.v_initial,self.v_final])
        output_matrix=np.array([self.cx,self.cy,self.cz,[self.cpsi[0],self.cpsi[1],statu,T]])
        data=[]
        data.append(input_matrix)
        data.append(output_matrix)
        self.datalist.append(data)

    def data_to_picle(self,i):
        f = open(f'dataset_final2{(i)}.pkl', 'wb')
        pickler = cPickle.Pickler(f)
        pickler.dump(np.array(self.datalist))
        f.close()







    
