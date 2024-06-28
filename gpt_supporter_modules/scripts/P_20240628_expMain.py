import os
from glob import glob

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from M_20240427_analysisManagement import analysisManagement
from M_20240430_expCommons import ExpCommons
# from L_20240625_defineNetwork import defineNetwork

class expMain(analysisManagement,ExpCommons):
    def __init__(self):
        super().__init__()
        self.logger=self.prepare_log()
        self.trial_dir_path=self.create_trial_dir()

        ## parameters
        self.control_dt=1
        self.control_hz=1/self.control_dt

        ## define trajectory
        self.t, self.traj_A, self.traj_B, self.traj_C=self.define_traj()

        ## load graph
        pickledata=self.load_picklelog("C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/20240628_125259_G.pickle")
        self.G=pickledata["G"]
        self.node_dict=pickledata["node_dict"]
        self.edge_list=pickledata["edge_list"]

        ## label the trajectory
        self.label_traj(self.t, self.traj_A,self.node_dict,self.edge_list)
        self.label_traj(self.t, self.traj_B,self.node_dict,self.edge_list)
        self.label_traj(self.t, self.traj_C,self.node_dict,self.edge_list)

    def define_traj(self):
        t=np.arange(0,50,self.control_dt)

        def fill_traj(checkpoints,traj,vel):
            current_idx=0
            for idx_checkpoint in range(len(checkpoints)-1):
                start_point=checkpoints[idx_checkpoint]
                goal_point=checkpoints[idx_checkpoint+1]
                if (start_point[0]==goal_point[0]) and (start_point[1]==goal_point[1]):
                    tmax=10
                    traj[0,current_idx:int(current_idx+tmax*self.control_hz)]=start_point[0]
                    traj[1,current_idx:int(current_idx+tmax*self.control_hz)]=start_point[1]
                else:
                    tmax=np.max([abs((goal_point[0]-start_point[0])/vel),abs((goal_point[1]-start_point[1])/vel)])
                    traj[0,current_idx:int(current_idx+tmax*self.control_hz)]=(goal_point[0]-start_point[0])/tmax*(t[current_idx:int(current_idx+tmax*self.control_hz)]-t[current_idx])+start_point[0]
                    traj[1,current_idx:int(current_idx+tmax*self.control_hz)]=(goal_point[1]-start_point[1])/tmax*(t[current_idx:int(current_idx+tmax*self.control_hz)]-t[current_idx])+start_point[1]
                current_idx=int(current_idx+tmax*self.control_hz)
            traj[:,current_idx:]=np.nan        
            return traj    

        ## A
        vel_A=1.0 # [m/s]
        checkpoints=np.array([
            [0,0],
            [2,0],
            [2,10],
            [2,10],
            [2,19],
            [4,19],
            [4,18],
            [4,18],
        ])
        traj_A=np.zeros((2,len(t)))
        traj_A=fill_traj(checkpoints=checkpoints,traj=traj_A,vel=vel_A)

        ## B
        vel_B=1.0 # [m/s]
        checkpoints=np.array([
            [0,19],
            [2,19],
            [2,10],
            [2,10],
            [2,0],
            [4,0],
            [4,1],
            [4,1],
        ])
        traj_B=np.zeros((2,len(t)))
        traj_B=fill_traj(checkpoints=checkpoints,traj=traj_B,vel=vel_B)


        ## C
        vel_C=1.0 # [m/s]
        checkpoints=np.array([
            [0,13],
            [0,13],
            [2,13],
            [2,10],
            [2,10],
            [2,7],
            [7,7],
            [7,4],
            [9,4],
            [9,4],
        ])
        traj_C=np.zeros((2,len(t)))
        traj_C=fill_traj(checkpoints=checkpoints,traj=traj_C,vel=vel_C)

        gs=GridSpec(2,2)
        plt.subplot(gs[:,0])
        plt.plot(traj_A[0,:],traj_A[1,:])
        plt.plot(traj_B[0,:],traj_B[1,:])
        plt.plot(traj_C[0,:],traj_C[1,:])
        plt.xlabel("Position $/it{x}$ [m]")
        plt.ylabel("Position $/it{y}$ [m]")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplot(gs[0,1])
        plt.plot(t,traj_A[0,:])
        plt.plot(t,traj_B[0,:])
        plt.plot(t,traj_C[0,:])
        plt.xlabel("Time $/it{t}$ [s]")
        plt.ylabel("Position $/it{x}$ [m]")
        plt.subplot(gs[1,1])
        plt.plot(t,traj_A[1,:])
        plt.plot(t,traj_B[1,:])
        plt.plot(t,traj_C[1,:])
        plt.xlabel("Time $/it{t}$ [s]")
        plt.ylabel("Position $/it{x}$ [m]")
        plt.savefig(self.trial_dir_path+"/"+os.path.basename(self.trial_dir_path)+"_traj.jpg")
        # plt.pause(1)

        return t, traj_A, traj_B, traj_C

    def label_traj(self,t,traj,node_dict,edge_list):
        print(node_dict)
        traj_label=[]
        traj_cat=[]
        for i in range(traj.shape[1]):
            min_distance=1000
            min_pos=""
            min_node=""
            label=""
            cat=""
            for candidate in node_dict.keys():
                distance=np.sqrt((traj[0,i]-self.node_dict[candidate]["loc"][0])**2+(traj[1,i]-self.node_dict[candidate]["loc"][1])**2) 
                if distance<min_distance:
                    min_distance=distance
                    min_node=self.node_dict[candidate]
                    min_pos=traj[:,i]
                    label=candidate
                    cat=self.node_dict[candidate]["cat"]
            if label=="":
                min_distance=np.nan
            traj_label.append(label)
            traj_cat.append(cat)
            print(label,cat)

        return traj_label

    def main(self):
        self.define_traj()

cls=expMain()
cls.main()