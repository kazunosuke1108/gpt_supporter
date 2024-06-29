import os
from glob import glob

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from openai import OpenAI
import os

from M_20240427_analysisManagement import analysisManagement
from M_20240430_expCommons import ExpCommons
# from L_20240625_defineNetwork import defineNetwork

class expMain(analysisManagement,ExpCommons):
    def __init__(self):
        super().__init__()
        self.logger=self.prepare_log()
        self.trial_dir_path=self.create_trial_dir()
        plt.rcParams["figure.figsize"] = (20,20)
        plt.rcParams["font.size"] = 12

        ## parameters
        self.control_dt=1
        self.control_hz=1/self.control_dt

        ## define trajectory
        # self.t, self.traj_A, self.traj_B, self.traj_C=self.define_traj()
        self.t, self.traj_A, self.traj_B=self.define_traj()

        ## GPT
        self.client = OpenAI(api_key="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")

        ## load graph
        pickledata=self.load_picklelog("C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/graph/20240629_120929_G.pickle")
        # pickledata=self.load_picklelog("/home/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/graph/20240628_165711_G.pickle")
        self.G=pickledata["G"]
        self.node_dict=pickledata["node_dict"]
        self.edge_list=pickledata["edge_list"]

        ## label the trajectory
        self.label_list_A=self.label_traj(self.t, self.traj_A,self.node_dict,self.edge_list)
        self.label_list_B=self.label_traj(self.t, self.traj_B,self.node_dict,self.edge_list)
        # self.label_list_C=self.label_traj(self.t, self.traj_C,self.node_dict,self.edge_list)
        # print(self.label_list_A)
        # print(self.label_list_B)
        # print(self.label_list_C)
        for index in range(len(self.t)):
            try:
                print(f"A: {self.traj_A[:,index]} {self.label_list_A[index]} {self.node_dict[self.label_list_A[index]]['cat']}")
                print(f"B: {self.traj_B[:,index]} {self.label_list_B[index]} {self.node_dict[self.label_list_B[index]]['cat']}")
            except KeyError:
                break

    def define_traj(self):
        t=np.arange(0,180,self.control_dt)

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
            [4,20],
            [4,20],
            [2,20],
            [2,14],
            [2,12],
            [2,8],
            [2,4],
            [2,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [2,0],
            [2,4],
            [2,8],
            [2,12],
            [2,14],
            [2,20],
            [4,20],
            [4,20],
        ])
        traj_A=np.zeros((2,len(t)))
        traj_A=fill_traj(checkpoints=checkpoints,traj=traj_A,vel=vel_A)

        ## B
        vel_B=1.0 # [m/s]
        checkpoints=np.array([
            [4,20],
            [4,20],
            [4,20],
            [4,20],
            [4,20],
            [2,20],
            [2,24],
            [2,26],
            [2,30],
            [3,30],
            [3,30],
            [2,30],
            [2,26],
            [2,24],
            [2,20],
            [2,14],
            [2,12],
            [2,8],
            [2,4],
            [2,0],
            [0,0],
            [0,0],
            [2,0],
            [2,4],
            [2,8],
            [2,12],
            [2,14],
            [2,20],
            [4,20],
            [4,20],
            [4,20],
            [4,20],
            [4,20],
        ])
        traj_B=np.zeros((2,len(t)))
        traj_B=fill_traj(checkpoints=checkpoints,traj=traj_B,vel=vel_B)


        ## C
        # vel_C=1.0 # [m/s]
        # checkpoints=np.array([
        #     [0,13],
        #     [0,13],
        #     [0,13],
        #     [2,13],
        #     [2,10],
        #     [2,10],
        #     [2,7],
        #     [7,7],
        #     [7,4],
        #     [9,4],
        #     [9,4],
        # ])
        # traj_C=np.zeros((2,len(t)))
        # traj_C=fill_traj(checkpoints=checkpoints,traj=traj_C,vel=vel_C)

        gs=GridSpec(2,2)
        plt.subplot(gs[:,0])
        plt.plot(traj_A[0,:],traj_A[1,:],label="A")
        plt.plot(traj_B[0,:],traj_B[1,:],label="B")
        plt.legend()
        # plt.plot(traj_C[0,:],traj_C[1,:])
        plt.grid()
        plt.xlabel("Position $\it{x}$ [m]")
        plt.ylabel("Position $\it{y}$ [m]")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplot(gs[0,1])
        plt.plot(t,traj_A[0,:],label="A")
        plt.plot(t,traj_B[0,:],label="B")
        for i in range(180):
            if i%15==0:
                plt.plot([i,i],[0,5],"r",linewidth=0.25)
            elif i%5==0:
                plt.plot([i,i],[0,5],"b",linewidth=0.25)
        plt.legend()
        # plt.plot(t,traj_C[0,:])
        plt.xlabel("Time $\it{t}$ [s]")
        plt.ylabel("Position $\it{x}$ [m]")
        plt.grid()
        plt.subplot(gs[1,1])
        plt.plot(t,traj_A[1,:],label="A")
        plt.plot(t,traj_B[1,:],label="B")
        for i in range(180):
            if i%15==0:
                plt.plot([i,i],[0,30],"r",linewidth=0.25)
            elif i%5==0:
                plt.plot([i,i],[0,30],"b",linewidth=0.25)
        plt.legend()
        # plt.plot(t,traj_C[1,:])
        plt.xlabel("Time $\it{t}$ [s]")
        plt.ylabel("Position $\it{x}$ [m]")
        plt.grid()
        # plt.show()
        plt.savefig(self.trial_dir_path+"/"+os.path.basename(self.trial_dir_path)+"_traj.jpg")
        # # plt.pause(1)

        return t, traj_A, traj_B, #traj_C

    def label_traj(self,t,traj,node_dict,edge_list):
        # print(node_dict)
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
            # print(label,cat)

        return traj_label

    def import_initialPrompt(self):
        initialPrompt_path="C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/prompt/20240628_initialPrompt.txt"
        # initialPrompt_path="/home/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/prompt/20240628_initialPrompt.txt"
        with open(initialPrompt_path,"r") as f:
            txt=f.read()
        return txt

    def generate_sequentialPrompt(self,timestamp,pos_a,pos_b):#,pos_c):
        prompt=f"""The next data is as shown below:
- timestamp:{timestamp} [s]
- Position of the nurse A: {pos_a}
- Position of the nurse B: {pos_b}
Please execute the task which was instructed in the first prompt."""
        return prompt
    
    def main(self):
        self.define_traj()
        
        messages=[]
        initial_prompt=self.import_initialPrompt()
        print(initial_prompt,"\n")
        self.logger.info(f"initial prompt:\n{initial_prompt}")
        messages.append({
            "role":"system",
            "content":initial_prompt
        })

        for idx in range(len(self.t)):
            if idx%15!=0:
                continue
            seq_prompt=self.generate_sequentialPrompt(self.t[idx],self.node_dict[self.label_list_A[idx]]["cat"],self.node_dict[self.label_list_B[idx]]["cat"])#,self.label_list_C[idx])
            print(seq_prompt,"\n")
            messages.append({
                "role":"user",
                "content":seq_prompt
            })
            self.logger.info(f"sequential prompt no.{idx}:\n{seq_prompt}")

            # GPT
            # completion = self.client.chat.completions.create(
            #     model="gpt-4o",
            #     messages=messages,
            # )
            # response=completion.choices[-1].message.content
            # messages.append({
            #     "role":"assistant",
            #     "content":response
            # })
            # print(response)
            # self.logger.info(f"response no.{idx}:\n{response}")
            # self.write_picklelog({
            #     "completion":completion,
            #     "messages":messages,
            #     },self.trial_dir_path+"/"+os.path.basename(self.trial_dir_path)+f"_completion_{str(idx).zfill(2)}.pickle")

cls=expMain()
cls.main()