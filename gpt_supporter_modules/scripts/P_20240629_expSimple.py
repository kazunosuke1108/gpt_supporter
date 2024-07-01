import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from openai import OpenAI

from M_20240427_analysisManagement import analysisManagement
from M_20240430_expCommons import ExpCommons

class expSimple(analysisManagement,ExpCommons):
    def __init__(self):
        super().__init__()
        self.logger=self.prepare_log()
        self.trial_dir_path=self.create_trial_dir()
        motion_history_csv_path="C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/motion_history.csv"
        self.motion_history=pd.read_csv(motion_history_csv_path,names=["timestamp","x_A","y_A","roomName_A","category_A","x_B","y_B","roomName_B","category_B"])
        # print(self.motion_history)
        self.client = OpenAI(api_key="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")

    def extract_scene(self,motion_history,mode="15"):
        if mode=="event":
            scenes=motion_history.loc[1]
            for index in range(1,len(motion_history)):
                if (motion_history.loc[index,"category_A"]!=motion_history.loc[index-1,"category_A"]) or (motion_history.loc[index,"category_B"]!=motion_history.loc[index-1,"category_B"]):
                    scenes=pd.concat([scenes,motion_history.loc[index]],axis=1)
            scenes=scenes.T
            return scenes
        else:
            scenes=motion_history[motion_history["timestamp"]%int(mode)==0]
            print(scenes)
            return scenes

    def get_initialPrompt(self,mode="category"):
        if mode=="category":
            initialPrompt="""You are an internal processing system for a sensor installed in a hospital ward.
You are going to estimate a person's task using only the location information of the person acquired by the sensor.
There are three policies for estimation.
- Judging from the current position
    - e.g.) Person A is in the kitchen: It is highly likely that person A is cooking or washing dishes.
- Judgment based on movement history
    - e.g.) Person A is in the kitchen immediately after returning home from the supermarket: Person A is likely to be cooking.
- Judging from others' location information
    - e.g.) Person B joins person A at the kitchen: Person B brings ingrediants for A.
Data is obtained each time the position of one of the characters changes.
Answers should be given as follows.
- Task of Nurse A (Most likely)
- Task of Nurse B (Most likely)
Describe the most likely task in detail, not raise multiple candidates.
You cannot add any response except instructed above. 
"""
        else:
            initialPrompt=f"""You are an internal processing system for a sensor installed in a hospital ward.
You are going to estimate a person's task using only the location information of the person acquired by the sensor.
There are three policies for estimation.
- Judging from the current position
    - e.g.) Person A is in the kitchen: It is highly likely that person A is cooking or washing dishes.
- Judgment based on movement history
    - e.g.) Person A is in the kitchen immediately after returning home from the supermarket: Person A is likely to be cooking.
- Judging from others' location information
    - e.g.) Person B joins person A at the kitchen: Person B brings ingrediants for A.
Data is obtained every {int(mode)} seconds.
Answers should be given as follows.
- Task of Nurse A (Most likely)
- Task of Nurse B (Most likely)
Describe the most likely task in detail, not raise multiple candidates.
You cannot add any response except instructed above. 

The location of each room or section of the hallway is as shown below:
name of the space,x,y,category of the space
4A_00_00,0,0,wardRoom_4
4B_00_04,0,4,wardRoom_4
4C_00_08,0,8,wardRoom_4
4D_00_30,0,30,wardRoom_4
4E_00_34,0,34,wardRoom_4
4F_00_38,0,38,wardRoom_4
4G_09_00,9,0,wardRoom_4
4H_09_04,9,4,wardRoom_4
4I_09_08,9,8,wardRoom_4
4J_09_30,9,30,wardRoom_4
4K_09_34,9,34,wardRoom_4
1A_09_24,9,24,wardRoom_1
1B_09_26,9,26,wardRoom_1
1C_09_36,9,36,wardRoom_1
1D_09_38,9,38,wardRoom_1
WCW1_04_02,4,2,wc_w
WCM1_05_02,5,2,wc_m
WCW2_05_36,5,36,wc_w
WCM2_04_36,4,36,wc_m
WC1_06_06,6,6,wc
WC2_03_32,3,32,wc
EV1_04_12,4,12,ev
EV2_05_12,5,12,ev
EV3_04_26,4,26,ev
BT1_06_10,6,10,bath
BT2_06_32,6,32,bath
T1_00_12,0,12,terrace
T2_09_28,9,28,terrace
P_00_26,0,26,pantry
M_09_14,9,14,mtgRoom
S1_00_28,0,28,stairs
S2_09_12,9,12,stairs
Sh_03_30,3,30,shelf_of_diaper
NS_04_20,4,20,nurse_station
H_01_20,1,20,hallway
H_01_26,1,26,hallway
H_02_00,2,0,hallway
H_02_04,2,4,hallway
H_02_08,2,8,hallway
H_02_12,2,12,hallway
H_02_14,2,14,hallway
H_02_20,2,20,hallway
H_02_24,2,24,hallway
H_02_26,2,26,hallway
H_02_30,2,30,hallway
H_02_32,2,32,hallway
H_02_34,2,34,hallway
H_02_38,2,38,hallway
"""
        return initialPrompt

    def get_sequentialPrompt(self,info=[]):
        
        sequentialPrompt=f"""
- timestamp: {info[0]} [s]
- Location of Nurse A: {info[1]}
- Location of Nurse B {info[2]}
"""# {info[3]}
        return sequentialPrompt
    
    def get_finalPrompt(self):
        finalPrompt="""
This is the end of the data. Answer the question about this scenario.
- Why did nurse A visit the ward room? 
- Why did nurse B visit the ward room? 
- Explain in detail what the Nurse A was doing when staying at the ward room, considering A and B's location and activity before and after visiting the ward room
"""
        return finalPrompt

    def main(self,gpt=False):
        self.scenes=self.extract_scene(motion_history=self.motion_history,mode="2")

        messages=[]
        initialPrompt=self.get_initialPrompt(mode="2")
        self.logger.info(f"initial prompt:\n{initialPrompt}")
        if gpt:
            messages.append({
            "role":"system",
            "content":initialPrompt
            })

        for index,scene in self.scenes.iterrows():
            # if (scene["timestamp"]>=64) & (scene["timestamp"]<=94):
            #     diapers="(Possesses diapers)"
            # else:
            #     diapers=""
            # sequentialPrompt=self.get_sequentialPrompt([scene["timestamp"],scene["category_A"],scene["category_B"]])#,diapers])
            sequentialPrompt=self.get_sequentialPrompt([scene["timestamp"],[scene["x_A"],scene["y_A"]],[scene["x_B"],scene["y_B"]]])#,diapers])
            self.logger.info(f"sequential prompt no. {index}\n{sequentialPrompt}")
            if gpt:
                messages.append({
                "role":"user",
                "content":sequentialPrompt
                })
            
            if gpt:
                completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                response=completion.choices[-1].message.content
                messages.append({
                    "role":"assistant",
                    "content":response
                })
                print(response)
                self.logger.info(f"response no.{index}:\n{response}")
                self.write_picklelog({
                    "completion":completion,
                    "messages":messages,
                    },self.trial_dir_path+"/"+os.path.basename(self.trial_dir_path)+f"_completion_{str(index).zfill(2)}.pickle")

        pass

        if gpt:
            finalPrompt=self.get_finalPrompt()
            self.logger.info(f"final prompt:\n{finalPrompt}")
            messages.append({
            "role":"user",
            "content":finalPrompt
            })            
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            response=completion.choices[-1].message.content
            messages.append({
                "role":"assistant",
                "content":response
            })
            print(response)
            self.logger.info(f"response no.{index}:\n{response}")
            self.write_picklelog({
                "completion":completion,
                "messages":messages,
                },self.trial_dir_path+"/"+os.path.basename(self.trial_dir_path)+f"_completion_final.pickle")

cls=expSimple()
cls.main(gpt=True)