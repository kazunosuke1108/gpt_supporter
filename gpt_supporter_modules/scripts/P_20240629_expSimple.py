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

    def get_initialPrompt(self):
        """
        あなたは病棟に設置されたセンサの内部処理システムである．
        あなたはこれから，センサで取得した人物の位置情報のみを用いて，人物のタスクを推定する．
        推定する際の思考方法として，3つの方針を与える．
        - 現在位置から判断する
            - 例) Aさんがキッチンにいる: Aさんは炊事か洗い物をしている可能性が高い
        - 移動履歴から判断する
            - 例) Aさんはスーパーから帰宅した直後にキッチンにいる: Aさんは炊事をしている可能性が高い
        - 他者の位置情報から判断する
            - 例) Aさんは多くの友達と一緒にキッチンに居る: Aさんはパーティの準備をする可能性が高い
        データは登場人物のいずれかの位置が変化する度に取得される．
        回答は以下のように行うこと．
        - Aのタスク
        - Bのタスク
        """
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
        return initialPrompt

    def get_sequentialPrompt(self,info=[]):
        if len(info)==3:
            sequentialPrompt=f"""
- timestamp: {info[0]} [s]
- Location of Nurse A: {info[1]}
- Location of Nurse B {info[2]}
"""
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
        self.scenes=self.extract_scene(motion_history=self.motion_history,mode="10")

        messages=[]
        initialPrompt=self.get_initialPrompt()
        self.logger.info(f"initial prompt:\n{initialPrompt}")
        if gpt:
            messages.append({
            "role":"system",
            "content":initialPrompt
            })

        for index,scene in self.scenes.iterrows():
            sequentialPrompt=self.get_sequentialPrompt([scene["timestamp"],scene["category_A"],scene["category_B"]])
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