
import os
import pickle
from glob import glob
import numpy as np
import pandas as pd
from M_20240427_analysisManagement import analysisManagement
from M_20240430_expCommons import ExpCommons

class analyzeGPT(analysisManagement,ExpCommons):
    def __init__(self):
        super().__init__()
        self.logger=self.prepare_log()
        self.path_management,self.csv_labels,self.color_dict=self.management_initial()

    def analyze_pickle(self):
        pickle_dir_path="C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/results/20240628/20240628_185040"
        self.logger.info(f"pickle_dir_path: {pickle_dir_path}")
        picklepaths=sorted(glob(pickle_dir_path+"/*.pickle"))
        for picklepath in picklepaths:
            pickle_data=self.load_picklelog(picklepath=picklepath)
            completion=pickle_data["completion"]
            self.logger.info(f"{os.path.basename(picklepath)}\n{completion.choices[-1].message.content}")

    def summarize_motion_history(self):
        motion_history_csv_path="C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/sources/motion_history.csv"
        motion_history=pd.read_csv(motion_history_csv_path,names=self.csv_labels["motion_history"])
        motion_history_event=motion_history.loc[1]
        for index in range(1,len(motion_history)):
            if (motion_history.loc[index,"roomCategory_A"]!=motion_history.loc[index-1,"roomCategory_A"]) or (motion_history.loc[index,"roomCategory_B"]!=motion_history.loc[index-1,"roomCategory_B"]):
                motion_history_event=pd.concat([motion_history_event,motion_history.loc[index]],axis=1)
        motion_history_event=motion_history_event.T
        motion_history_event.to_csv(motion_history_csv_path[:-4]+"_event.csv")

        motion_history_10=motion_history[motion_history["timestamp"]%int(10)==0]
        motion_history_10.to_csv(motion_history_csv_path[:-4]+"_10.csv")
        pass

cls=analyzeGPT()
cls.summarize_motion_history()