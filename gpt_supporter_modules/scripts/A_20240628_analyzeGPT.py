
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
        pickle_dir_path="C:/Users/hayashide/kazu_ws/gpt_supporter/gpt_supporter_modules/results/20240628/20240628_185040"
        self.logger.info(f"pickle_dir_path: {pickle_dir_path}")
        picklepaths=sorted(glob(pickle_dir_path+"/*.pickle"))
        for picklepath in picklepaths:
            pickle_data=self.load_picklelog(picklepath=picklepath)
            completion=pickle_data["completion"]
            self.logger.info(f"{os.path.basename(picklepath)}\n{completion.choices[-1].message.content}")
    
cls=analyzeGPT()