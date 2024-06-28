#! /usr/bin/python3
# -*- coding: utf-8 -*- 
import os
import numpy as np
from glob import glob
from datetime import datetime
from pprint import pprint

class ExpCommons():
    csv_labels={}
    csv_labels["odometry"]=["t","x","y","theta","pan"]
    csv_labels["detectron2_joint"]=["gravity","nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow","l_hand","r_hand","l_base","r_base","l_knee","r_knee","l_foot","r_foot"]
    csv_labels["detectron2_joint_2d"]=["timestamp"]
    csv_labels["detectron2_joint_3d"]=["timestamp"]
    csv_labels["detectron2_joint_3d_4"]=["timestamp"]
    for joint_name in csv_labels["detectron2_joint"][1:]:
        suffixes=["_x","_y","_z"]
        for suffix in suffixes:
            csv_labels["detectron2_joint_2d"].append(joint_name+suffix)

    for joint_name in csv_labels["detectron2_joint"]:
        suffixes=["_x","_y","_z"]
        for suffix in suffixes:
            csv_labels["detectron2_joint_3d"].append(joint_name+suffix)

    for joint_name in csv_labels["detectron2_joint"]:
        suffixes=["_x","_y","_z","_0"]
        for suffix in suffixes:
            csv_labels["detectron2_joint_3d_4"].append(joint_name+suffix)

    csv_labels["result_chart"]=["patient_id","type_id","trial_id","n_frames","n_partialout_head","n_partialout_foot","n_partialout_left","n_partialout_right","n_totalout","time_partialout_head","time_partialout_foot","time_partialout_left","time_partialout_right","time_totalout","csvpath"]
    def __init__(self):
        if os.name == "nt":
            if os.path.isdir("//192.168.1.5/common/FY2023/02_M1/05_hayashide/ytlab_nlpmp_modules/results"):
                self.resultsdirpath = f"//192.168.1.5/common/FY2023/02_M1/05_hayashide/ytlab_nlpmp_modules/results"
            else:
                self.resultsdirpath = f"C:/Users/hayashide/ytlab_ros_ws/ytlab_nlpmp/ytlab_nlpmp_modules/results"
                self.scriptsdirpath = f"C:/Users/hayashide/ytlab_ros_ws/ytlab_nlpmp/ytlab_nlpmp_modules/scripts"
        else:
            if os.path.exists("/home/hayashide/catkin_ws"):
                self.resultsdirpath = f"/home/hayashide/catkin_ws/src/ytlab_nlpmp_modules/results"
                self.scriptsdirpath = f"/home/hayashide/catkin_ws/src/ytlab_nlpmp_modules/scripts"
            else:
                self.resultsdirpath = f"/home/hayashide/ytlab_ros_ws/ytlab_nlpmp/ytlab_nlpmp_modules/results"
                self.scriptsdirpath = f"/home/hayashide/ytlab_ros_ws/ytlab_nlpmp/ytlab_nlpmp_modules/scripts"




    def get_now(self):
        try:
            import rospy
            now=rospy.Time.now()
            ans=float(str(now.secs)+"."+format(now.nsecs,'09'))
        except ModuleNotFoundError:
            import time
            ans=time.time()
        return ans
    
    def get_time(self,rostime=None):
        try:
            import rospy
            if rostime==None:
                now=rospy.Time.now()
            else:
                now=rostime
            ans=float(str(now.secs)+"."+format(now.nsecs,'09'))
        except ModuleNotFoundError:
            import time
            ans=time.time()
        return ans
    
    def get_datetime(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def get_date(self):
        return datetime.now().strftime('%Y%m%d')
    
    def create_logdir(self,title):
        daydir = datetime.now().strftime('%Y%m%d')
        daypath = self.resultsdirpath+f"/{daydir}"
        os.makedirs(daypath, exist_ok=True)
        trialdir = datetime.now().strftime('%Y%m%d_%H%M%S')+"_"+title
        self.trialdirpath = daypath+f"/{trialdir}"
        os.makedirs(self.trialdirpath, exist_ok=True)
        return self.trialdirpath
    # self.tfcsvpath=self.trialdirpath+"/"+datetime.now().strftime('%Y%m%d_%H%M%S')+"_tf.csv"
    # self.odomcsvpath=self.trialdirpath+"/"+datetime.now().strftime('%Y%m%d_%H%M%S')+"_od.csv"

    def follow_logdir(self):
        import time
        time.sleep(0.2)
        today=sorted(glob(self.resultsdirpath+"/*"))[-1]
        latestdir=sorted(glob(today+"/*"))[-1]
        return latestdir
    
    def prepare_log(self):
        import os
        from datetime import datetime
        import logging

        logdir=os.path.split(os.path.split(__file__)[0])[0]+"/log"
        logdaydir=logdir+"/"+datetime.now().strftime('%Y%m%d')
        os.makedirs(logdaydir,exist_ok=True)

        logger = logging.getLogger(os.path.basename(__file__))
        logger.setLevel(logging.DEBUG)
        format = "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(levelname)-9s  %(message)s"
        st_handler = logging.StreamHandler()
        st_handler.setLevel(logging.WARNING)
        # StreamHandlerによる出力フォーマットを先で定義した'format'に設定
        st_handler.setFormatter(logging.Formatter(format))

        fl_handler = logging.FileHandler(filename=logdaydir+"/"+datetime.now().strftime('%Y%m%d_%H%M%S')+".log", encoding="utf-8")
        fl_handler.setLevel(logging.INFO)
        # FileHandlerによる出力フォーマットを先で定義した'format'に設定
        fl_handler.setFormatter(logging.Formatter(format))

        logger.addHandler(st_handler)
        logger.addHandler(fl_handler)
        return logger

    def write_csvlog(self,output_data,csvpath,fmt="%s",dim=1):
        if dim==1:
            output_data=[output_data]
        else:
            pass
        try:
            with open(csvpath, 'a') as f_handle:
                np.savetxt(f_handle,output_data,delimiter=",")
        except TypeError:
            with open(csvpath, 'a') as f_handle:
                np.savetxt(f_handle,output_data,delimiter=",",fmt=fmt)    
        except FileNotFoundError:
            np.savetxt(csvpath,output_data,delimiter=",")
        pass  

    def write_picklelog(self,output_dict,picklepath):
        import pickle
        with open(picklepath, mode='wb') as f:
            pickle.dump(output_dict,f)

    def load_picklelog(self,picklepath):
        import pickle
        with open(picklepath,mode="rb") as f:
            data=pickle.load(f)
        return data        

    def get_image(self,data,datatype="rgb_ZED",save=False,rotate=False):
        import cv2
        from cv_bridge import CvBridge
        
        data_time=self.get_time(data.header.stamp)
        if datatype=="rgb_ZED":
            rgb_array = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            rgb_array=np.nan_to_num(rgb_array, copy=False)
            if rotate:
                rgb_array=cv2.rotate(rgb_array,cv2.ROTATE_90_COUNTERCLOCKWISE)
            if save:
                cv2.imwrite(self.bigdata_rgb_dir_path+"/"+str(data_time)+".jpg",rgb_array)
            # rgb_array=cv2.cvtColor(rgb_array,cv2.COLOR_BGR2RGB)
            return rgb_array
        elif datatype=="dpt":
            dpt_array=CvBridge().imgmsg_to_cv2(data)
            if save:
                dpt_array_t=cv2.rotate(dpt_array,cv2.ROTATE_90_COUNTERCLOCKWISE)
                # cv2.imwrite(self.bigdata_dpt_dir_path+"/"+str(data_time)+".jpg",dpt_array_t)
            dpt_array=np.array(dpt_array,dtype=np.float32)            
            return dpt_array


# debug
# cls=ExpCommons()
# dammy_picklepath="/home/hayashide/ytlab_ros_ws/ytlab_mpc/ytlab_mpc_modules/results/20240402/20240402_01/dammy_pickle.pickle"
# a=np.zeros((4,4,100))
# b=10
# output_dict={"a":a,"b":b}
# cls.write_picklelog(output_dict=output_dict,picklepath=dammy_picklepath)
# print(cls.load_picklelog(dammy_picklepath)["b"])