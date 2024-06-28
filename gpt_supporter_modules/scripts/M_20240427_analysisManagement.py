#! /usr/bin/python3
# -*- coding: utf-8 -*-

from glob import glob
import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class analysisManagement():
    def __init__(self):
        plt.rcParams["figure.figsize"] = (15,10)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["font.size"] = 24
        plt.rcParams['font.family'] = 'Times New Roman'
        
    def management_initial(self):
        ## パス管理dict作成
        path_management={}
        # csvのnames作成
        csv_labels={}
        csv_labels["odometry"]=["timestamp","x","y","theta","pan"]
        csv_labels["command_velocity"]=["timestamp","v_x","v_y","v_z","omg_x","omg_y","omg_z"]
        csv_labels["detectron2_joint"]=["gravity","nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow","l_hand","r_hand","l_base","r_base","l_knee","r_knee","l_foot","r_foot"]
        csv_labels["detectron2_joint_trunk"]=["gravity","trunk","nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow","l_hand","r_hand","l_base","r_base","l_knee","r_knee","l_foot","r_foot"]
        csv_labels["detectron2_joint_2d"]=["timestamp"]
        csv_labels["detectron2_joint_3d"]=["timestamp"]
        csv_labels["detectron2_joint_3d_4"]=["timestamp"]
        for joint_name in csv_labels["detectron2_joint"][1:]:
            suffixes=["_x","_y"]
            for suffix in suffixes:
                csv_labels["detectron2_joint_2d"].append(joint_name+suffix)
        
        for joint_name in csv_labels["detectron2_joint_trunk"]:
            suffixes=["_x","_y","_z"]
            for suffix in suffixes:
                csv_labels["detectron2_joint_3d"].append(joint_name+suffix)
        
        for joint_name in csv_labels["detectron2_joint_trunk"]:
            suffixes=["_x","_y","_z","_0"]
            for suffix in suffixes:
                csv_labels["detectron2_joint_3d_4"].append(joint_name+suffix)

        csv_labels["result_chart"]=["patient_id","type_id","trial_id","n_frames","n_partialout_head","n_partialout_foot","n_partialout_left","n_partialout_right","n_totalout","time_partialout_head","time_partialout_foot","time_partialout_left","time_partialout_right","time_totalout","csvpath"]
        csv_labels["imu"]=["timestamp","orien_x","orien_y","orien_z","orien_w","ang_vel_x","ang_vel_y","ang_vel_z","lin_acc_x","lin_acc_y","lin_acc_z"]
        csv_labels["hsrzr8"]=["timestamp","x","y","theta","pan","v_x","v_y","omega_theta","omega_pan"]
        cov_list=["orien_cov","ang_vel_cov","lin_acc_cov"]
        for cov_name in cov_list:
            for i in np.arange(1,10,1):
                csv_labels["imu"].append(cov_name+"_"+str(i).zfill(2))
        

        # 実験種別と色の対応表
        color_dict={"00":"r",
                "01":"r",
                "02":"k",
                "03":"m",
                "04":"k",
                "05":"k",
                "06":"b",
                "07":"k",
                "08":"c",
                "09":"k",
                "10":"k",
                "11":"k",
                }
        
        return path_management,csv_labels,color_dict

    def json_saver(path_management,csv_labels,color_dict):
        json_dict={
        "path_management":path_management,
        "csv_labels":csv_labels,
        "color_dict":color_dict,
        }
        tf = open(path_management["json_dir_path"]+f"/analysis_database.json", "w")
        json.dump(json_dict, tf)
        tf.close()

    def draw_labels_timeseries_position(self):
        plt.xlabel("Time $\it{t}$ [s]")
        plt.ylabel("Position $\it{x}$ [m]")
        plt.grid()
        # plt.legend()

    def draw_labels_timeseries_velocity(self):
        plt.xlabel("Time $\it{t}$ [s]")
        plt.ylabel("Velocity $\it{v}$ [m/s]")
        plt.grid()
        # plt.legend()

    def draw_labels_fft(self):
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

    def get_module_path(self):
        if os.name == "nt": # Windows
            home = os.path.expanduser("~")
        else: # ubuntu
            home=os.environ['HOME']        
        # print("HOME: "+home)
        
        workspace_dir_name="kazu_ws"
        module_name="gpt_supporter_modules"
        module_dir_path=home+"/"+workspace_dir_name+"/"+module_name[:-len("_modules")]+"/"+module_name
        if os.path.isdir(module_dir_path):
            pass
        else:#dockerのとき
            module_dir_path=home+"/"+"catkin_ws/src"+"/"+module_name
            if not os.path.isdir(module_dir_path):
                raise FileNotFoundError("module directory not found: "+module_dir_path)
        
        return module_dir_path

    def create_results_dir(self):
        module_dir_path=self.get_module_path()

        results_dir_path=module_dir_path+"/"+"results"
        os.makedirs(results_dir_path,exist_ok=True)
        todays_results_dir_path=results_dir_path+"/"+datetime.now().strftime('%Y%m%d')
        os.makedirs(todays_results_dir_path,exist_ok=True)
        print("results_dir_path: "+module_dir_path)
        return todays_results_dir_path
    
    def create_trial_dir(self):
        todays_results_dir_path=self.create_results_dir()
        trial_dir_path=todays_results_dir_path+"/"+self.get_datetime()
        os.makedirs(trial_dir_path,exist_ok=True)
        return trial_dir_path
    
    def get_datetime(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def plot_ffts(self,freqs,Amps,labels,result_dir_path,memo=""):
        nData=len(freqs)
        margin = 0.2  #0 <margin< 1
        totoal_width = 0.8 - margin
        for i,(freq, Amp, label) in enumerate(zip(freqs,Amps,labels)):
            pos = freq[:len(freq)//2] - totoal_width *( 1- (2*i+1)/nData )/2
            # print(len(freq))
            # print(len(Amp))
            # print(label)
            # plt.bar(freq[:len(freq)//2],Amp[:len(Amp)//2],label=label)
            print(pos.shape)
            print(len(Amp))
            print(len(Amp[:len(Amp)//2]))
            plt.bar(pos,Amp[:len(Amp)//2],label=label,width = totoal_width/nData)
        plt.title(f"FFT: {memo}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        # plt.xlim([3,10])
        # plt.ylim([0,5])
        plt.legend()
        plt.savefig(result_dir_path+"/"+"fft_"+memo+".png")
        plt.close()

    def draw_labels_timeseries_position(self):
        plt.xlabel("Time $\it{t}$ [s]")
        plt.ylabel("Position $\it{x}$ [m]")
        plt.grid()
        plt.legend()

    def draw_labels_timeseries_velocity(self):
        plt.xlabel("Time $\it{t}$ [s]")
        plt.ylabel("Velocity $\it{v}$ [m/s]")
        plt.grid()
        plt.legend()

    def draw_labels_fft(self):
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()

    def jpg2gif(self,image_paths,gif_path):
        from PIL import Image
        images=[]
        for idx,image in enumerate(image_paths):
            img=Image.open(image)
            images.append(img)
            print("[jpg->gif] Now processing...",idx)
        images[0].save(gif_path,save_all=True, append_images=images[1:],optimize=True, duration=33, loop=0)

    def jpg2mp4(self,image_paths,mp4_path,size):
        import cv2
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(mp4_path,fourcc, 30.0, size)
        for idx,image in enumerate(image_paths):
            img=cv2.imread(image)
            video.write(img)
            print(f"now processing: {os.path.basename(image)} {idx}/{len(image_paths)}")
        video.release()