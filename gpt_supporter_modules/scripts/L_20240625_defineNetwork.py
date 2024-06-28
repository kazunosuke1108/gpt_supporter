import os
from pprint import pprint

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

from M_20240427_analysisManagement import analysisManagement
from M_20240430_expCommons import ExpCommons

class defineNetwork(analysisManagement,ExpCommons):
    def __init__(self):
        super().__init__()
        self.trial_dir_path=self.create_trial_dir()
        self.G=nx.Graph()

        self.node_dict=self.define_nodes()
        # pprint(self.node_dict)
        self.edge_list=self.define_edges()
        # pprint(self.edge_list)

        self.define_graph()

        self.main()

    def define_nodes(self):
        node_dict={}
        ## room
        node_dict["4A"]={"loc":(0,0),"cat":"room_4"}
        node_dict["4B"]={"loc":(0,2),"cat":"room_4"}
        node_dict["4C"]={"loc":(0,4),"cat":"room_4"}
        node_dict["4D"]={"loc":(0,15),"cat":"room_4"}
        node_dict["4E"]={"loc":(0,17),"cat":"room_4"}
        node_dict["4F"]={"loc":(0,19),"cat":"room_4"}
        node_dict["4G"]={"loc":(9,0),"cat":"room_4"}
        node_dict["4H"]={"loc":(9,2),"cat":"room_4"}
        node_dict["4I"]={"loc":(9,4),"cat":"room_4"}
        node_dict["4J"]={"loc":(9,15),"cat":"room_4"}
        node_dict["4K"]={"loc":(9,17),"cat":"room_4"}

        node_dict["1A"]={"loc":(9,12),"cat":"room_1"}
        node_dict["1B"]={"loc":(9,13),"cat":"room_1"}
        node_dict["1C"]={"loc":(9,18),"cat":"room_1"}
        node_dict["1D"]={"loc":(9,19),"cat":"room_1"}

        node_dict["WCW1"]={"loc":(4,1),"cat":"wc_w"}
        node_dict["WCM1"]={"loc":(5,1),"cat":"wc_m"}
        node_dict["WCW2"]={"loc":(5,18),"cat":"wc_w"}
        node_dict["WCM2"]={"loc":(4,18),"cat":"wc_m"}
        node_dict["WC1"]={"loc":(6,3),"cat":"wc"}
        node_dict["WC2"]={"loc":(3,16),"cat":"wc"}

        node_dict["EV1"]={"loc":(4,6),"cat":"ev"}
        node_dict["EV2"]={"loc":(5,6),"cat":"ev"}
        node_dict["EV3"]={"loc":(4,13),"cat":"ev"}

        node_dict["BT1"]={"loc":(6,5),"cat":"bath"}
        node_dict["BT2"]={"loc":(6,16),"cat":"bath"}

        node_dict["T1"]={"loc":(0,6),"cat":"terrace"}
        node_dict["T2"]={"loc":(9,14),"cat":"terrace"}

        node_dict["P"]={"loc":(0,13),"cat":"pantry"}

        node_dict["M"]={"loc":(9,7),"cat":"mtgRoom"}

        node_dict["S1"]={"loc":(0,14),"cat":"stairs"}
        node_dict["S2"]={"loc":(9,6),"cat":"stairs"}

        ## Hallway
        hallway_loc_list=np.array([
            [1,10],
            [1,13],
            [2,0],
            [2,2],
            [2,4],
            [2,6],
            [2,7],
            [2,10],
            [2,12],
            [2,13],
            [2,15],
            [2,16],
            [2,17],
            [2,19],
            [4,0],
            [4,7],
            [4,10],
            [4,12],
            [4,19],
            [5,0],
            [5,7],
            [5,19],
            [5,19],
            [7,0],
            [7,2],
            [7,3],
            [7,4],
            [7,5],
            [7,7],
            [7,10],
            [7,12],
            [7,13],
            [7,14],
            [7,15],
            [7,16],
            [7,17],
            [7,18],
            [7,19],
            # [8,6],
            [8,7],
            [8,10],
            [8,14],
        ])
        
        for loc in hallway_loc_list:
            node_dict[f"H_{str(loc[0]).zfill(2)}_{str(loc[1]).zfill(2)}"]={"loc":tuple(loc),"cat":"hallway"}
        return node_dict
    
    def define_edges(self):
        edge_list=[]
        # room2hallway
        edge_list.append(("4A","H_02_00"))
        edge_list.append(("4B","H_02_02"))
        edge_list.append(("4C","H_02_04"))
        edge_list.append(("4D","H_02_15"))
        edge_list.append(("4E","H_02_17"))
        edge_list.append(("4F","H_02_19"))
        edge_list.append(("4G","H_07_00"))
        edge_list.append(("4H","H_07_02"))
        edge_list.append(("4I","H_07_04"))
        edge_list.append(("4J","H_07_15"))
        edge_list.append(("4K","H_07_17"))
        edge_list.append(("1A","H_07_12"))
        edge_list.append(("1B","H_07_13"))
        edge_list.append(("1C","H_07_18"))
        edge_list.append(("1D","H_07_19"))
        # equipment2hallway
        edge_list.append(("WCW1","H_04_00"))
        edge_list.append(("WCM1","H_05_00"))
        edge_list.append(("WCM2","H_04_19"))
        edge_list.append(("WCW2","H_05_19"))
        edge_list.append(("WC1","H_07_03"))
        edge_list.append(("WC2","H_02_16"))
        edge_list.append(("BT1","H_07_05"))
        edge_list.append(("BT2","H_07_16"))
        edge_list.append(("EV1","H_04_07"))
        edge_list.append(("EV2","H_05_07"))
        edge_list.append(("EV3","H_04_12"))
        edge_list.append(("T1","H_02_06"))
        edge_list.append(("T2","H_08_14"))
        edge_list.append(("S1","H_01_13"))
        # edge_list.append(("S2","H_01_13"))
        edge_list.append(("S2","H_08_07"))
        edge_list.append(("M","H_08_07"))
        edge_list.append(("P","H_01_13"))
        # hallway2hallway
        ## main loop
        edge_list.append(("H_02_00","H_02_02"))
        edge_list.append(("H_02_02","H_02_04"))
        edge_list.append(("H_02_04","H_02_06"))
        edge_list.append(("H_02_04","H_02_06"))
        edge_list.append(("H_02_06","H_02_07"))
        edge_list.append(("H_02_07","H_02_10"))
        edge_list.append(("H_02_10","H_02_12"))
        edge_list.append(("H_02_12","H_02_13"))
        edge_list.append(("H_02_13","H_02_15"))
        edge_list.append(("H_02_15","H_02_16"))
        edge_list.append(("H_02_16","H_02_17"))
        edge_list.append(("H_02_17","H_02_19"))

        edge_list.append(("H_02_19","H_04_19"))
        edge_list.append(("H_04_19","H_05_19"))
        edge_list.append(("H_05_19","H_07_19"))

        edge_list.append(("H_07_19","H_07_18"))
        edge_list.append(("H_07_18","H_07_17"))
        edge_list.append(("H_07_17","H_07_16"))
        edge_list.append(("H_07_16","H_07_15"))
        edge_list.append(("H_07_15","H_07_14"))
        edge_list.append(("H_07_14","H_07_13"))
        edge_list.append(("H_07_13","H_07_12"))
        edge_list.append(("H_07_12","H_07_10"))
        edge_list.append(("H_07_10","H_07_07"))
        edge_list.append(("H_07_07","H_07_05"))
        edge_list.append(("H_07_05","H_07_04"))
        edge_list.append(("H_07_04","H_07_03"))
        edge_list.append(("H_07_03","H_07_02"))
        edge_list.append(("H_07_02","H_07_00"))

        edge_list.append(("H_07_00","H_05_00"))
        edge_list.append(("H_05_00","H_04_00"))
        edge_list.append(("H_04_00","H_02_00"))
        ## around ss
        edge_list.append(("H_02_07","H_04_07"))
        edge_list.append(("H_04_07","H_05_07"))
        edge_list.append(("H_05_07","H_07_07"))

        edge_list.append(("H_02_10","H_04_10"))
        edge_list.append(("H_04_10","H_07_10"))

        edge_list.append(("H_02_12","H_04_12"))
        edge_list.append(("H_04_12","H_07_12"))

        ## around halls
        edge_list.append(("H_01_10","H_02_10"))
        edge_list.append(("H_01_13","H_02_13"))
        edge_list.append(("H_01_10","H_01_13"))

        edge_list.append(("H_07_07","H_08_07"))
        edge_list.append(("H_07_10","H_08_10"))
        edge_list.append(("H_08_07","H_08_10"))
        
        return edge_list

    def define_graph(self):
        self.G.add_nodes_from(self.node_dict.keys())
        self.G.add_edges_from(self.edge_list)
        pass

    def export_adjacency_mtx(self):
        adjacency_mtx=nx.adjacency_matrix(self.G).todense().astype(int)
        df=pd.DataFrame(adjacency_mtx,index=self.node_dict.keys(),columns=self.node_dict.keys())
        df.to_csv(self.trial_dir_path+"/adjacency_mtx.csv")

    def export_pos(self):
        node_dict_key=self.node_dict.keys()
        pos={}
        for key in node_dict_key:
            pos[key]=self.node_dict[key]["loc"]
        df=pd.DataFrame(pos).T
        df = df.rename(columns={0:'x',1:'y'})
        print(df)
        df.to_csv(self.trial_dir_path+"/pos.csv")

    def export_g(self):
        import pickle
        self.export_dir_path=os.path.split(os.path.split(os.path.split(self.trial_dir_path)[0])[0])[0]+"/sources"
        exportdata={}
        exportdata["G"]=self.G
        exportdata["node_dict"]=self.node_dict
        exportdata["edge_list"]=self.edge_list
        with open(self.export_dir_path+"/"+os.path.basename(self.trial_dir_path)+"_G.pickle", mode='wb') as f:
                pickle.dump(exportdata,f)

    def plot_graph(self):
        node_dict_key=self.node_dict.keys()
        pos={}
        col=[]
        for key in node_dict_key:
            pos[key]=self.node_dict[key]["loc"]
            if self.node_dict[key]["cat"]=="room_4":
               c="r"
            elif self.node_dict[key]["cat"]=="room_1":
               c="orange"
            elif "wc" in self.node_dict[key]["cat"]:
               c="g"
            elif self.node_dict[key]["cat"]=="ev":
               c="b"
            elif self.node_dict[key]["cat"]=="bath":
               c="m"
            elif self.node_dict[key]["cat"]=="terrace":
               c="y"
            elif self.node_dict[key]["cat"]=="hallway":
                c="c"
            else:
               c="purple"
            col.append(c)
        # print(pos)
        nx.draw_networkx(self.G,pos=pos,node_color=col,with_labels=True)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.grid()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.savefig(self.trial_dir_path+"/"+os.path.basename(self.trial_dir_path)+".jpg")
        pass


    def main(self):

        # self.export_adjacency_mtx()
        # self.export_pos()
        self.plot_graph()  
        self.export_g()      
        
        pass
cls=defineNetwork()