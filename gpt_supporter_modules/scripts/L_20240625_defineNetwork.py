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
        plt.rcParams["figure.figsize"] = (10,30)
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
        node_dict["4A_00_00"]={"loc":(0,0),"cat":"wardRoom_4"}
        node_dict["4B_00_04"]={"loc":(0,4),"cat":"wardRoom_4"}
        node_dict["4C_00_08"]={"loc":(0,8),"cat":"wardRoom_4"}
        node_dict["4D_00_30"]={"loc":(0,30),"cat":"wardRoom_4"}
        node_dict["4E_00_34"]={"loc":(0,34),"cat":"wardRoom_4"}
        node_dict["4F_00_38"]={"loc":(0,38),"cat":"wardRoom_4"}
        node_dict["4G_09_00"]={"loc":(9,0),"cat":"wardRoom_4"}
        node_dict["4H_09_04"]={"loc":(9,4),"cat":"wardRoom_4"}
        node_dict["4I_09_08"]={"loc":(9,8),"cat":"wardRoom_4"}
        node_dict["4J_09_30"]={"loc":(9,30),"cat":"wardRoom_4"}
        node_dict["4K_09_34"]={"loc":(9,34),"cat":"wardRoom_4"}

        node_dict["1A_09_24"]={"loc":(9,24),"cat":"wardRoom_1"}
        node_dict["1B_09_26"]={"loc":(9,26),"cat":"wardRoom_1"}
        node_dict["1C_09_36"]={"loc":(9,36),"cat":"wardRoom_1"}
        node_dict["1D_09_38"]={"loc":(9,38),"cat":"wardRoom_1"}

        node_dict["WCW1_04_02"]={"loc":(4,2),"cat":"wc_w"}
        node_dict["WCM1_05_02"]={"loc":(5,2),"cat":"wc_m"}
        node_dict["WCW2_05_36"]={"loc":(5,36),"cat":"wc_w"}
        node_dict["WCM2_04_36"]={"loc":(4,36),"cat":"wc_m"}
        node_dict["WC1_06_06"]={"loc":(6,6),"cat":"wc"}
        node_dict["WC2_03_32"]={"loc":(3,32),"cat":"wc"}

        node_dict["EV1_04_12"]={"loc":(4,12),"cat":"ev"}
        node_dict["EV2_05_12"]={"loc":(5,12),"cat":"ev"}
        node_dict["EV3_04_26"]={"loc":(4,26),"cat":"ev"}

        node_dict["BT1_06_10"]={"loc":(6,10),"cat":"bath"}
        node_dict["BT2_06_32"]={"loc":(6,32),"cat":"bath"}

        node_dict["T1_00_12"]={"loc":(0,12),"cat":"terrace"}
        node_dict["T2_09_28"]={"loc":(9,28),"cat":"terrace"}

        node_dict["P_00_26"]={"loc":(0,26),"cat":"pantry"}

        node_dict["M_09_14"]={"loc":(9,14),"cat":"mtgRoom"}

        node_dict["S1_00_28"]={"loc":(0,28),"cat":"stairs"}
        node_dict["S2_09_12"]={"loc":(9,12),"cat":"stairs"}

        node_dict["Sh_03_30"]={"loc":(3,30),"cat":"shelf_of_diaper"}

        node_dict["NS_04_20"]={"loc":(4,20),"cat":"nurse_station"}


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
            # [4,10],
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
        hallway_loc_list[:,1]=2*hallway_loc_list[:,1]
        
        for loc in hallway_loc_list:
            node_dict[f"H_{str(loc[0]).zfill(2)}_{str(loc[1]).zfill(2)}"]={"loc":tuple(loc),"cat":"hallway"}
        return node_dict
    
    def define_edges(self):
        edge_list=[]
        # room2hallway
        edge_list.append(("4A_00_00","H_02_00"))
        edge_list.append(("4B_00_04","H_02_04"))
        edge_list.append(("4C_00_08","H_02_08"))
        edge_list.append(("4D_00_30","H_02_30"))
        edge_list.append(("4E_00_34","H_02_34"))
        edge_list.append(("4F_00_38","H_02_38"))
        edge_list.append(("4G_09_00","H_07_00"))
        edge_list.append(("4H_09_04","H_07_04"))
        edge_list.append(("4I_09_08","H_07_08"))
        edge_list.append(("4J_09_30","H_07_30"))
        edge_list.append(("4K_09_34","H_07_34"))
        edge_list.append(("1A_09_24","H_07_24"))
        edge_list.append(("1B_09_26","H_07_26"))
        edge_list.append(("1C_09_36","H_07_36"))
        edge_list.append(("1D_09_38","H_07_38"))
        # equipment2hallway
        edge_list.append(("WCW1_04_02","H_04_00"))
        edge_list.append(("WCM1_05_02","H_05_00"))
        edge_list.append(("WCM2_04_36","H_04_38"))
        edge_list.append(("WCW2_05_36","H_05_38"))
        edge_list.append(("WC1_06_06","H_07_06"))
        edge_list.append(("WC2_03_32","H_02_32"))
        edge_list.append(("BT1_06_10","H_07_10"))
        edge_list.append(("BT2_06_32","H_07_32"))
        edge_list.append(("EV1_04_12","H_04_14"))
        edge_list.append(("EV2_05_12","H_05_14"))
        edge_list.append(("EV3_04_26","H_04_24"))
        edge_list.append(("T1_00_12","H_02_12"))
        edge_list.append(("T2_09_28","H_08_28"))
        edge_list.append(("S1_00_28","H_01_26"))
        edge_list.append(("Sh_03_30","H_02_30"))
        # edge_list.append(("S2","H_01_13"))
        edge_list.append(("S2_09_12","H_08_14"))
        edge_list.append(("M_09_14","H_08_14"))
        edge_list.append(("P_00_26","H_01_26"))
        # hallway2hallway
        ## main loop
        edge_list.append(("H_02_00","H_02_04"))
        edge_list.append(("H_02_04","H_02_08"))
        edge_list.append(("H_02_08","H_02_12"))
        edge_list.append(("H_02_08","H_02_12"))
        edge_list.append(("H_02_12","H_02_14"))
        edge_list.append(("H_02_14","H_02_20"))
        edge_list.append(("H_02_20","H_02_24"))
        edge_list.append(("H_02_24","H_02_26"))
        edge_list.append(("H_02_26","H_02_30"))
        edge_list.append(("H_02_30","H_02_32"))
        edge_list.append(("H_02_32","H_02_34"))
        edge_list.append(("H_02_34","H_02_38"))

        edge_list.append(("H_02_38","H_04_38"))
        edge_list.append(("H_04_38","H_05_38"))
        edge_list.append(("H_05_38","H_07_38"))

        edge_list.append(("H_07_38","H_07_36"))
        edge_list.append(("H_07_36","H_07_34"))
        edge_list.append(("H_07_34","H_07_32"))
        edge_list.append(("H_07_32","H_07_30"))
        edge_list.append(("H_07_30","H_07_28"))
        edge_list.append(("H_07_28","H_07_26"))
        edge_list.append(("H_07_26","H_07_24"))
        edge_list.append(("H_07_24","H_07_20"))
        edge_list.append(("H_07_20","H_07_14"))
        edge_list.append(("H_07_14","H_07_10"))
        edge_list.append(("H_07_10","H_07_08"))
        edge_list.append(("H_07_08","H_07_06"))
        edge_list.append(("H_07_06","H_07_04"))
        edge_list.append(("H_07_04","H_07_00"))

        edge_list.append(("H_07_00","H_05_00"))
        edge_list.append(("H_05_00","H_04_00"))
        edge_list.append(("H_04_00","H_02_00"))
        ## around ss
        edge_list.append(("H_02_14","H_04_14"))
        edge_list.append(("H_04_14","H_05_14"))
        edge_list.append(("H_05_14","H_07_14"))

        edge_list.append(("H_02_20","NS_04_20"))
        edge_list.append(("NS_04_20","H_07_20"))

        edge_list.append(("H_02_24","H_04_24"))
        edge_list.append(("H_04_24","H_07_24"))

        ## around halls
        edge_list.append(("H_01_20","H_02_20"))
        edge_list.append(("H_01_26","H_02_26"))
        edge_list.append(("H_01_20","H_01_26"))

        edge_list.append(("H_07_14","H_08_14"))
        edge_list.append(("H_07_20","H_08_20"))
        edge_list.append(("H_08_14","H_08_20"))
        
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