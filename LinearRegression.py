#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
# Author 2 : Uthama Kadengodlu          NETID: 
#
# This file implements gradiant descent class

import numpy as np
import pandas as pd 

Temperature=[]
Temperature_list=[]
Exh_vacc=[]
Exh_vacc_list=[]
Amb_press=[]
Amb_press_list=[]
Rel_humid=[]
Rel_humid_list=[]
Elec_output=[]
Elec_output_list=[]

Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",skiprows=[0], header = None)

Training_file.columns = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",nrows=0).columns
print((Training_file.columns))

def parse_file(data_file):
        Temperature = data_file[data_file.columns[0]]
        Temperature_list = Temperature.to_list()
        for i in range (0,len(Temperature_list)):
            Temperature_list[i] = int(Temperature_list[i])


        Exh_vacc = data_file[data_file.columns[1]]
        Exh_vacc_list = Exh_vacc.to_list()
        for i in range (0,len(Exh_vacc_list)):
            Exh_vacc_list[i] = int(Exh_vacc_list[i])

        Amb_press = data_file[data_file.columns[2]]
        Amb_press_list = Amb_press.to_list()
        for i in range (0,len(Amb_press_list)):
            Amb_press_list[i] = int(Amb_press_list[i])
       
        Rel_humid = data_file[data_file.columns[3]]
        Rel_humid_list = Rel_humid.to_list()
        print(len(Rel_humid_list))
       
        
        
        Elec_output = data_file[data_file.columns[4]]
        Elec_output_list = Elec_output.to_list()
        for i in range (0,len(Elec_output_list)):
            Elec_output_list[i] = int(Elec_output_list[i])
        
        
        

     

if __name__ == "__main__":
    parse_file(Training_file)
