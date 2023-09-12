#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
# Author 2 : Uthama Kadengodlu          NETID: 
#
# This file implements gradiant descent class

import numpy as np
import pandas as pd 



Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",skiprows=[0], header = None)



def parse_file(data_file):
        Temperature = data_file[data_file.columns[0]]
        Temperature_list = Temperature.to_list()
        for i in range (0,len(Temperature_list)):
            Temperature_list[i] = float(Temperature_list[i])

        
        Exh_vacc = data_file[data_file.columns[1]]
        Exh_vacc_list = Exh_vacc.to_list()
        for i in range (0,len(Exh_vacc_list)):
            Exh_vacc_list[i] = float(Exh_vacc_list[i])

        
        Amb_press = data_file[data_file.columns[2]]
        Amb_press_list = Amb_press.to_list()
        for i in range (0,len(Amb_press_list)):
            Amb_press_list[i] = float(Amb_press_list[i])
       
        
        Rel_humid = data_file[data_file.columns[3]]
        Rel_humid_list = Rel_humid.to_list()
        for i in range (0,len(Amb_press_list)):
            Rel_humid_list[i] = float(Rel_humid_list[i])
       
        
        
        Elec_output = data_file[data_file.columns[4]]
        Elec_output_list = Elec_output.to_list()
        for i in range (0,len(Elec_output_list)):
            Elec_output_list[i] = float(Elec_output_list[i])
        
        
        

     

if __name__ == "__main__":
    parse_file(Training_file)
