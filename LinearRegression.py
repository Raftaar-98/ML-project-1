#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
# Author 2 : Uthama Kadengodlu          NETID: 
#
# This file implements gradiant descent class

import numpy as np
import pandas as pd 


Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv")


def parse_file(data_file):
        mpg = data_file[data_file.columns[0]]
        mpg_list = mpg.to_list()
        
        cylinders = data_file[data_file.columns[1]]
        cylinders_list = cylinders.to_list()
       
        displacement = data_file[data_file.columns[2]]
        displacement_list = displacement.to_list()
        
        horsepower = data_file[data_file.columns[3]]
        horsepower_list = horsepower.to_list()
        
        weight = data_file[data_file.columns[4]]
        weight_list = weight.to_list()
        
        acceleration = data_file[data_file.columns[5]]
        acceleration_list = acceleration.to_list()
        
        model_year = data_file[data_file.columns[6]]
        model_year_list = model_year.to_list()
        
        origin = data_file[data_file.columns[7]]
        origin_list = origin.to_lise()
        
        car_name = data_file[data_file.columns[8]]
        car_name_list = car_name.to_list()

     

if __name__ == "__main__":
    parse_file(Training_file)
