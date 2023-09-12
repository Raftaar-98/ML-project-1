#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
#
# Author 2 : Uthama Kadengodlu          NETID: 
# This file implements gradiant descent class

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 



Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",skiprows=[0], header = None)
Training_file = (Training_file - Training_file.mean()/Training_file.std())



def parse_file(data_file):
        Ind_var = data_file.iloc[:,0:4]
        ones = np.ones([Ind_var.shape[0],1])
        Ind_var = np.concatenate((ones,Ind_var),axis = 1)

        Dep_var = data_file.iloc[:,4:5].values
        func = np.zeros([1,5])
        

        learning_rate = 0.01
        iterations = 1000
        return Ind_var,Dep_var,func,learning_rate,iterations

def epsilon(Ind_var,Dep_Var,func):
         tobesummed = np.power(((Ind_var @ func.T)-Dep_Var),2)
         return np.sum(tobesummed)/(2 * len(Ind_var))

def gradientDescent(Ind_var,Dep_var,func,iterations,learning_rate):
        summ = np.zeros(iterations)
        for i in range(iterations):
            func = func - (learning_rate/len(Ind_var)) * np.sum(Ind_var * (Ind_var @ func.T - Dep_var), axis = 0)
            summ[i] = epsilon(Ind_var,Dep_var,func)

        return func,summ
            


if __name__ == "__main__":
    Ind_var,Dep_var,func,learning_rate,iterations = parse_file(Training_file)
    print(Ind_var)
    print(Dep_var)
    print(func)
    g,cost = gradientDescent(Ind_var,Dep_var,func,iterations,learning_rate)
    print(g)

    final_error = epsilon(Ind_var,Dep_var,g)
    print(final_error)

    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations),cost,'r')
    
