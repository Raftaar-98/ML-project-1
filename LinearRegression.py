#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
#
# Author 2 : Uthama Kadengodlu          NETID: 
# This file implements gradiant descent class
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error


Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",skiprows=[0], header = None)
Training_file = (Training_file - Training_file.mean())/Training_file.std()
Testing_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Test_Data.csv",skiprows=[0], header = None)
Testing_file = (Testing_file - Testing_file.mean())/Testing_file.std()


class user_def_model:
    def preprocess_data(data_file):
            independent_variable = data_file.iloc[:,0:4]
            dependent_variable = data_file.iloc[:,4:5].values
        
            independent_variable=np.concatenate(((np.ones([independent_variable.shape[0],1])),independent_variable),axis = 1)
     
            func = np.zeros([1,5])
            return independent_variable,dependent_variable,func

    def gen_cost_function(ind_var,dep_var,theta):
            return np.sum(np.power(((ind_var@theta.T)-dep_var),2))/(2 * len(ind_var))

    def GD(ind_var, dep_var,theta, iterations, learning_rate):
            epsilon = np.zeros(iterations)
            for i in range(iterations):
                theta = theta - (learning_rate/len(ind_var)) * np.sum(ind_var * (ind_var @ theta.T - dep_var),axis = 0)
                epsilon[i] = np.sum(np.power(((ind_var@theta.T)-dep_var),2))/(2 * len(ind_var))
            return theta,epsilon

    def predict(x,y):
            pred_value = np.zeros(len(y))
            for  i in range(len(y)):
                pred_value[i] = x @ y[i]
            return pred_value
    def mse(theta,pred_val,actual_val):
            summing = 0
            for i in range(len(pred_val)):
                    summing = summing + (np.power((pred_val[i]-actual_val[i]),2))
            summing = summing / (2 * len(pred_val))
            return summing

if __name__ == "__main__":
    md = user_def_model
    ind_var,dep_var,theta = md.preprocess_data(Training_file)
    test_ind_var, test_dep_var, test_theta = md.preprocess_data(Testing_file)
    learn_rate = 0.05
    iterations = 100
    theta,epsilon = md.GD(ind_var,dep_var,theta,iterations,learn_rate)
    print(theta)
    print(md.gen_cost_function(test_ind_var,test_dep_var,theta))
    pred_data = md.predict(theta,test_ind_var)
   
    print("Mean squared error: ",md.mse(theta,pred_data,test_dep_var))

    fig,ax = plt.subplots()
    ax.plot(np.arange(iterations),epsilon,'r')
    plt.show()

    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    zline = pred_data[:100]
    yline = test_ind_var[:100,0]
    xline = test_ind_var[:100,1]
    ax2.scatter3D(xline, yline, zline, 'gray')
    
    zline2 = test_dep_var[:100]
    yline2 = test_ind_var[:100,0]
    xline2 = test_ind_var[:100,1]
    ax2.scatter3D(xline2, yline2, zline2, 'red')
    plt.show()

   
  
    
