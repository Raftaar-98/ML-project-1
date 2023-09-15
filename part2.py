import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",skiprows=[0], header = None)
Training_file = (Training_file - Training_file.mean())/Training_file.std()
Testing_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Test_Data.csv",skiprows=[0], header = None)
Testing_file = (Testing_file - Testing_file.mean())/Testing_file.std()

def preprocess_data(data_file):
        independent_variable = data_file.iloc[:,0:4]
        dependent_variable = data_file.iloc[:,4:5]
        
        independent_variable=np.concatenate(((np.ones([independent_variable.shape[0],1])),independent_variable),axis = 1)
     
        func = np.zeros([1,5])
        return independent_variable,dependent_variable,func

if __name__ == "__main__":
    ind_var,dep_var,theta = preprocess_data(Training_file)
    test_ind_var, test_dep_var, test_theta = preprocess_data(Testing_file)

    model = SGDRegressor(max_iter=5000,eta0=0.05).fit(ind_var,dep_var)
    print(model)
    print(model.intercept_)
    
    #fig,ax = plt.subplots()
    #ax.plot(np.arange(iterations),epsilon,'r')