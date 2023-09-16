from tkinter.tix import COLUMN
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



def preprocess_data(data_file):
        independent_variable = data_file.drop(columns = [4], axis=1)
        dependent_variable = data_file.drop(columns = [0,1,2,3], axis=1)
        dependent_variable = dependent_variable.values
        dependent_variable = dependent_variable.ravel()
        independent_variable=np.concatenate(((np.ones([independent_variable.shape[0],1])),independent_variable),axis = 1)
        return independent_variable,dependent_variable

if __name__ == "__main__":
    independent_variable,dependent_variable = preprocess_data(Training_file)
   

    model = SGDRegressor(max_iter=5000,eta0=0.05)
    model.fit(independent_variable,dependent_variable)
   
    print(model.coef_)
    
    #fig,ax = plt.subplots()
    #ax.plot(np.arange(iterations),epsilon,'r')