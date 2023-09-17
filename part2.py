#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
#
# Author 2 : Uthama Kadengodlu          NETID: UXK210012

from tkinter.tix import COLUMN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

    print("Model Coeff: ",model.coef_)

    dependent_variable = dependent_variable[1:3071]
    independent_variable_test,dependent_variable_test = preprocess_data(Testing_file)
    pred_data = model.predict(independent_variable_test)
    print("predicted data:", pred_data)
    print("Mean squared error: ",mean_squared_error(dependent_variable_test,pred_data))
    #max_value = np.min(independent_variable_test)
    #index = np.where(independent_variable_test == max_value)[0]
    Test_data = Testing_file.drop(columns = [1,3,4])
    plt.plot(Test_data[:12],dependent_variable_test[:12])
    plt.legend(["Test","Actual"])
    plt.show()
    plt.plot(Test_data[:12],pred_data[:12])
    plt.legend(["Test","Pred"])
    plt.show()


