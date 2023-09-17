#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
#
# Author 2 : Uthama Kadengodlu          NETID: UXK210012



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

# getting training file and test file
Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Training_data.csv",skiprows=[0], header = None)
Training_file = (Training_file - Training_file.mean())/Training_file.std()  #Normalizing the data
Testing_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ML-project-1/main/Test_Data.csv",skiprows=[0], header = None)
Testing_file = (Testing_file - Testing_file.mean())/Testing_file.std() #Normalizing the data


# Function to extract features from the data
def preprocess_data(data_file):
        independent_variable = data_file.drop(columns = [4], axis=1)
        dependent_variable = data_file.drop(columns = [0,1,2,3], axis=1)
        dependent_variable = dependent_variable.values
        dependent_variable = dependent_variable.ravel()
        independent_variable=np.concatenate(((np.ones([independent_variable.shape[0],1])),independent_variable),axis = 1)
        return independent_variable,dependent_variable

if __name__ == "__main__":
    independent_variable,dependent_variable = preprocess_data(Training_file)
    learn_rate = 0.05
    iterations = 5000
    model = SGDRegressor(max_iter=iterations,eta0=learn_rate)
    model.fit(independent_variable,dependent_variable)
    theta = model.coef_
    print("Model Coeff: ",model.coef_)
    accu = model.score(independent_variable,dependent_variable)
    print("R square value is ", accu)

    dependent_variable = dependent_variable[1:3071]
    independent_variable_test,dependent_variable_test = preprocess_data(Testing_file)
    pred_data = model.predict(independent_variable_test)
    MSE = mean_squared_error(dependent_variable_test,pred_data)
    print("Mean squared error: ",MSE)

 
    #Plotting in 3d actual value and predicted value
    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')
    zline = pred_data[:250]
    yline = independent_variable_test[:250,1]
    xline = independent_variable_test[:250,3]
    plt.xlabel('Relative Humidity')
    plt.ylabel('Exhaust Vaccum')
    
    ax2.scatter3D(xline, yline, zline, 'gray')
    
    zline2 = dependent_variable_test[:250]
    yline2 = independent_variable_test[:250,1]
    xline2 = independent_variable_test[:250,3]
    plt.xlabel('Relative Humidity')
    plt.ylabel('Exhaust Vaccum')
   
    ax2.scatter3D(xline2, yline2, zline2, 'red')
    ax2.legend(['predicted value','actual value'])
    plt.show()
    
    #Write to a log file
    L = ["Part2: \n","Iterations = " + str(iterations) + ",Learning rate = " + str(learn_rate) + "\nTheta = " + str(theta) + ",MSE = " + str(MSE) + "\n"]
    file = open("log.txt","a")
    file.writelines(L)
    file.close()
