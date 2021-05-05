# this script executes training of the model based on polynomial regression
# vanilla gradient descent will be be used will MSE and regularisation term

from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import math

def polynomial_expansion(X):
	# introduction of columns upto 3 degree of binomial X1, X2
	X1 = X.iloc[:, 0]
	X2 = X.iloc[:, 1]
	expanded_X = pd.DataFrame (data, columns = ['1st term','2nd term','3rd term','4th term'])
	expanded_X['1st term'] = X1**3
	expanded_X['2nd term'] = X2**3 
	expanded_X['3rd term'] = 3*(X1**2)*X2
	expanded_X['4th term'] = 3*(X2**2)*X1
	return expanded_X

def train_GD( X_train, Y_train, epochs, learning_rate):
	# training using batch GD
	theta = np.zeros(X_train.shape[1])
	i = 0
	while i<epochs:
		theta = theta - diff_cost( X_train, Y_train, theta)
		i+=1
	return theta

def cross_product(a, b):
	product = pd.DataFrame (data, columns = ['1st term'])
	product['1st term'] = b[0]*a[0] + b[1]*a[1] + b[2]*a[2] + b[3]*a[3]
	return product

def diff_cost( X_train, Y_train, theta):
	# calculating the differential of cost using MSE cost function
    y1 = hypothesis(X_train, theta)
    print(y1-Y_train)
   # print(Y_train)
    # differential = X1^3 + X2^3 + 3*X1^2*X2 + 3*X1^2*X1
    #return (2/(X_train.shape[0])) * sum( np.dot( (y1-Y_train), X_train) )

def hypothesis( X_train, theta):
	return X_train*theta.T

##foreign function

def computeRSS(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, Y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = 4 #int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - Y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeRSS(X, Y, theta)

    return theta #, cost

##

if __name__ == "__main__":

	data = pd.read_csv("FINAL_DATASET.csv")
	
	# columns split
	X = data.iloc[:, 2:4]
	Y = data.iloc[:, 2]
	print("Splitting done!")

	# ploynomial columns introduced to X
	X = polynomial_expansion(X)
	#print(X)

	# dataset split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=True)
	#print(Y_train)
	#theta = np.zeros(4)
	# OPTIMISATION OF COST FUNCTION:
	theta = train_GD(X_train, Y_train, 150, 0.03)

	#theta = gradientDescent(X_train, Y_train, theta, 0.03, 100)

	print("Output:")
	print(theta)