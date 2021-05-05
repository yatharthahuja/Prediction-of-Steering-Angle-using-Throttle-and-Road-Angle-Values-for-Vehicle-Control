#!/usr/bin/env python
# coding: utf-8

# In[266]:


import pandas as pd
import seaborn as sns
import numpy as np
import operator 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt   
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
 


# In[267]:


df = pd.read_csv('DataSet.csv') 
df.head()


# In[268]:


df.describe()


# In[269]:


# Analyzing dataset
from mpl_toolkits import mplot3d
x = df.ThrottleValue
y = df.CorrectionAngle 
z = df.SteeringAngle

# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
# Add x, y gridlines 
ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2) 
# Creating color map
my_cmap = plt.get_cmap('hsv')
# Creating plot
sctt = ax.scatter3D(x, y, z, alpha = 0.8, c = z, cmap = my_cmap, marker ='o')
 
plt.title("3D scatter plot")
ax.set_xlabel('Throttle Value', fontweight ='bold') 
ax.set_ylabel('Correction Angle', fontweight ='bold') 
ax.set_zlabel('Steering Angle', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
plt.show()


# In[270]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[271]:


sns.pairplot(data=df)
plt.show()


# In[292]:


X = df.drop(df.columns[[0],], axis=1)
y = df[df.columns[0]] 


# In[ ]:





# In[335]:


#Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[336]:


def Poly_equation(feature, coef, intercept, power):
    '''
    Return fitted polynomial equation as a string
    
    feature: list of feature in dataset
    power: degree of each feature in each term (can get from poly.powers_)
    '''
    poly_string = ""
    
    for i in range(len(coef)): # create polynomial term
        
        #Coefficients
        if i == 0:
            term_string = "y = %.3E" % coef[i]
        elif coef[i] >= 0: # add + sign in front of coef
            term_string = "+%.3E" % coef[i]
        else:
            term_string = "%.3E" % coef[i]
        
        #Powers
        feature_order = 0
        for power_iter in power[i]: # power for each feature
            if power_iter == 1 : #degree of that feature = 1
                term_string += '*' + str(feature[feature_order])
            elif power_iter > 1 : #degree of that feature > 1
                term_string += '*' + str(feature[feature_order]) + '^' + str(power_iter)
            feature_order += 1
        poly_string += term_string
    
    #Add intercept
    if intercept >= 0:
        poly_string += "+"
    poly_string += "%.3E" % intercept
    
    return poly_string


# In[337]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[338]:


nPoly = 2
poly = PolynomialFeatures(degree=nPoly, include_bias=None)
x_train_poly = poly.fit_transform(X_train)
print("Polynomial degree: "+str(nPoly))
print("Degree for each feature(X1,X2):\n" + str(poly.powers_))


# In[339]:


from sklearn import linear_model
clf = linear_model.LinearRegression()
y_train_hat = clf.fit(x_train_poly, y_train)
# The coefficients and intercept
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


# In[340]:


print("Equation: " + Poly_equation(['X1','X2'], clf.coef_, clf.intercept_, poly.powers_))


# In[341]:


x1 = X_train[:,0]
x2 = X_train[:,1]
y = y_train


# In[342]:


x1_lin = np.linspace(min(x1),max(x1)) 
x2_lin = np.linspace(min(x2),max(x2))  
x1_grid, x2_grid = np.meshgrid(x1_lin, x2_lin)  
y_grid =y = (0.004177*x1_grid)-(0.1232*x2_grid)-(0.007179*x1_grid**2)-(0.00039*x1_grid*x2_grid)+(0.01136*x2_grid**2)-0.0288
 


# In[343]:


import matplotlib.cm as cm

fig = plt.figure(figsize = (16, 9))
ax = fig.add_subplot(111, projection='3d') 
 

# 2. Line plot
ax.plot_surface(x1_grid, x2_grid, y_grid, cmap = cm.magma, alpha = 0.7) 
# Decoration
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax.set_title("Polynomial degree 2: 3D plot of x1, x2 and y")


plt.show()


# In[344]:



test_x_poly = poly.fit_transform(X_test)
test_y_hat = clf.predict(test_x_poly)
print("R-Square = "+ str(np.round(clf.score(test_x_poly,y_test),3)))
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))

 


# In[345]:


poly_df = PolynomialFeatures(degree = 2)
transform_poly = poly_df.fit_transform(X_test)

linreg2 = LinearRegression()
linreg2.fit(transform_poly,y_test)

polynomial_predict = linreg2.predict(transform_poly)
rmse = np.sqrt(mean_squared_error(y_test,polynomial_predict))
r2 = r2_score(y_test,polynomial_predict)
print("RMSE Score for Test set: " +"{:.2}".format(rmse))
print("R2 Score for Test set: " +"{:.2}".format(r2))


# In[ ]:





# In[ ]:





# In[346]:


#X = df.iloc[:, 1:3].values
#y = df.iloc[:, 0].values
#X


# In[ ]:





# In[347]:


# Feature Scaling
#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


# In[348]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[349]:


from sklearn import metrics
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE Score for Test set: " +"{:.2}".format(rmse))
print("R2 Score for Test set: " +"{:.2}".format(r2))


# In[ ]:





# In[353]:


from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()  
dt_reg.fit(X_train,y_train)
dt_predict = dt_reg.predict(X_test)
 


# In[351]:


rmse = np.sqrt(mean_squared_error(y_test,dt_predict))
r2 = r2_score(y_test,dt_predict)
print("RMSE Score for Test set: " +"{:.2}".format(rmse))
print("R2 Score for Test set: " +"{:.2}".format(r2))
