# Automatic-Steering-Angle-based-Vehicle-Control

The main objective of this project is to create and compare different models for the prediction of steering angle for a vehicle using the values of Throttle and Road angles as input.

Road images and corresponding throttle and steering angle values were obtained from a Unity3D simulation of an unmanned autonomous ground vehicle being run in a virtual road environment. Each of the 2018 data points obtained contains road image along with a tuple having steering angle and throttle values. The images were processed using Gaussian Blur, Sobel Filter, Non-Maxima Suppression in progression. Then geometric formulas are deployed to obtain the angle of edges and hence angle of the road in image. All these corresponding angles are recorded to provide a parameter vector. All the parameter values were stored in convenient format and analysed further for

For the prediction models we used the following algorithms: Polynomial Regression, Decision Tree Regression, Random Forest.
The results produced by models upon training on the obtained dataset were then evaluated on the metrics: R-square and RMSE. The evaluations were then compared to provide a conclusion.

This study helped in practical exploration of mathematical basics and intricacies of analysing a dataset. Finding patterns in the data provides avenues in automation on existing and abundant data. Thus, the aforementioned Steering Angle prediction project tries to help in realising the true intentions of this course.
