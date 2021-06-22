# Programming Exercise 2: Logistic Regression Machine Learning
<br>

- [Introduction](#Introduction)
- [1&nbsp;Logistic&nbsp;regression](#1-Logistic-regression)
    - [1.1&nbsp;Visualizing&nbsp;the&nbsp;data](#11-Visualizing-the-data)
    - [1.2&nbsp;Implementation](#12-Implementation)
        - [1.2.1&nbsp;Warmup&nbsp;exercise:&nbsp;sigmoid&nbsp;function](#121-Warmup-exercise-sigmoid-function)
        - [1.2.2&nbsp;Cost&nbsp;function](#122-Cost-function)
        - [1.2.3&nbsp;Gradient](#123-Gradient)
        - [1.2.4&nbsp;Learning&nbsp;&nbsp;θ&nbsp;parameters](#124-Learning--θ-parameters)
        - [1.2.5&nbsp;Evaluating&nbsp;logistic&nbsp;regression](#125-Evaluating-logistic-regression)
        - [1.3&nbsp;Decision&nbsp;boundary（决策边界）](#13-Decision-boundary（决策边界）)
    - [2&nbsp;Regularized&nbsp;logistic&nbsp;regression](#2-Regularized-logistic-regression)
        - [2.1&nbsp;Visualizing&nbsp;the&nbsp;data](#21-Visualizing-the-data)
        - [2.2&nbsp;Feature&nbsp;mapping](#22-Feature-mapping)
        - [2.3&nbsp;Cost&nbsp;function](#23-Cost-function)
        - [2.4&nbsp;Regularized&nbsp;Gradient](#24-Regularized-Gradient)
        - [2.5&nbsp;Learning&nbsp;&nbsp;θ&nbsp;parameters](#25-Learning--θ-parameters)
        - [2.6&nbsp;Evaluating&nbsp;logistic&nbsp;regression](#27-Evaluating-logistic-regression)
        - [2.7&nbsp;Plotting&nbsp;the&nbsp;decision&nbsp;boundary](#26-Plotting-the-decision-boundary)


<font size=3>如果想了解更多的知识，可以去我的机器学习之路 The Road To Machine Learning[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)

# Introduction

>In this exercise, you will implement logistic regression and apply it to two different datasets. Before starting on the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

# 1 Logistic regression
>&nbsp;&nbsp;&nbsp;&nbsp;In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.<br>
>&nbsp;&nbsp;&nbsp;&nbsp;Suppose that you are the administrator of a university department and you want to determine each applicant's chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant's scores on two exams and the admissions decision.<br>
>&nbsp;&nbsp;&nbsp;&nbsp;Your task is to build a classication model that estimates an applicant's probability of admission based the scores from those two exams. 

<font size=3> 简单来说，在这个练习中，我们需要建立一个逻辑回归模型去预测一个学生是否能被大学录取。现在假设你是一所大学的管理者，并且你可以根据每个申请人的两门成绩去决定他们是否被录取。你还有以前申请人的历史数据，你可以将其作为逻辑回归模型的训练集，对于每个例子，您都有申请人在两项考试中的分数和录取结果。
现在你的任务就是建立一个基于两项考试的分数的分类模型去评估申请人被录取的可能性。

## 1.1 Visualizing the data
<font size=3>建议无论打算用什么算法，如果可能的话，都最好将数据可视化，有时候数据可视化以后，你能更加清晰用什么模型更加好


首先导入包Package
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

再读入数据
```python
path = 'data/ex2data1.txt'
data = pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted']) # 给data设置title
data.head()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129202557989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
接着就画出散点图，⚪ 表示的是Admitted，X 表示的是Not Admitted，一个是正一个是负
```python
positive = data[data['Admitted'].isin([1])] # 1
negative = data[data['Admitted'].isin([0])] # 0

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(x = positive['Exam 1'],y = positive['Exam 2'],s = 50,color = 'b',marker = 'o',label = 'Admitted')
ax.scatter(x = negative['Exam 1'],y = negative['Exam 2'],s = 50,color = 'r',marker = 'x',label = 'Not Addmitted')
plt.legend() # 显示label
ax.set_xlabel('Exam 1 Score') # set x_label
ax.set_ylabel('Exam 2 Score') # set y_label
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021012920325410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
可以粗略的看出来，在这两者之间，可能存在一个决策边界进行分来，接着就可以建立逻辑回归模型解决这个分类模型
## 1.2 Implementation
### 1.2.1 Warmup exercise: sigmoid function
Before you start with the actual cost function, recall that the logistic regression hypothesis is defined as:
$$h_\theta(x)=g(\theta^TX)$$
where function g is the sigmoid function. The sigmoid function is defined as:
$$g(z) = \frac{1}{1+e^{-z}}$$
结合起来，获得逻辑回归的假设：
$$h_\theta(x) =  \frac{1}{1+e^{-\theta^TX}}$$
以上我们回顾了逻辑回归模型里面的sigmod函数，接下来就开始定义他
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
为了让我们更加对sigmod函数有一个更加清晰的认识，可视化一部分

```python
x1 = np.arange(-10, 10, 0.1)
plt.plot(x1, sigmoid(x1), c='r')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129204236945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
### 1.2.2 Cost function
Now you will implement the cost function and gradient for logistic regression.
logistic regression中的cost function与线性回归不同，因为这是一个凸函数 convex function
<img src='https://i.loli.net/2018/12/01/5c01d18b66e49.png' width=400 >
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021012921062434.png)
```python
def cost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta.T))
    second = (1-y) * np.log(1 - sigmoid(X @ theta.T))
    return np.mean(first - second)
```
如果为了得到cost的值，我们还需要对原有的训练集data进行一些操作
```python
# add a ones column - this makes the matrix multiplication work out easier
if 'Ones' not in data.columns:
    data.insert(0,'Ones',1)
    
# set X (training data) and y (target variable)
X = data.iloc[:, :-1] # Convert the frame to its Numpy-array representation.
y = data.iloc[:,-1] # Return is NOT a Numpy-matrix, rather, a Numpy-array.

theta = np.zeros(X.shape[1])

X = np.array(X.values)
y = np.array(y.values)
```
让我们最好检验一下矩阵的维度
```python
X.shape, theta.shape, y.shape 
# ((100, 3), (3,), (100,))
```
一切良好
最好就可以计算初始数据的cost值了
```python
cost(theta, X, y) 
# 0.6931471805599453
```
代价大约是0.6931471805599453
### 1.2.3 Gradient
the gradient of the cost is a vector of the same length as θ where the j<sup>th</sup> element (for j = 0; 1; : : : ; n) is defined as follows:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129205856295.png)
与线性回归的相比，公式没有很大的区别，只是函数改为了sigmod函数
```python
def gradient(theta, X, y):
    return (X.T @ (sigmoid(X @ theta.T) - y))/len(X)
# the gradient of the cost is a vector of the same length as θ where the jth element (for j = 0, 1, . . . , n)
```
```python
gradient(theta, X, y)
# array([ -0.1       , -12.00921659, -11.26284221])
```

### 1.2.4 Learning  θ parameters
<font size=3>现在要试图找出让 $J(\theta)$取得最小值的参数$\theta$。  
<img src='https://i.loli.net/2018/12/01/5c01d34b3ee5e.png' width=400 >
反复更新每个参数，用这个式子减去学习率 α 乘以后面的微分项。求导后得到： 
<img src='https://i.loli.net/2018/12/01/5c01d3aa903c7.png' width=400 >
计算得到等式： 
$$\theta_j := \theta_j - \alpha \frac{1}{m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$$
来它同时更新所有$\theta$的值。 
这个更新规则和之前用来做线性回归梯度下降的式子是一样的， 但是假设的定义发生了变化。即使更新参数的规则看起来基本相同，但由于假设的定义发生了变化，所以逻辑函数的梯度下降，跟线性回归的梯度下降实际上是两个完全不同的东西。

<font size=3>可是如果用代码实现怎么办呢，在exp2.pdf中，一个称为“fminunc”的Octave函数是用来优化函数来计算成本和梯度参数。由于我们使用Python，我们可以用SciPy的“optimize”命名空间来做同样的事情。

<font size=3>这里我们使用的是高级优化算法，运行速度通常远远超过梯度下降。方便快捷。
只需传入cost函数，已经所求的变量theta，和梯度。cost函数定义变量时变量tehta要放在第一个，若cost函数只返回cost，则设置fprime=gradient。
这里使用fimin_tnc或者minimize方法来拟合，minimize中method可以选择不同的算法来计算，其中包括TNC
```python
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
result
#  (array([-25.16131878,   0.20623159,   0.20147149]), 36, 0)
```
下面是第二种方法，结果是一样的
```python
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='TNC', jac=gradient)
res
# help(opt.minimize) 
# res.x  # final_theta
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129211450228.png)
```python
cost(result[0], X, y)
# 0.20349770158947394
```
### 1.2.5 Evaluating logistic regression
After learning the parameters, you can use the model to predict whether a particular student will be admitted. For a student with an Exam 1 score of 45 and an Exam 2 score of 85, you should expect to see an admission probability of 0.776.
<font size=3>我们现在已经学号了参数，我们需要利用这个模型去预测是否能被录取，比如一个Exam1得分为45，而Exam2得分为85的，他被录取的可能性大约是0.776
我们可以测试45分和85分的
```python
# 实现hθ
def hfunc1(theta, X):
    return sigmoid(np.dot(theta.T, X))
hfunc1(result[0],[1,45,85])
```
> 0.7762906256930321

<font size=3>看来大约是77.6%没错，nice
我们定义：
当${{h}_{\theta }}$大于等于0.5时，预测 y=1
当${{h}_{\theta }}$ 小于0.5时，预测 y=0 。
```python
# 定义预测函数
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
```
```python
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
```
>accuracy = 89%

<font size=3>在整个数据集上进行测试，发现我们的accuracy大约达到了89%，还是挺不错的
当然，也可以利用sklearn库来得到准确率

```python
from sklearn.metrics import classification_report
print(classification_report(predictions, y))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129215633524.png)

## 1.3 Decision boundary（决策边界）
在这之中，其实边界是一个线性函数，自变量为$x_{1}$，因变量为$x_{2}$
$X × {\theta=0}$  (this is the line)
$\theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2}  = 0$

```python
x1 = np.linspace(30, 100, 100) # 自变量
x2 = -(final_theta[0] + x1*final_theta[1]) / final_theta[2] # 因变量

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x1, x2, 'y', label='Prediction')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129214233944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
# 2 Regularized logistic regression
>&nbsp;&nbsp;&nbsp;&nbsp;In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.<br>
>&nbsp;&nbsp;&nbsp;&nbsp;Suppose you are the product manager of the factory and you have the test results for some microchips on two dierent tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

<font size=3>在这部分练习中，你将实施正则化逻辑回归来预测来自制造工厂的微芯片是否通过了质量保证(QA)。在QA期间，每个芯片都要经过各种测试，以确保其正常工作。
假设您是工厂的产品经理，在两个不同的测试中有一些微芯片的测试结果。您想通过这两项测试来确定该微芯片应该被接受还是被拒绝。为了帮助您做出决定，您有一个关于过去微芯片的测试结果数据集，您可以根据该数据集建立一个逻辑回归模型。

<font size=3>在这一部分，我们发现，出现了正则化这个词。简而言之，正则化是成本函数中的一个术语，它使算法更倾向于“更简单”的模型（在这种情况下，模型将更小的系数）。这个理论助于减少过拟合，提高模型的泛化能力。
## 2.1 Visualizing the data
一样先读入数据
```python
path = '../data_files/data/ex2data2.txt'
data2 = pd.read_csv(path,header=None,names=['Microchip 1','Microchip 2','Accepted'])
data2.head()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021012922363467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
接着就画出散点图，⚪ 表示的是Accept，**+** 表示的是Reject，一个是正一个是负

```python
def plot_data():
    positive = data2[data2['Accepted'].isin([1])]
    negative = data2[data2['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(x = positive['Microchip 1'],y = positive['Microchip 2'],c = 'black', s = 50,marker = '+',label = 'Accepted')
    ax.scatter(x = negative['Microchip 1'],y = negative['Microchip 2'],c = 'y', s = 50,marker = 'o',label = 'Reject')
    ax.legend()
    ax.set_xlabel('Microchip 1')
    ax.set_ylabel('Microchip 2')
#     plt.show()
plot_data()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129224646505.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
## 2.2 Feature mapping
One way to the data better is to create more features from each data point.We will map the features into all polynomial terms of x1 and x2 up to the sixth power.
<font size=3>一个拟合数据的更好的方法是从每个数据点创建更多的特征。我们将把这些特征映射到所有的x1和x2的多项式项上，直到第六次幂。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129225303288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

```python
def feature_mapping(x1, x2, power,as_ndarray = False):
    data = {}
#     for i in np.arange(power + 1):
#         for p in np.arange(i + 1):
#             data["f{}{}".format(i - p, p)] = np.power(x1, i-p)* np.power(x2,p)
    data = {"f'{}{}".format( i-p , p ):np.power(x1,i-p) * np.power(x2,p)
                    for i in np.arange(power+1)
                    for p in np.arange(i+1)
           }
    if as_ndarray:
        return np.array(pd.DataFrame(data))
    else:
        return pd.DataFrame(data)
```
```python
x1 = np.array(data2['Microchip 1'])
x2 = np.array(data2['Microchip 2'])

_data2 = feature_mapping(x1, x2, power = 6)
print(_data2.shape)
_data2.head()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129230235483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

>As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 28-dimensional vector. A logistic regression classier trained on this higher-dimension feature vector will have a more complex decision boundary and will appear nonlinear when drawn in our 2-dimensional plot.
>While the feature mapping allows us to build a more expressive classier, it also more susceptible to overfitting. In the next parts of the exercise, you will implement regularized logistic regression to the data and also see for yourself how regularization can help combat the overfitting problem.

<font size=3>经过映射，我们的两个特征向量(两个QA测试的分数)被转换成一个28维向量
在这个高维特征向量上训练的logistic回归分类器将具有更复杂的决策边界，并且在我们的二维图中绘制时将出现非线性。
虽然特征映射允许我们建立一个更有表现力的分类器，但它也更容易被过度拟合。在本练习的下一部分中，您将实现对数据的正则化逻辑回归，并亲自了解正则化如何帮助解决过拟合问题。
## 2.3 Cost function 
<font size=3>正则化的基本方法：对高次项添加惩罚值，让高次项的系数接近于0。
假如我们有非常多的特征，我们并不知道其中哪些特征我们要惩罚，我们将对所有的特征进行惩罚，并且让代价函数最优化的软件来选择这些惩罚的程度。这样的结果是得到了一个较为简单的能防止过拟合问题的假设： 
$$
J(\theta) = \frac{1}{2m} [ \sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^{2} + \lambda\sum_{j=1}^n\theta^2_j  ]
$$

<font size=3>其中$\lambda$又称为正则化参数（Regularization Parameter）。 注：根据惯例，我们不对$\theta_0$进行惩罚。
sigmod函数
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
先获取特征，再进行操作
```python
theta = np.zeros(_data2.shape[1])
X = feature_mapping(x1, x2, power = 6,as_ndarray = True)
print(X.shape)

y = np.array(data2.iloc[:,-1])
print(y.shape)
```

```python
def regularized_cost(theta, X, y, l=1):
    thetaReg = theta[1:]
    first = ( -y * np.log(sigmoid(X @ theta) ))  - (1-y) * np.log(1-sigmoid( X @ theta ))
    reg = (thetaReg @ thetaReg) * l / ( 2*len(X) )
    return np.mean(first) + reg

regularized_cost(theta,X,y,l=1)
```
>0.6931471805599454
## 2.4 Regularized Gradient
<img src='http://imgbed.momodel.cn//20190427094457.png' width=600>

<font size=3>注：看上去同线性回归一样，但是由于假设$h_\theta(x)=g(\theta^TX)$，所以与线性回归不同。

注意： 
<font size=3>1. 虽然正则化的逻辑回归中的梯度下降和正则化的线性回归中的表达式看起来一样，但 由于两者的$h_\theta(x)$不同所以还是有很大差别。 
<font size=3>2. $\theta_0$不参与其中的任何一个正则化。 

```python
def regularized_gradient(theta, X, y, l=1):
    thetaReg = theta[1:]
    first = ( X.T @ (sigmoid(X @ theta) - y)) / len(X)
#     print(first)
     # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
#     print(reg)
# [8.47457627e-03 1.87880932e-02 7.77711864e-05 5.03446395e-02
#  1.15013308e-02 3.76648474e-02 1.83559872e-02 7.32393391e-03
#  8.19244468e-03 2.34764889e-02 3.93486234e-02 2.23923907e-03
#  1.28600503e-02 3.09593720e-03 3.93028171e-02 1.99707467e-02
#  4.32983232e-03 3.38643902e-03 5.83822078e-03 4.47629067e-03
#  3.10079849e-02 3.10312442e-02 1.09740238e-03 6.31570797e-03
#  4.08503006e-04 7.26504316e-03 1.37646175e-03 3.87936363e-02]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0.]
    return first + reg
regularized_gradient(theta,X,y)
```

## 补充改进
<font size=3>除了上述可以重新定义函数之外，其实我们可以在原有基础上的cost函数和gradent进行改进就可以了

```python
def cost(theta, X, y):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y)*np.log(1 - sigmoid(X @ theta))
    return np.mean(first - second)
```

```python
def gradient(theta, X, y):
    return (X.T @ (sigmoid(X @ theta) - y))/len(X)  
# the gradient of the cost is a vector of the same length as θ where the jth element (for j = 0, 1, . . . , n)
```

```python
def costReg(theta, X, y, l=1):
    # 不惩罚第一项
    _theta = theta[1: ]
    reg = (l / (2 * len(X))) *(_theta @ _theta)  # _theta@_theta == inner product
    return cost(theta, X, y) + reg
```

```python
def gradientReg(theta, X, y, l=1):
    reg = (1 / len(X)) * theta
    reg[0] = 0  
    return gradient(theta, X, y) + reg
```

## 2.5 Learning  θ parameters
与上一题一样，利用optimize的函数
```python
import scipy.optimize as opt
print('init cost = {}'.format(regularized_cost(theta,X,y)))
#init cost = 0.6931471805599454
res = opt.minimize(fun=regularized_cost,x0=theta,args=(X,y),method='CG',jac=regularized_gradient)
res
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129232924562.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
## 2.6 Evaluating logistic regression

```python
def predict(theta, X):
    probability = sigmoid( X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]  # return a list
```

```python
final_theta = result2[0]
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
accuracy
```
>0.8050847457627118
```python
from sklearn.metrics import classification_report

final_theta = res.x
y_predict = predict(final_theta, X)
predict(final_theta, X)
print(classification_report(y,y_predict))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210129233406789.png)
<font size=3>可以得出大概是0.83左右
除此之外，还可以调用sklearn里的线性回归包
```python
from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y.ravel())
```
```python
model.score(X, y)   # 0.8305084745762712
```
## 2.7 Plotting the decision boundary
$X × {\theta=0}$  (this is the line)
<font size=3>在我们可视化的时候，我们发现，这个函数是不太好求的，我们利用等高线画图，最后将高度设为0，这样的话就可以得到我们的图了，真不错
```python
x = np.linspace(-1, 1.5, 50)
#从-1到1.5等间距取出50个数
xx, yy = np.meshgrid(x, x)
#将x里的数组合成50*50=250个坐标
z = np.array(feature_mapping(xx.ravel(), yy.ravel(), 6))
z = z @ final_theta
z = z.reshape(xx.shape)

plot_data()
plt.contour(xx, yy, z, 0, colors='black')
#等高线是三维图像在二维空间的投影，0表示z的高度为0
plt.ylim(-.8, 1.2)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210130221654207.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)


<font size=3>这样就可视化成功了！！！

好了，真好，我们又完成了exp 2，争取过几天把exp 3也弄出来，加油

<br><br>
每日一句
<font size=3 color=purple>take control of your own desting.（命运掌握在自己手上）



<font size=3>如果需要数据和代码，可以自提
- 路径1：[我的gitee](https://gitee.com/DK-Jun/machine-learning)
- 路径2：百度网盘
 链接：[https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw ](https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw )
 提取码：5605 
