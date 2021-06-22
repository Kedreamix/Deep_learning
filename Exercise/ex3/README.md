# Programming Exercise 3:Multi-class Classfication and Neural Networks 

- [Introduction](#Introduction)
- [1&nbsp;Multi-class&nbsp;Classfication](#1-Multiclass-Classfication)
    - [1.1&nbsp;Dataset](#11-Dataset)
    - [1.2&nbsp;Visualizing&nbsp;the&nbsp;data](#12-Visualizing-the-data)
    - [1.3&nbsp;Vectorizing&nbsp;Logistic&nbsp;Regression](#13-Vectorizing-Logistic-Regression)
        - [1.3.1&nbsp;Vectorizing&nbsp;the&nbsp;regularized&nbsp;cost&nbsp;function](#131-Vectorizing-the-regularized-cost-function)
        - [1.3.2&nbsp;Vectorizing&nbsp;the&nbsp;regularized&nbsp;gradient](#132-Vectorizing-the-regularized-gradient)
    - [1.4&nbsp;One-vs-all&nbsp;Classfication](#14-Onevsall-Classfication)
- [2&nbsp;Neural&nbsp;Networks](#2-Neural-Networks)
    - [2.1&nbsp;Model&nbsp;representation](#21-Model-representation)
    - [2.2&nbsp;Feedforward&nbsp;Propagation&nbsp;and&nbsp;Prediction](#22-Feedforward-Propagation-and-Prediction)



<font size=3>如果想了解更多的知识，可以去我的机器学习之路 The Road To Machine Learning[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)
# Introduction
>&nbsp;&nbsp;&nbsp;&nbsp;In this exercise, you will implement one-vs-all logistic regression and neural networks to recognize hand-written digits. Before starting the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

<font size=3>在这个练习中，我们将实现一对一的逻辑回归和神经网络来识别手写数字
# 1 Multi-class Classfication
>&nbsp;&nbsp;&nbsp;&nbsp;For this exercise, you will use logistic regression and neural networks to recognize handwritten digits (from 0 to 9). Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. This exercise will show you how the methods you've learned can be used for this classfication task.<br>
>&nbsp;&nbsp;&nbsp;&nbsp;In the first part of the exercise, you will extend your previous implemention of logistic regression and apply it to one-vs-all classfication.

<font size=3>在这个练习中，我们会用逻辑回归和神经网络模型来识别手写数字（从0到9）自动识别手写数字现在已经应用的很广泛了
现在我们首先拓展我们在练习2中写的logistic回归的实现，并将其应用于一对多的分类。

导入Package
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
```

## 1.1 Dataset
<font size=3>首先我们需要加载数据集，与之前的不同的是，这是一种mat文件（mat文件是matlab的数据存储的标准格式。mat文件是标准的二进制文件,还可以ASCII码形式保存和加载，在MATLAB中打开显示类似于单行EXCEL表格）
所以文件的读入方式也有所不同

```python
path = '../data_files/data/ex3data1.mat' 
data = loadmat(path)
data
```
<font size=3>首先我们可以看看数据的分布，好对数据进行处理，可以看出来，有X和Y两个数据集，我们再对数据进行一个读入操作
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201120736897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
```python
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y
X, y = load_data('ex3/ex3data1.mat')
X.shape,y.shape,np.unique(y)
'''
((5000, 400),
 (5000, 1),
 array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=uint8))
''' 
```
<font size=3>然后定义了一个load_data 函数，在这个函数分别return了X和y值，这个数据集中有5000个训练样本，每个样本是20*20像素的数字的灰度图像。每个像素代表一个浮点数，表示该位置的灰度强度。20×20的像素网格被展开成一个400维的向量。在我们的数据矩阵X中，每一个样本都变成了一行，这给了我们一个5000×400矩阵X，每一行都是一个手写数字图像的训练样本。
第二部分y其实就是我们的label，是一个5000×1的向量，由于matlab没有index为0的，所以在这一部分中，如果识别为0，label变为了10，其他的1-9还是正常的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201121428292.png)
## 1.2 Visualizing the data
<font size=3>为了使我们更清晰，这里打算可视化一下这个图片，这里利用了matshow函数
首先打印一个数字
```python
def plot_an_image(X):
    '''
    随机打印一个数字
    '''
    pick_one = np.random.randint(0,5000) # 随机选一个数字
    image = X[pick_one,:] # 得到这个数字这一行的向量
    fig,ax = plt.subplots(figsize=(1,1)) 
    ax.matshow(image.reshape((20,20)).T, cmap='gray_r') # reshape成20X20
    plt.xticks([]) # 去掉x刻度线
    plt.yticks([]) # 去掉y刻度线
    plt.show()
    print('this should be {}'.format(y[pick_one]))

plot_an_image(X)
```
<font size=3>会出现图像和对应的label，这里要说的是，原来的图像的pixel正常可视化下来是反的，为了使图像更加清晰，我这里将其转置，就变成了正的了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201121749829.png)
<font size=3>然后再对其进行加工，打印100个数字
```python
def plot_100_image(X):
    '''
    随机打印100个数字
    '''
    sample_idx = np.random.choice(np.arange(X.shape[0]),100)
    sample_images = X[sample_idx, :] # (100,400)
    
    fig,ax_array = plt.subplots(nrows = 10,ncols= 10,sharey = True, sharex = True, figsize=(8,8))
    
    for row in range(10):
        for column in range(10):
            plt.imshow(sample_images[10*row+column].reshape((20,20)).T)
            ax_array[row, column].matshow(sample_images[10*row+column].reshape((20,20)).T,cmap='gray_r')
    plt.xticks([]) # 去掉x刻度线
    plt.yticks([]) # 去掉y刻度线
    plt.show()
plot_100_image(X)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201122112393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 1.3 Vectorizing Logistic Regression
>You will be using multiple one-vs-all logistic regression models to build a multi-class classfier. Since there are 10 classes, you will need to train 10 separate logistic regression classfiers. To make this training efficient, it is important to ensure that your code is well vectorized. In this section, you will implement a vectorized version of logistic regression that does not employ any for loops. You can use your code in the last exercise as a starting point for this exercise.

<font size=3>我们将使用多个one-vs-all(一对多)logistic回归模型来构建一个多类分类器。由于有10个类，需要训练10个独立的分类器。为了提高训练效率，重要的是向量化。在本节中，我们将实现一个不使用任何for循环的向量化的logistic回归版本。
### 1.3.1 Vectorizing the regularized cost function
<font size=3>在逻辑回归中，我们只有一个输出变量，又称标量（scalar），也只有一个因变量 y,逻辑回归问题中代价函数为：
$$
J(\theta) = \frac{1}{2m} [ \sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^{2} + \lambda\sum_{j=1}^n\theta^2_j  ]
$$

<img src='https://i.loli.net/2018/12/01/5c02131965e2b.png' width=500>

<font size=3>这是我们的代价函数，其次我们首先要对每个样本进行计算$h_{\theta}(x^{(i)}),h_{\theta}(x^{(i)}) = g(\theta^TX^{(i)}),g(z) = \frac{1}{1 + e^{-z}}sigmoid函数$
其实我们可以利用矩阵乘法的规则，很快得到，这是X和 $\theta$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201134327802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>如果我们需要得到$h_{\theta}(x^{(i)}) = g(\theta^TX^{(i)})$，那我们可以计算X$\theta$，然后接着，就可以代入sigmoid函数中了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201134335475.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
```python
def regularized_cost(theta, X, y, l):
    thetaReg = theta[1:]
    first = -y * np.log(sigmoid(X @ theta))
    second = -(1 - y) * np.log(1 - sigmoid(X @ theta))
    reg = (thetaReg@thetaReg)*l / (2*len(X))
    return np.mean(first+second) + reg
```
### 1.3.2 Vectorizing the regularized gradient
<font size=3>回顾正则化logistic回归代价函数的梯度下降法如下表示，因为不惩罚theta_0，所以分为两种情况
<img src='http://imgbed.momodel.cn//20190427094457.png' width=600>

```python
def regularized_gradient(theta, X, y, l):
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    reg = np.concatenate([np.array([0]), (l / len(X)) * l *thetaReg])
    return first + reg
```

## 1.4 One-vs-all Classfication
<font size=3>在这一部分我们将实现一对多分类通过训练多个正则化logistic回归分类器，每个对应数据集中K类中的一个。
对于这个任务，我们有10个可能的类，并且由于logistic回归只能一次在2个类之间进行分类，每个分类器在“类别 i”和“不是 i”之间决定。 我们将把分类器训练包含在一个函数中，该函数计算10个分类器中的每个分类器的最终权重，并将权重返回shape为(k, (n+1))数组，其中 n 是参数数量。(记住0的label是10)
在这里我们直接利用了optimize里的minimize来帮我们找到最小值
```python
from scipy.optimize import minimize
def one_vs_all( X, y, l, K):
    all_theta = np.zeros((K, X.shape[1]))
    for i in range(1,K+1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])
        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                       jac=regularized_gradient, options={'disp': True})
        all_theta[i-1,:] = ret.x
        
    return all_theta
```

<font size=3>在这之中，首先，我们为X添加了一列常数项 1 ，以计算截距项（常数项）。 其次，我们将y转化为一维，为了之后更好的判断和对比。
```python
raw_X, raw_y = load_data(path)
X = np.insert(raw_X, 0 ,1,axis=1)
y = raw_y.flatten()

all_theta = one_vs_all(X, y, 1, 10)
all_theta
```
<font size=3>得到了all_theta以后，还要定义一个predict的函数得到预测的值
这里的prob共5000行，10列，每行代表一个样本，每列是预测对应数字的概率。我们取概率最大对应的index加1就是我们分类器最终预测出来的类别。返回的pred_argmax是一个array，包含5000个样本对应的预测值。
```python
def predict_all(X,all_theta):
    '''
    X: (5000,401)
    all_theta(10,401)
    X@all_theta.T (5000,10)
    '''
    #计算每个训练集样本 的 分类概率
    prob = sigmoid(X @ all_theta.T)  #注意这里Theta要转置，因为X为(5000,401) all_theta为(10,401)，形状不匹配
    prob_argmax = np.argmax(prob,axis=1)#axis =1 每行最大值对应的index值 index:{0,1,2,3,4,5,6,7,8,9}
    #i.e 第一行(第一个训练样本)，每列（共10列）都会对第一行样本做出分类，
    #比如{0,1,0,0,0,0,0,0,0,0,} 那么这意味着该样本 属于第二类，第二列的index为1，
    #所以在index基础上加上1 就是 类别的值！ ^_^
    pred_argmax = prob_argmax +1
    return pred_argmax
```
<font size=3>最后我们再进行预测
```python
y_pred = predict_all(X, all_theta)
# print(y_pred.shape)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))
# accuracy = 94.46%
```
<font size=3>为了使结果更加清晰，这里我们调用classification_report
```python
from sklearn.metrics import classification_report
report=classification_report(y_pred,y)
print(report)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201142505540.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
# 2 Neural Networks
>&nbsp;&nbsp;&nbsp;&nbsp;In the previous part of this exercise, you implemented multi-class logistic regression to recognize handwritten digits. However, logistic regression cannot form more complex hypotheses as it is only a linear classifier.<br>
>&nbsp;&nbsp;&nbsp;&nbsp;In this part of the exercise, you will implement a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. For this week, you will be using parameters from a neural network that we have already trained. Your goal is to implement the feedforward propagation algorithm to use our weights for prediction. In next week's exercise, you will write the backpropagation algorithm for learning the neural network parameters.

<font size=3>上一部分中，我们用了多类逻辑回归来识别手写数字，但是逻辑回归不能形成更复杂的假设，因为它只是一个线性分类器。
在这一部分中，我们将使用与前面相同的训练集实现一个神经网络来识别手写数字。神经网络将能够代表形成非线性假设的复杂模型。而且，我们将使用已经训练过的神经网络的参数。主要是为了实现前向传播算法，使用我们的权值进行预测。
## 2.1 Model representation
<font size=3>首先与刚刚一样的，我们需要将数据读入先，并展示了解这个神经网络结构
neural networks
```python
def load_weights(path):
    data = loadmat(path)
    return data['Theta1'],data['Theta2']
theta1,theta2 = load_weights('../data_files/data/ex3weights.mat')
theta1.shape,theta2.shape
# ((25, 401), (10, 26))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020114333625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>这是一个三层的神经网络结构，分别有输入层，隐藏层，输出层，两层直接就有一个权重连接，接下来我们就用前向传播方法实现他
## 2.2 Feedforward Propagation and Prediction
<font size=3>现在就开始实现Feedforward Propagation
让我们recall回顾一下一个例子
假设训练集只有一个实例$(x^{(1)},y^{(1)})$，神经网络是一个四层的神经网络，其中$K=4,S_L=4,L=4$：
前向传播算法： 
 <img src='https://i.loli.net/2018/12/02/5c03dbc252db6.png' width=500>
接着如果是三层的话，那就更简单了呀
首先是第一层到第二层
```python
y = y.flatten()
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
a1 = X
z2 = a1 @ theta1.T
z2.shape
# (5000, 25)
z2 = np.insert(z2,0,1,axis=1)
a2 = sigmoid(z2)
```
<font size=3>然后再从第二层到第三层
```python
z3 = a2 @ theta2.T
z3.shape
# (5000, 10)
a3 = sigmoid(z3)
a3.shape
# (5000, 10)
```
<font size=3>最后我们再预测一下结果

```python
y_pred = np.argmax(a3,axis=1) + 1
report = classification_report(y_pred,y)
print(report)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020114442882.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font szie=3>得到的结果会比单纯的逻辑回归高，虽然人工神经网络是非常强大的模型，但训练数据的准确性并不能完美预测实际数据，有时候很容易就会过拟合


每日一句
<font size=3 color =purple>Learn to walk before you run.（先学走，再学跑）



<font size=3>如果需要数据和代码，可以自提
- 路径1：[我的gitee](https://gitee.com/DK-Jun/machine-learning)
- 路径2：百度网盘
 链接：[https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw ](https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw )
 提取码：5605 
