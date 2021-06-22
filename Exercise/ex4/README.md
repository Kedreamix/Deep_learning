# Programming Exercise 4:Neural Networks Learning Machine Learning

    
- [Introduction](#Introduction)
- [1&nbsp;Neural&nbsp;Networks](#1-Neural-Networks)
    - [1.1&nbsp;Visualizing&nbsp;the&nbsp;data](#11-Visualizing-the-data)
    - [1.2&nbsp;Model&nbsp;representation](#12-Model-representation)
    - [1.2.1&nbsp;对y进行one-hot编码](#121-对y进行onehot编码)
    - [1.2.2&nbsp;load_weight](#122-loadweight)
    - [1.2.3&nbsp;提取参数和展开参数](#123-提取参数和展开参数)
- [1.3&nbsp;Feedforward&nbsp;and&nbsp;cost&nbsp;function](#13-Feedforward-and-cost-function)
    - [1.3.1&nbsp;Feedforward](#131-Feedforward)
    - [1.3.2&nbsp;Cost&nbsp;Function](#132-Cost-Function)
- [2&nbsp;Backpropagation](#2-Backpropagation)
    - [2.1&nbsp;Sigmoid&nbsp;gradient](#21-Sigmoid-gradient)
    - [2.2&nbsp;Random&nbsp;initialization](#22-Random-initialization)
    - [2.3&nbsp;Backpropagation](#23-Backpropagation)
    - [2.4&nbsp;Gradient&nbsp;checking](#24-Gradient-checking)
    - [2.5&nbsp;Regularized&nbsp;Neural&nbsp;Networks](#25-Regularized-Neural-Networks)
    - [2.6&nbsp;Learning&nbsp;parameters&nbsp;using&nbsp;fmincg](#26-Learning-parameters-using-fmincg)
- [3&nbsp;Visualizing&nbsp;the&nbsp;hidden&nbsp;layer](#3-Visualizing-the-hidden-layer)
    - [3.1&nbsp;Optional&nbsp;(ungraded)&nbsp;exercise](#31-Optional-ungraded-exercise)

<font size=3>如果想了解更多的知识，可以去我的机器学习之路 The Road To Machine Learning[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)
# Introduction
>In this exercise, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.

<font size=3>在这个练习中，我们将实现反向传播算法来学习神经网络的参数，依旧是上次预测手写数数字的例子
# 1 Neural Networks
>In the previous exercise, you implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we provided. In this exercise, you will implement the backpropagation algorithm to learn the parameters for the neural network.

<font size=3>在前面的练习中,我们实现了Feedforward Propagation,并且用来识别数字.在这个练习中,我们将会运用反向传播算法去学习神经网络的参数
由于前面的部分几乎一模一样,就粗略给出代码,详细可以看[吴恩达机器学习ex3](https://editor.csdn.net/md/?articleId=113504698#22_Feedforward_Propagation_and_Prediction_218)

导入Package
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
```
## 1.1 Visualizing the data
```python
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y
X, y = load_data('../data_files/data/ex4data1.mat')
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


<font size=3>为了使我们更清晰，这里打算可视化一下这个图片，这里利用了matshow函数
首先打印100个数字,会出现图像和对应的label，这里要说的是，原来的图像的pixel正常可视化下来是反的，为了使图像更加清晰，我这里将其转置，就变成了正的了

```python
sample_idx = np.random.choice(np.arange(X.shape[0]),100)
sample_images = data['X'][sample_idx, :]
fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T,cmap='gray_r')
        plt.xticks(np.array([]))
        plt.yticks(np.array([])) 
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210201122112393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
## 1.2 Model representation
>Our neural network is shown in Figure 2. It has 3 layers { an input layer,a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20 X 20, this gives us 400 input layer units (not counting the extra bias unit which always outputs +1). Thetraining data will be loaded into the variables X and y by the ex4.m script.

<font size=3>我们的神经网络如图所示,它有三层，分别是输入层，隐藏层，输出层。我们的输入是数字图像的像素值，因为每个数字的图像大小为20*20，所以我们输入层有400个单元（这里不包括总是输出要加一个偏置单元）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226233306164.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
### 1.2.1 对y进行one-hot编码
<font size=3>一开始我们得到的y是$5000 * 1$维的向量，但我们要把他编码成$5000 * 10$的矩阵。
比如说，原始$y_0 = 2$，那么转化后的Y对应行就是[0,1,0...0]，原始$y_1 = 0$转化后的Y对应行就是[0,0...0,1]
Scikitlearn有一个内置的编码函数，我们可以使用这个

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse= False)
y_onehot = encoder.fit_transform(y.reshape(-1,1))
y_onehot.shape
# (5000, 10)
```
<font size=3>我们可以看看编码后的结果
```python
y[0], y_onehot[0,:]
# (10, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]))
y = y_onehot
```

### 1.2.2 load_weight
<font size=3>我们的权重可以从ex4weight.mat文件中读取,这些参数的维度由神经网络的大小决定，第二层有25个单元，输出层有10个单元(对应10个数字类)。
```python
weights = loadmat('../data_files/data/ex4weights.mat')
theta1,theta2 = weights['Theta1'],weights['Theta2']
theta1.shape,theta2.shape
# ((25, 401), (10, 26))
```

### 1.2.3 提取参数和展开参数
<font size=3>当我们使用高级优化方法来优化神经网络时，我们需要将多个参数矩阵展开，才能传入优化函数，然后再恢复形状。

```python
def serialize(a, b):
    '''展开参数'''
    return np.r_[a.flatten(),b.flatten()]
```

```python
theta = serialize(theta1,theta2)
theta.shape  # (10285,)
```

```python
def deserialize(seq):
    '''提取参数'''
    return seq[:25*401].reshape(25, 401), seq[25*401:].reshape(10, 26)
```
## 1.3 Feedforward and cost function
### 1.3.1 Feedforward
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210226235911432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

sigmoid函数
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

```python
X = np.insert(X,0,1,axis=1)
X.shape,y.shape
# ((5000, 401), (5000,))
```
<font size=3>实现前向传播函数
确保每层的单元数，注意输出时加一个偏置单元，s(1)=400+1，s(2)=25+1，s(3)=10
```python
# 前向传播函数
def forward_propagate(theta, X):
    theta1,theta2 = deserialize(theta)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2),0,values=1,axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1,z2,a2,z3,h
```

```python
a1, z2, a2, z3, h = forward_propagate(theta, X)
```

### 1.3.2 Cost Function
<font size=3>回顾一下神经网络的cost function(没有正则项的)
<font size=3>Recall that the cost function for the neural network (without regularization) is
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K[-y_k^{(i)}log((h_\theta(x^{(i)}))_k) - (1 -y_k^{(i)})log(1 - h_\theta(x^{(i)}))_k)]
$$
```python
def cost(theta,X,y):
    a1, z2, a2, z3, h = forward_propagate(theta, X)
    J = 0
    for i in range(len(X)):
        first = - y[i] * np.log(h[i])
        second = (1 - y[i]) * np.log(1 - h[i])
        J = J + np.sum(first - second)
    J = J / len(X)
    return J
```
最后我们使用提供训练好的参数θ，算出的cost应该为0.287629

```python
cost(theta, X, y)
# 0.2876291651613188
```

## 1.4 Regularized cost function
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402225729225.png#pic_center)
**注意不要将每层的偏置项正则化**

```python
def costReg(theta,X,y,learning_rate=1):
    theta1,theta2 = deserialize(theta)
    a1,z2,a2,z3,h = forward_propagate(theta, X)
    J = 0
    reg = np.sum(theta1[:,1:] ** 2) + np.sum(theta2[:,1:] ** 2)
    J = (learning_rate / (2 * len(X))) * reg + cost(theta,X,y)
    return J
costReg(theta,X,y)
```
>0.38376985909092354

You should see that the cost is about 0.383770.
# 2 Backpropagation
## 2.1 Sigmoid gradient
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402230737605.png)
<font size=3>这个可以很简单推一下sigmoid函数的导数

```python
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
```
<font size=3>还可以大概看一下图像的分布
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402231052572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 2.2 Random initialization
![.](https://img-blog.csdnimg.cn/20210402231208351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>当我们训练神经网络的时候，初始化参数是非常重要的，可以用来打破对称性。一个有效的方法就是在均匀分布(−e，e)中随机选择值，我们可以选择 e = 0.12 。这个范围可以保证我们的参数足够小并且使得学习更有效率
```python
def random_init(size):
    '''从服从的均匀分布的范围中随机返回size大小的值'''
    return np.random.uniform(-0.12, 0.12, size)
```

## 2.3 Backpropagation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402231454631.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>现在，我们要实现反向传播算法（backpropagation Algorithms)
首先，我们要弄清楚，每个单元的维度，为之后的实现做准备，有个更清晰的认识

```python
print('a1', a1.shape,'theta1', theta1.shape)
print('z2', z2.shape)
print('a2', a2.shape, 'theta2', theta2.shape)
print('z3', z3.shape)
print('a3', h.shape)
'''
a1 (5000, 401) theta1 (25, 401)
z2 (5000, 25)
a2 (5000, 26) theta2 (10, 26)
z3 (5000, 10)
a3 (5000, 10)
'''
```

```python
def backpropagation(theta,X,y):
    a1,z2,a2,z3,h = forward_propagate(theta, X)
    t1,t2 = deserialize(theta) 
    d3 = h - y # (5000, 10)
    d2 = d3 @ t2[:,1:] * sigmoid_gradient(z2)  # (5000, 25)
    D2 = d3.T @ a2  # (10, 26)
    D1 = d2.T @ a1 # (25, 401)
    D = (1 / len(X)) * serialize(D1, D2)  # (10285,)
    return D
```

## 2.4 Gradient checking
<font size=3>在你的神经网络,你是最小化代价函数J(Θ)。执行梯度检查你的参数,你可以想象展开参数Θ(1)Θ(2)成一个长向量θ。通过这样做,你能使用以下梯度检查过程。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402231839919.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
不过这个代码实现麻烦，运行也很慢，这里先省略，注意是对梯度的验证

## 2.5 Regularized Neural Networks
<font size=3>接着我们需要正则化我们的神经网络，依据以下公式进行计算
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402231929475.png)

```python
def regularized_gradient(theta,X,y,learning_rate=1):
    a1,z2,a2,z3,h = forward_propagate(theta, X)
    D = backpropagation(theta, X, y)
    theta1,theta2 = deserialize(theta)
    D1, D2 = deserialize(backpropagation(theta, X, y))
    theta1[:0] = 0 
    theta2[:0] = 0
    reg_D1 = D1 + (learning_rate / len(X)) * theta1
    reg_D2 = D2 + (learning_rate / len(X)) * theta2
    return serialize(reg_D1,reg_D2)
```

## 2.6 Learning parameters using fmincg
<font size=3>在pdf中，老师说用fmincg，但是这个不是python的，python也有自己的优化函数库，可以去学习参数

```python
def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401 + 10*26

    res = opt.minimize(fun=costReg,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402232135369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>同时我们也进行一下准确率的计算，这个在ex3就有提过
```python
def accuracy(theta, X, y):
    _, _, _, _, h = forward_propagate(res.x, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))
```
<font size=3>可以得到以下结果
```python
accuracy(res.x, X, raw_y)
'''
				precision    recall  f1-score   support

           1       0.99      1.00      0.99       500
           2       0.99      0.99      0.99       500
           3       0.99      0.99      0.99       500
           4       1.00      1.00      1.00       500
           5       1.00      1.00      1.00       500
           6       1.00      1.00      1.00       500
           7       0.99      0.99      0.99       500
           8       1.00      1.00      1.00       500
           9       0.99      0.99      0.99       500
          10       0.99      1.00      1.00       500

    accuracy                           0.99      5000
   macro avg       0.99      0.99      0.99      5000
weighted avg       0.99      0.99      0.99      5000

'''
```

# 3 Visualizing the hidden layer
## 3.1 Optional (ungraded) exercise
<font size=3>一种理解你的神经网络正在学习什么的方法是可视化隐藏单位的表现形式是什么。非正式地,给定一个特定的隐藏单元，一种可视化它所计算的东西的方法是和输入x，使其激活(即具有激活值)。
对于你训练的神经网络，注意第i行是表示第i个参数的401维向量隐藏的单位。如果我们去掉偏见项，我们得到一个400维的向量表示从每个输入像素到隐藏单元的权重。
通过使用displayData来实现这一点函数，它将向你展示一张带有25个单元的图像，每一个都对应着网络中的一个隐藏单元。在你训练过的网络中，你应该知道隐藏的单位是正确的对探测器作出粗略的反应，这些探测器在大脑中寻找笔划和其他图案输入。

```python
def plot_hidden(theta):
    t1, _ = deserialize(theta)
    t1 = t1[:, 1:]
    fig,ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6,6))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()
```

```python
plot_hidden(res.x)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402232742682.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>其实在我们神经网络的隐藏层中，很多是不断的分解，不断的寻找特征，将特征组合起来就是我们的数字。例如在这些隐藏层，可能是一些笔画，比如9这个数字他可以分解为两个特征，因为对于手写数字来说，他上面有一个圈，右边有一个竖，对于8来说，可能就是两个圈，下图展示了9，8，4这三个数字
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402233411858.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
然后在我们隐藏层中，可能就是这一个一个笔画，我们的神经网络就是判断这些神经元哪个占比更大，然后预测出最有可能的那个数字作为我们的概率，这里给出视频链接[什么是神经网络？ ](https://www.youtube.com/watch?v=aircAruvnKk)(有可能登不上Youtube，可以去b站找相关视频)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210402233346208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

每日一句
<font size=3 color =purple>When things are looking down, look up.
当一切都在走下坡路时，往上看。

<font size=3>如果需要数据和代码，可以自提
- 路径1：[我的gitee](https://gitee.com/DK-Jun/machine-learning)
- 路径2：百度网盘
 链接：[https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw ](https://pan.baidu.com/s/1uA5YU06FEW7pW8g9KaHaaw )
 提取码：5605 
