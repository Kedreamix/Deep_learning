

# 预测汽车油耗效率 MPG

<font size=3>这次做一个简单的线性回归的实验，用来预测汽车的油耗效率 MPG，让我们通过这次实验，更加清晰的了解一下LinearRegression，如果想更加清晰的了解的话，可以看看[吴恩达机器学习ex1 Linear Regression (python)](https://blog.csdn.net/weixin_45508265/article/details/112690593)

<font size=3>如果想了解更多的知识，可以去我的机器学习之路 The Road To Machine Learning[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)

[toc]

# Read In Data
<font size=3>我们先读入数据，其中，这里面一个有九列，他们分别都有对应的意义，其中有一列是汽车油耗效率mpg
- mpg - > 燃油效率
- cylinders -> 气缸
- displacement - > 排量
- horsepower - > 马力
- weight - > 重量
- acceleration - > 加速度
- model year - > 型号年份
- origin = > 编号
- car name - > 原产地
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '../data_files/3.MPG/auto-mpg.data'
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
# mpg - > 燃油效率
# cylinders -> 气缸
# displacement - > 排量
# horsepower - > 马力
# weight - > 重量
# acceleration - > 加速度
# model year - > 型号年份
# origin = > 编号
# car name - > 原产地
cars = pd.read_csv(path, delim_whitespace=True, names=columns)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203011104851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
```python
cars.info()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203011123194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>这时候就读完数据了
# 探究数据模型
<font size=3>在我们可视化数据的时候，我们会发现origin和car name都是离散型的，就没有选择他们进行一个线性回归模型的搭建，除此之外，由于在horsepower中，有一些值是存在’？‘我们就要选取那些不是’？‘的进行操作
```python
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
cars = cars[cars.horsepower != '?']
#用散点图分别展示气缸、排量、重量、加速度与燃油效率的关系
fig = plt.figure(figsize=(13,20))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax1.scatter(cars['cylinders'],cars['mpg'],alpha=0.5)
ax1.set_title('cylinders')
ax2.scatter(cars['displacement'],cars['mpg'],alpha=0.5)
ax2.set_title('displacement')
ax3.scatter(cars['weight'],cars['mpg'],alpha=0.5)
ax3.set_title('weight')
ax4.scatter(cars['acceleration'],cars['mpg'],alpha=0.5)
ax4.set_title('acceleration')
ax5.scatter([float(x) for x in cars['horsepower'].tolist()],cars['mpg'],alpha=0.5)
ax5.set_title('horsepower')
```
<font size=3>从下图我们可以看出，汽车的燃油效率mpg与排量displacement、重量weight、马力horsepower三者都存在一定的线性关系，其中汽车重量weight与燃油效率线性关系最为明显，首先我们就利用weight一个单变量去构建线性回归模型，看看是否能预测出来
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203012509668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203012514555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
#  拆分训练集和测试集
```python
Y = cars['mpg']
X = cars[['weight']]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
<font size=3>取数据中的20%作为测试集，其他均为测试集
# 单变量线性回归
## 搭建线性回归模型
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(X_train,Y_train)
```
<font size=3>利用训练集去训练模型
## 可视化结果
<font size=3>利用我们训练完的模型去测试一下我们的训练集和测试集
### 训练集
```python
plt.scatter(X_train, Y_train, color = 'red', alpha=0.3)
plt.scatter(X_train, lr.predict(X_train),color = 'green',alpha=0.3)
plt.xlabel('weight')
plt.ylabel('mpg')
plt.title('train data')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020301373579.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

### 测试集

```python
plt.scatter(X_test,Y_test,color = 'blue',alpha=0.3)
plt.scatter(X_train,lr.predict(X_train),color='green',alpha=0.3)
plt.xlabel('weight')
plt.ylabel('mpg')
plt.title('test data')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203013751310.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 模型评价
```python
print(lr.coef_)
print(lr.intercept_)
print('score = {}'.format(lr.score(X,Y)))
'''
[-0.00772198]
46.43412847740396
score = 0.6925641006507041
'''
```
<font size=3>可以看到，最后的结果的分数大约是0.69左右，还是挺不错的

# 多变量线性回归模型
<font size=3>刚刚我们是利用了单变量的线性回归模型，我们猜测，如果用多变量的线性回归模型会不会更好呢，因为汽车的燃油效率mpg与排量displacement、重量weight、马力horsepower三者都存在一定的线性关系
首先就要重新提取数据，为了对数据更加清晰，我们将预测出来mpg_prediction也加入数据中
```python
cars = cars[cars.horsepower != '?']
mul = ['weight','horsepower','displacement'] # 选择三个变量进行建立模型
mul_lr = LinearRegression()
mul_lr.fit(cars[mul],cars['mpg']) # 训练模型
cars['mpg_prediction'] = mul_lr.predict(cars[mul])
cars.head()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203014424457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

## 模型得分
```python
mul_score = mul_lr.score(cars[mul],cars['mpg'])
mul_score
# 0.7069554693444708
```

<font size=3>从结果可以看出来，这个模型得分大约是71，说明多变量线性回归模型还是比单变量线性回归模型优的，预测的也更加准确一点
```python
from sklearn.metrics import mean_squared_error as mse
mse = mse(cars['mpg'],cars['mpg_prediction'])
print('mse = %f'%mse)
print('rmse = %f'%np.sqrt(mse))
'''
mse = 17.806188
rmse = 4.219738
'''
```
<font size=3>并且得出了MSE和RMSE的值
## 可视化

```python
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.scatter(cars['weight'], cars['mpg'], c='blue', alpha=0.3)
ax1.scatter(cars['weight'], cars['mpg_prediction'], c='red', alpha=0.3)
ax1.set_title('weight')
ax2.scatter([ float(x) for x in cars['horsepower'].tolist()], cars['mpg'], c='blue', alpha=0.3)
ax2.scatter([ float(x) for x in cars['horsepower'].tolist()], cars['mpg_prediction'], c='red', alpha=0.3)
ax2.set_title('horsepower')
ax3.scatter(cars['displacement'], cars['mpg'], c='blue', alpha=0.3)
ax3.scatter(cars['displacement'], cars['mpg_prediction'], c='red', alpha=0.3)
ax3.set_title('displacement')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203014649329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
<font size=3>到这里又成功了，真不错，继续加油

每日一句
<font size=3 color=purple>If you find a path with no obstacles, it probably doesn’t lead anywhere.
太容易的路，可能根本就不能带你去任何地方。

<font size=3>如果需要数据和代码，可以自提
- 路径1：[我的gitee](https://gitee.com/DK-Jun/machine-learning)
- 路径2：百度网盘
 链接：[https://pan.baidu.com/s/1U9dteXf56yo3fQ7b9LETsA ](https://pan.baidu.com/s/1U9dteXf56yo3fQ7b9LETsA)
 提取码：5odf 