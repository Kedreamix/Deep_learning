

# 手写数字识别 Digit Recognizer

<font size=3>在这次Machine Learning中，我做一个比较经典的手写数字识别的一个项目，巩固一下自己所学的知识，也带领大家进入神经网络的时代，神经网络可以在这个分类任务上大展身手，万物皆可卷积。

<font size=3>如果想了解更多的知识，可以去我的机器学习之路 The Road To Machine Learning[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)- [手写数字识别 Digit Recognizer](#手写数字识别-Digit-Recognizer)
- [OverView 项目概述](#OverView-项目概述)
- [Data Description 数据描述](#Data-Description-数据描述)
- [1. Introduction 项目介绍](#1-Introduction-项目介绍)
- [2. Data preparation 数据预处理](#2-Data-preparation-数据预处理)
	- [2.1 Load data 加载数据](#21-Load-data-加载数据)
	
	- [2.2 Check for null and missing values 缺失值处理](#22-Check-for-null-and-missing-values-缺失值处理)
	
	- [2.3 Normalization 标准化处理](#23-Normalization-标准化处理)
	
	- [2.4 Reshape 重塑](#24-Reshape-重塑)
	
	- [2.5 Label encoding 标签编码](#25-Label-encoding-标签编码)
	
	- [2.6 Split training and valdiation set 拆分训练和验证集](#26-Split-training-and-valdiation-set-拆分训练和验证集)
- [3. CNN 卷积神经网络](#3-CNN-卷积神经网络)
	- [3.1 Define the model 定义模型](#31-Define-the-model-定义模型)
	
	- [3.2 Set the optimizer and annealer 设置优化器和退火器](#32-Set-the-optimizer-and-annealer-设置优化器和退火器)
	
	- [3.3 Data augmentation 数据增强](#33-Data-augmentation-数据增强)
- [4. Evaluate the model 评估模型](#4-Evaluate-the-model-评估模型)
	- [4.1 Training and validation curves 训练和验证集的学习曲线](#41-Training-and-validation-curves-训练和验证集的学习曲线)
	
	- [4.2 Confusion matrix 混淆矩阵](#42-Confusion-matrix-混淆矩阵)
- [5. Prediction and submition 预测和提交](#5-Prediction-and-submition-预测和提交)
	- [5.1 Predict and Submit results 预测并提交结果](#51-Predict-and-Submit-results-预测并提交结果)

# OverView 项目概述

首先我们就来看看这个识别任务的简单介绍吧
>MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.<br>
>In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

MNIST (Modified National Institute of Standards and Technology)实际上是计算机视觉的“hello world”数据集。自从1999年发布以来，这个经典的手写图像数据集一直是分类算法的基准。随着新的机器学习技术的出现，MNIST仍然是研究人员和学习者的可靠资源。
在这场比赛中，您的目标是从数万张手写图像的数据集中正确识别数字。我们策划了一套教程风格的内核，涵盖了从回归到神经网络的所有内容。我们鼓励您使用不同的算法进行实验，以第一手了解哪些算法更有效以及如何比较技术。

# Data Description 数据描述
MNIST 包括6万张28x28的训练样本，1万张测试样本，很多教程都会对它”下手”几乎成为一个 “典范”，可以说它就是计算机视觉里面的Hello World。所以我们这里也会使用MNIST来进行实战。
在卷积神经网络中，首先出现了一个很强大的LeNet-5，LeNet-5之所以强大就是因为在当时的环境下将MNIST数据的识别率提高到了99%，这里我们也自己从头搭建一个卷积神经网络，也达到99%的准确率


# 1. Introduction 项目介绍
This Notebook follows three main parts:
- The data preparation
- The CNN modeling and evaluation
- The results prediction and submission
305 / 5000

这是一个 5 层顺序卷积神经网络，用于在 MNIST 数据集上训练的数字识别。 我选择使用非常直观的 keras API（Tensorflow 后端）来构建它。 首先，我将准备数据（手写数字图像），然后我将专注于 CNN 建模和评估。 

我在单个 GPU (i7 1050Ti) 上用 训练的 CNN 达到了 99.4%的准确率。 对于那些拥有多核 GPU 能力的人，您可以将 tensorflow-gpu 与 keras 结合使用。 计算速度会快很多！！！

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
%matplotlib inline
```
# 2. Data preparation 数据预处理
## 2.1 Load data 加载数据

```python
# Load the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
```

我们可以查看一下数据的分布和类型等
```python
train.info()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603171045852.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603171112935.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
我们可以看到，数据中有标签label和784(28*28)个pixel，也就是我们的像素值

```python
Y_train = train['label'] 

X_train = train.drop('label',axis=1)

sns.countplot(y_train)
y_train.value_counts()
```
>1    4684
>7    4401
>3    4351
>9    4188
>2    4177
>6    4137
>0    4132
>4    4072
>8    4063
>5    3795
>Name: label, dtype: int64
>![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603171329835.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

We have similar counts for the 10 digits.

我们差不多有十个标签，数目差不多，就是0-9，我们的数字标签

## 2.2 Check for null and missing values 缺失值处理

```python
# Check the data
X_train.isnull().any().describe()
```
>count       784
>unique        1
>top       False
>freq        784
>dtype: object

```python
test.isnull().any().describe()
```
>count       784
>unique        1
>top       False
>freq        784
>dtype: object

There is no missing values in the train and test dataset. So we can safely go ahead
训练和测试数据集中没有缺失值。 所以我们可以放心地继续前进

## 2.3 Normalization 标准化处理
我们实现灰度归一化以减少光照差异的影响，也就是标准化。

此外，CNN 在 [0..1] 数据上的收敛速度比在 [0..255] 上更快

```python
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
```

## 2.4 Reshape 重塑

```python
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
# 也就是将 784 像素的向量重塑为 28x28x3 的 3D 矩阵。
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
```
训练和测试图像 (28px x 28px) 已作为 784 个值的一维向量存入 pandas.Dataframe。 我们将所有数据重塑为 28x28x1 3D 矩阵。

Keras 最后需要一个与通道相对应的额外维度。 MNIST 图像是灰度化的，所以它只使用一个通道。 对于 RGB 图像，有 3 个通道，我们会将 784 像素的向量重塑为 28x28x3 的 3D 矩阵。

## 2.5 Label encoding 标签编码
我们利用我们one-hot编码的原理来对我们的标签进行编码
```python
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = to_categorical(y_train, num_classes = 10)
```
Labels are 10 digits numbers from 0 to 9. We need to encode these lables to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0]).

标签是从 0 到 9 的 10 位数字。我们需要将标签编码为一个one-hot 向量（例如：2 -> [0,0,1,0,0,0,0,0,0,0]）

## 2.6 Split training and valdiation set 拆分训练和验证集
设置随机种子
```python
# Set the random seed
random_seed = 2
```
拆分为训练集和验证集进行一个训练
```python
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)
```
I choosed to split the train set in two parts : a small fraction (10%) became the validation set which the model is evaluated and the rest (90%) is used to train the model.

Since we have 42 000 training images of balanced labels (see 2.1 Load data), a random split of the train set doesn't cause some labels to be over represented in the validation set. Be carefull with some unbalanced dataset a simple random split could cause inaccurate evaluation during the validation.

我选择将训练集分成两部分：一小部分 (10%) 成为评估模型的验证集，其余 (90%) 用于训练模型。

由于我们有 42 000 张平衡标签的训练图像（参见 2.1 加载数据），训练集的随机拆分不会导致某些标签在验证集中过度表示。 小心一些不平衡的数据集，简单的随机拆分可能会导致验证过程中的评估不准确。

We can get a better sense for one of these examples by visualising the image and looking at the label.
我们可以可视化一下我们的图片
```python
plt.imshow(X_train[0][:,:,0])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210603175350773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
# 3. CNN 卷积神经网络
## 3.1 Define the model 定义模型
我使用了 Keras Sequential API，您只需从输入开始，一次添加一层。

第一个是卷积（Conv2D）层。它就像一组可学习的过滤器。我选择为前两个 conv2D 层设置 32 个过滤器，为最后两个层设置 64 个过滤器。每个过滤器使用内核过滤器转换图像的一部分（由内核大小定义）。内核滤波器矩阵应用于整个图像。过滤器可以看作是图像的一种变换。

CNN 可以从这些转换后的图像（特征图）中分离出处处有用的特征。

CNN 中的第二个重要层是池化 (MaxPool2D) 层。该层仅充当下采样滤波器。它查看 2 个相邻像素并选择最大值。这些用于降低计算成本，并在一定程度上减少过拟合。我们必须选择池化大小（即每次池化的区域大小），池化维度越高，下采样越重要。

结合卷积层和池化层，CNN 能够结合局部特征并学习图像的更多全局特征。

Dropout 是一种正则化方法，对于每个训练样本，层中一定比例的节点被随机忽略（将它们的权重设置为零）。这会随机丢弃网络的一部分，并强制网络以分布式方式学习特征。这种技术还提高了泛化能力并减少了过拟合。

'relu' 是激活函数 max(0,x)。激活函数用于为网络添加非线性。

Flatten 层用于将最终的特征图转换为一个单一的一维向量。 需要此展平步骤，以便您可以在一些卷积/最大池层之后使用完全连接的层。 它结合了先前卷积层的所有找到的局部特征。

最后，我使用了两个完全连接（密集）层中的特征，这只是人工神经网络（ANN）分类器。 在最后一层 (Dense(10,activation="softmax")) 中，每个类别的概率的净输出分布。

```python
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
```
## 3.2 Set the optimizer and annealer 设置优化器和退火器

一旦我们的层被添加到模型中，我们需要设置一个评分函数、一个损失函数和一个优化算法。

我们定义了损失函数来衡量我们的模型在具有已知标签的图像上的表现有多差。它是观察到的标签和预测的标签之间的错误率。我们使用称为“categorical_crossentropy”的分类分类（> 2 个类别）的特定形式。

最重要的功能是优化器。该函数将迭代地改进参数（过滤核值、神经元的权重和偏差......）以最小化损失。

我选择了 RMSprop（带有默认值），它是一个非常有效的优化器。 RMSProp 更新以非常简单的方式调整 Adagrad 方法，以尝试降低其激进的、单调递减的学习率。我们也可以使用随机梯度下降 ('SGD') 优化器，但它比 RMSprop 慢。

度量函数“准确度”用于评估我们模型的性能。该度量函数类似于损失函数，不同之处在于在训练模型时不使用度量评估的结果（仅用于评估）。

```python
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
```

```python
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
```

为了使优化器收敛得更快并且最接近损失函数的全局最小值，我使用了学习率（LR）的退火方法。

LR 是优化器遍历“损失情况”的步骤。 LR越高，步长越大，收敛越快。 然而，高 LR 的采样非常差，优化器可能会陷入局部最小值。

最好在训练期间降低学习率，以有效地达到损失函数的全局最小值。

为了保持具有高 LR 的快速计算时间的优势，我根据是否有必要（当精度未提高时）每 X 步（时期）动态地减少 LR。

使用 Keras.callbacks 的 ReduceLROnPlateau 函数，如果 3 个 epoch 后精度没有提高，我选择将 LR 减少一半。 

```python
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

```python
epochs = 30
batch_size = 86
```


## 3.3 Data augmentation 数据增强
为了避免过拟合问题，我们需要人为地扩展我们的手写数字数据集。我们可以使您现有的数据集更大。这个想法是通过小的转换来改变训练数据，以重现有人写数字时发生的变化。

比如数字不居中 比例不一样（有的写大/小数字） 图像旋转...

以改变数组表示同时保持标签不变的方式改变训练数据的方法被称为数据增强技术。人们使用的一些流行的增强是灰度、水平翻转、垂直翻转、随机裁剪、颜色抖动、平移、旋转等等。

通过对我们的训练数据应用这些转换中的几个，我们可以轻松地将训练示例的数量增加一倍或三倍，并创建一个非常强大的模型。

```python
# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
```
对于数据增强，我选择：
- 将一些训练图像随机旋转 10 度
- 将一些训练图像随机放大 10%
- 将图像水平随机移动 10% 的宽度
- 将图像垂直随机移动 10% 的高度

我没有应用vertical_flip 或horizontal_flip，因为它可能导致错误分类对称数，例如6 和9。
一旦我们的模型准备就绪，我们就会拟合训练数据集。

```python
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
```

# 4. Evaluate the model 评估模型
## 4.1 Training and validation curves 训练和验证集的学习曲线

```python
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
```


## 4.2 Confusion matrix 混淆矩阵
混淆矩阵对于查看我们的模型缺点非常有帮助。
我绘制了验证结果的混淆矩阵。

```python
# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605010309747.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
在这里我们可以看到，考虑到验证集的大小（4 200 张图像），我们的 CNN 在所有数字上都表现得非常好，几乎没有错误。

然而，我们的 CNN 似乎对 4 位数字有一些小问题，嘿嘿被误分类为 9。有时在曲线平滑时很难捕捉 4 和 9 之间的差异。

```python
# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1


# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605010331420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
最重要的错误也是最有趣的。

对于这六个案例，该模型并不可笑。 其中一些错误也可能是人类造成的，尤其是对于非常接近 4 的 9。最后一个 9 也非常具有误导性，对我来说似乎是 0
# 5. Prediction and submition 预测和提交
## 5.1 Predict and Submit results 预测并提交结果
```python
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
```

```python
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210605012149589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)