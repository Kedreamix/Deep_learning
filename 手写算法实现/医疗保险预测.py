import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./premium.csv')
print('Train data shape:',train.shape)

# 查看数据
print(train.info())
print(train.head()) 

numeric_features = ['age','bmi','children','charges']
categorical_features = ['sex','smoker','region']

# 离散值数值化
for c in range(len(categorical_features)):
    train[categorical_features[c]] = train[categorical_features[c]].astype('category').cat.codes
    

# 得到各个变量的相关性
plt.subplots(figsize=(16,9))
correlation_mat = train.corr()
sns.heatmap(correlation_mat , annot = True)
plt.show()


## 1) 总体分布概况（无界约翰逊分布等）
import scipy.stats as st
y = train['charges']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
plt.show()

plt.hist(train['charges'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()

# log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
plt.hist(np.log(train['charges']), orientation = 'vertical',histtype = 'bar', color ='red') 
plt.show()

#------------------------建模调参-------------------------------------
#取20%的数据作为我们的测试集，其他都是训练集，来对我们的目标进行优化


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
X, y = train[train.columns.delete(-1)],train['charges']#去掉价格那一列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)#分训练集和测试集

#线性回归模型 LinearRegression
from sklearn.linear_model import LinearRegression#导入线性回归模型
linear_model = LinearRegression()#建立模型
linear_model.fit(X_train, y_train)#训练模型，在训练集上
coef = linear_model.coef_#回归系数
linear_pre = linear_model.predict(X_test)#在测试集上预测
print('利用线性回归模型 LinearRegression进行训练......')
print('SCORE:{:.4f}'.format(linear_model.score(X_test, y_test)))#得分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, linear_pre))))
print('MSE:{:.4f}'.format(mean_squared_error(y_test,linear_pre)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,linear_pre)))#判断误差
#支持向量机 利用线性核

from sklearn.svm import SVR
linear_svr = SVR(kernel="linear")
linear_svr.fit(X_train, y_train)
linear_svr_pre = linear_svr.predict(X_test)#预测值

print('利用SVR（线性核）进行训练......')
print('SCORE:{:.4f}'.format(linear_svr.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, linear_svr_pre))))#RMSE(标准误差)
print('MSE:{:.4f}'.format(mean_squared_error(y_test,linear_svr_pre)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,linear_svr_pre)))
#支持向量机 多项式核

poly_svr = SVR(kernel="poly")
poly_svr.fit(X_train, y_train)
poly_svr_pre = poly_svr.predict(X_test)#预测值
print('利用SVR（多项式核）进行训练......')
print('SCORE:{:.4f}'.format(poly_svr.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, poly_svr_pre))))#RMSE(标准误差)
print('MSE:{:.4f}'.format(mean_squared_error(y_test,poly_svr_pre)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,poly_svr_pre)))
#决策树

from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(max_depth=10)
tree_reg.fit(X_train, y_train)
tree_reg_pre = tree_reg.predict(X_test)#预测值
print('利用决策树进行训练......')
print('SCORE:{:.4f}'.format( tree_reg.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,tree_reg_pre))))#RMSE(标准误差)
print('MSE:{:.4f}'.format(mean_squared_error(y_test,tree_reg_pre)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,tree_reg_pre)))
#随机森林 RandomForest

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1,max_depth=1,min_samples_split=30,random_state=0)
forest.fit(X_train,y_train)
forest_pre = tree_reg.predict(X_test)#预测值
print('利用随机森林 RandomForest进行训练......')
print('SCORE:{:.4f}'.format(forest.score(X_test, y_test)))#模型评分
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,forest_pre))))#RMSE(标准误差)
print('MSE:{:.4f}'.format(mean_squared_error(y_test,forest_pre)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,forest_pre)))
