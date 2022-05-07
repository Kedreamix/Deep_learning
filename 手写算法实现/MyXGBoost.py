import numpy as np
import pandas as pd
#---------------------读入数据-----------------------#
train = pd.read_csv('./premium.csv')
print('Train data shape:',train.shape)

#--------------------离散值数值化--------------------#
categorical_features = ['sex','smoker','region']
# 离散值数值化
for c in range(len(categorical_features)):
    train[categorical_features[c]] = train[categorical_features[c]].astype('category').cat.codes

#--------------------划分训练集以及测试集------------#
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
X, y = train[train.columns.delete(-1)],train['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#--------------------构建树节点---------------------#
class Node:
    def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, 
                 min_child_weight = 1, lambda_ = 1, colsample = 1):
        self.x = x
        self.y = y
        self.grad = grad    # 一阶导数
        self.hess = hess    # 海瑟矩阵 二阶导数
        self.depth = depth  # 每颗树的最大深度
        self.gamma = gamma  # 叶子节点的数目的惩罚系数
        self.lambda_ = lambda_ # 叶子节点的权重的惩罚系数
        self.min_child_weight = min_child_weight # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
        self.colsample = colsample # 随机取一定的特征训练，默认所有特征
        self.cols = np.random.permutation(x.shape[1])[:round(colsample * x.shape[1])]
        self.sim_score = self.similarity_score([True]*x.shape[0])
        self.gain = float("-inf") 
        
        self.split_col = None # 当前结点测试的特征的索引 列
        self.split_row = None # 当前结点测试的特征的索引 行
        self.lhs_tree = None # 左子树
        self.rhs_tree = None # 右子树
        self.pivot = None # 当前结点测试的特征的阈值
        self.val = None # 节点的值
        # making split
        self.split_node()
        
        # 判断是否为叶子节点
        if self.is_leaf:
            self.val = - np.sum(grad) / (np.sum(hess) + lambda_)
        
    
    def split_node(self):
        
        self.find_split()
        
        # 判断是否为也子节点
        if self.is_leaf:
            return
        
        x = self.x[:, self.split_col]

        lhs = x <= x[self.split_row]

        rhs = x > x[self.split_row]
        
        # 递归构建左子树和右子树
        self.lhs_tree = Node(
            self.x[lhs],
            self.y[lhs],
            self.grad[lhs],
            self.hess[lhs],
            depth = self.depth - 1,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample
        )
        
        self.rhs_tree = Node(
            self.x[rhs],
            self.y[rhs],
            self.grad[rhs],
            self.hess[rhs],
            depth = self.depth - 1,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample
        )
    # 寻找最优的分类点
    def find_split(self):
        # 对每个特征都计算
        for c in self.cols:
            x = self.x[:, c] # 去第c个特征
            # 遍历特征c所有可能的值
            for row in range(self.x.shape[0]):
                pivot= x[row] # 利用pivot进行分割
                lhs = x <= pivot
                rhs = x > pivot
                # 计算近似的分数
                sim_lhs = self.similarity_score(lhs)
                sim_rhs = self.similarity_score(rhs)
                
                # 计算信息增益
                gain = sim_lhs + sim_rhs - self.sim_score - self.gamma

                if gain < 0 or self.not_valid_split(lhs) or self.not_valid_split(rhs):
                    continue
                
                # 如果信息增益更高，则进行更新
                if gain > self.gain:
                    self.split_col = c # 选择c特征
                    self.split_row = row 
                    self.pivot = pivot
                    self.gain = gain # 得到的信息
    
    # 判断是否合理 如果是不符合条件 min_child_weigth 就返回False
    def not_valid_split(self, masks):
        if np.sum(self.hess[masks]) < self.min_child_weight:
            return True
        return False
    
    # 判断叶子节点
    @property
    def is_leaf(self):
        # 如果最大深度小于0 或者 信息值未更新，依旧是默认的float('-inf')
        # 说明是叶子节点
        if self.depth < 0 or self.gain == float("-inf"):
            return True
        return False
    
    # 目标函数的最优质的计算，不带正则化项
    def similarity_score(self, masks):
        return  np.sum(self.grad[masks]) ** 2 / ( np.sum(self.hess[masks]) + self.lambda_ )
    
    
    # 计算多个输入
    def predict(self, x):
        return np.array([self.predict_single_val(row) for row in x])
    
    # 计算一个输入
    def predict_single_val(self, x):
        """
        预测样本，沿着树递归搜索
        """
        if self.is_leaf:
            return self.val
        # 递归计算
        return self.lhs_tree.predict_single_val(x) if x[self.split_col] <= self.pivot else self.rhs_tree.predict_single_val(x)

#--------------------构建XGBoost的数，实际上是CART数的原型------------------#
# XGBoost的树
class XGBTree:
    # 参数和前面的节点是部分一样的，重复这里不做阐述
    # subsample 用于训练模型的子样本占整个样本集合的比例。
    # 如果设置为0.5则意味着XGBoost将随机的冲整个样本集合中随机的抽取出
    # 50%的子样本建立树模型，这能够防止过拟合。
    
    # np.random.permutation是随机排列,所以是根据subsample随机取出一定的数进行训练
    def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, 
                 lambda_ = 1, colsample = 1, subsample = 1):
        indices = np.random.permutation(x.shape[0])[:round(subsample * x.shape[0])]
        
        # 定义树的根节点
        self.tree = Node(
            x[indices],
            y[indices],
            grad[indices],
            hess[indices],
            depth = depth,
            gamma = gamma,
            min_child_weight = min_child_weight,
            lambda_ =  lambda_,
            colsample = colsample,
        )
    # 从根节点进行遍历
    def predict(self, x):
        return self.tree.predict(x)
    

# -------------------------回归的XGBoost-------------------------#
class XGBRegressor:
    # eta代表为了防止过拟合，更新过程中用到的收缩步长。缺省为0.3
    # n_estimators 代表建立的数的个数
    # max_depth
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0,
                 min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        self.history = {
            "train" : list(),
            "test" : list()
        }
        
        # 表示所有迭代的树
        self.trees = list()
        
        self.base_pred = None
        
        
    # 进行训练，eval_set代表的数验证集
    def fit(self, x, y, eval_set = None):
        # 判断X的数据格式，全部转为np.ndarray的数据格式
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        # 构建一个y.shape的数组，里面数据全为np.mean(y)
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        # 预测结果
        self.base_pred = np.mean(y)
        # 构建n_estimators
        for n in range(self.n_estimators):
            # 得到一阶梯度和二阶梯度
            grad = self.grad(y, base_pred)
            hess = self.hess(y, base_pred)
            # 构建XGBTree，实际上是CART树的原型
            estimator = XGBTree(
                x,
                y,
                grad,
                hess,
                depth = self.max_depth,
                gamma = self.gamma,
                min_child_weight = self.min_child_weight,
                lambda_ = self.lambda_,
                colsample = self.colsample,
                subsample = self.subsample
            )
            # 加权预测结果
            base_pred = base_pred + self.eta * estimator.predict(x)
            # 将树加入
            self.trees.append(estimator)
            # 如果存在验证集，计算并且打印
            if eval_set:
                X = eval_set[0]
                Y = eval_set[1]
                cost = np.sqrt(np.mean(self.loss(Y, self.predict(X))))
                self.history["test"].append(cost)
                print(f"[{n}] validation_set-rmse : {cost}", end="\t")
            # 得到训练集的值
            cost = np.sqrt(np.mean(self.loss(y, base_pred)))
            self.history["train"].append(cost)
            print(f"[{n}] train_set-rmse : {cost}")
    
    def score(self, x, y):
        # 判断X的数据格式，全部转为np.ndarray的数据格式
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")

        pre = self.predict(x)
        print(pre.shape,y.shape)

        u = np.power((y - pre),2).sum()
        v = np.power(y - y.mean(),2).sum()
        return 1 - u/v
    
    # 利用n_estimators进行预测
    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred
    
    # 利用平方差损失作为目标
    def loss(self, y, a):
        return (y - a)**2
    # 对数据进行求导，得到一阶导数
    def grad(self, y, a):
        # for 0.5 * (y - a)**2
        return a - y
    # 对损失二次求导，得到全为1的数组
    def hess(self, y, a):
        # for 0.5 * (y - a)**2
        return np.full((y.shape), 1)
    
#--------------------------训练以及预测--------------------------#
# 调包的XGBoost
import xgboost
xgb = xgboost.XGBRegressor(learning_rate=0.18,max_depth=3,n_estimators=28,subsample=0.7)
xgb.fit(X_train, y_train, verbose=True, eval_metric="rmse", eval_set=[(X_test, y_test)])

y_pred = xgb.predict(X_test)
eval_res = xgb.evals_result()
print('SCORE:{:.4f}'.format(xgb.score(X_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
print('MSE:{:.4f}'.format(mean_squared_error(y_test,y_pred)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,y_pred)))

# 自己实现XGBoost
myxgb = XGBRegressor(eta=0.18,n_estimators=22,max_depth=3)
myxgb.fit(X_train.values, y_train, eval_set = (X_test.values, y_test))
y_pred = myxgb.predict(X_test.values)

print('SCORE:{:.4f}'.format(myxgb.score(X_test, y_test)))
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
print('MSE:{:.4f}'.format(mean_squared_error(y_test,y_pred)))
print('MAE:{:.4f}'.format(mean_absolute_error(y_test,y_pred)))

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.lineplot(y=eval_res["validation_0"]["rmse"], x=range(len(eval_res["validation_0"]["rmse"])), label="Original XGBoost")
sns.lineplot(y=myxgb.history["test"], x=range(len(myxgb.history["test"])), label="My XGBoost")
plt.show()