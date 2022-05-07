import numpy as np
import pandas as pd
import math

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

#--------------------构建基分类器决策树--------------#

########-----DecisionTree------#########
class DecisionNode():

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # 当前结点测试的特征的索引
        self.threshold = threshold          # 当前结点测试的特征的阈值
        self.value = value                  # 结点值（如果结点为叶子结点）
        self.true_branch = true_branch      # 左子树（满足阈值， 将特征值大于等于切分点值的数据划分为左子树）
        self.false_branch = false_branch    # 右子树（未满足阈值， 将特征值小于切分点值的数据划分为右子树）

def divide_on_feature(X, feature_i, threshold):
    """
    依据切分变量和切分点，将数据集分为两个子区域
    """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    return np.array([X_1, X_2])

class DecisionTree(object):
    
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # 根结点
        self.min_samples_split = min_samples_split  # 满足切分的最少样本数
        self.min_impurity = min_impurity  # 满足切分的最小纯度
        self.max_depth = max_depth  # 树的最大深度
        self._impurity_calculation = None  # 计算纯度的函数，如对于分类树采用信息增益
        self._leaf_value_calculation = None  # 计算y在叶子结点值的函数
        self.one_dim = None  # y是否为one-hot编码 (n,) -> (n,1)
    
    # 训练X,y数据
    def fit(self, X, y):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """
        递归方法建立决策树
        """
        largest_impurity = 0
        best_criteria = None    # 当前最优分类的特征索引和阈值
        best_sets = None        # 数据子集

        # 转换维度
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        
        Xy = np.concatenate((X,y),axis=1)
        n_samples, n_features = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 对每个特征计算纯度
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values) # 不同的值

                # 遍历特征i所有的可能值找到最优纯度
                for threshold in unique_values:
                    # 基于X在特征i处是否满足阈值来划分X和y， Xy1为满足阈值的子集
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # 取出Xy中y的集合
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # 计算纯度
                        impurity = self._impurity_calculation(y, y1, y2)

                        # 如果纯度更高，则更新
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X的左子树
                                "lefty": Xy1[:, n_features:],   # y的左子树
                                "rightX": Xy2[:, :n_features],  # X的右子树
                                "righty": Xy2[:, n_features:]   # y的右子树
                                }
        if largest_impurity > self.min_impurity:
            # 建立左子树和右子树
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # 如果是叶结点则计算值
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)
    
    def predict_value(self, x, tree=None):
        """
        预测样本，沿着树递归搜索
        """
        # 根结点
        if tree is None:
            tree = self.root

        # 递归出口
        if tree.value is not None:
            return tree.value

        # 选择当前结点的特征
        feature_value = x[tree.feature_i]
        
        # 默认右子树
        branch = tree.false_branch
        
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    # 预测X值
    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred
    
    # 定义得分
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

#-----------------回归决策树------------------------
def calculate_mse(y):
    return np.mean((y - np.mean(y)) ** 2)


def calculate_variance(y):
    n_samples = np.shape(y)[0]
    variance = (1 / n_samples) * np.diag((y - np.mean(y)).T.dot(y - np.mean(y)))
    return variance


class RegressionTree(DecisionTree):
    """
    回归树，在决策书节点选择计算MSE/方差降低，在叶子节点选择均值。
    """
    def _calculate_mse(self, y, y1, y2):
        """
        计算MSE降低
        """
        mse_tot = calculate_mse(y)
        mse_1 = calculate_mse(y1)
        mse_2 = calculate_mse(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        mse_reduction = mse_tot - (frac_1 * mse_1 + frac_2 * mse_2)
        return mse_reduction
    
    def _calculate_variance_reduction(self, y, y1, y2):
        """
        计算方差降低
        """
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        """
        计算均值
        """
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_mse
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)
    
from abc import ABC, abstractmethod
#####----定义损失----#######
class Loss(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod    
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    @abstractmethod    
    def grad(self, y, y_pred):
        raise NotImplementedError()
    
# grad 为一阶导数，hess为海瑟矩阵
class SquareLoss(Loss):
    
    def __init__(self): 
        pass

    def loss(self, y, y_pred):
        pass

    def grad(self, y, y_pred):
        return -(y - y_pred)
    
    def hess(self, y, y_pred):
        return np.ones_like(y)
    
# 交叉熵损失
class CrossEntropyLoss(Loss):
    
    def __init__(self): 
        pass

    def loss(self, y, y_pred):
        pass

    def grad(self, y, y_pred):
        return - (y - y_pred)  
    
    def hess(self, y, y_pred):
        return y_pred * (1-y_pred)

#-------------------构建xgboost---------------------#
#####----XGBoost----#######
class XGBoostRegressionTree(DecisionTree):
    """
    XGBoost 回归树。此处基于第五章介绍的决策树，故采用贪心算法找到特征上分裂点 (枚举特征上所有可能的分裂点)。
    """
    
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None, gamma=0., lambd=0.):
        super(XGBoostRegressionTree, self).__init__(min_impurity=min_impurity, 
            min_samples_split=min_samples_split, 
            max_depth=max_depth)
        self.gamma = gamma   # 叶子节点的数目的惩罚系数
        self.lambd = lambd   # 叶子节点的权重的惩罚系数
        self.loss = loss     # 损失函数
    
    # 分离y_true 和 y_pred
    def _split(self, y):
        # y 包含 y_true 在左半列，y_pred 在右半列
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred
    
   
    def _gain(self, y, y_pred):
        # 计算信息
        nominator = np.power((y * self.loss.grad(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return nominator / (denominator + self.lambd)
    
    # 得到分裂后的增益
    def _gain_by_taylor(self, y, y1, y2):
        # 分割为左子树和右子树
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        # 计算信息增益
        return 0.5 * (true_gain + false_gain - gain) - self.gamma
    
    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        # 计算叶节点权重
        gradient = self.loss.grad(y, y_pred).sum() # G 为一阶导数的和
        hessian = self.loss.hess(y, y_pred).sum() # H 为二阶导数的和
        leaf_approximation = -gradient / (hessian + self.lambd)
         # 使目标函数最小，令其导数为0，解得节点的最优值
        return leaf_approximation
    
    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)
        
import progressbar
# 进度条
bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]


def to_categorical(x, n_classes=None):
    """
    One-hot编码
    """
    if not n_classes:
        n_classes = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_classes))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def line_search(self, y, y_pred, h_pred):
    Lp = 2 * np.sum((y - y_pred) * h_pred)
    Lpp = np.sum(h_pred * h_pred)
    return 1 if np.sum(Lpp) == 0 else Lp / Lpp

''
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class XGBoost(object):
    """
    XGBoost学习器。
    """
    def __init__(self, n_estimators=30, learning_rate=0.18, min_samples_split=2,
                 min_impurity=1e-7, max_depth=3, is_regression=False, gamma=0., lambd=0.):
        self.n_estimators = n_estimators            # 树的数目，迭代次数
        self.learning_rate = learning_rate          # 训练过程中沿着负梯度走的步长，也就是学习率
        self.min_samples_split = min_samples_split  # 分割所需的最小样本数
        self.min_impurity = min_impurity            # 分割所需的最小纯度
        self.max_depth = max_depth                  # 树的最大深度
        self.gamma = gamma                          # 叶子节点的数目的惩罚系数
        self.lambd = lambd                          # 叶子节点的权重的惩罚系数
        self.is_regression = is_regression          # 分类或回归问题
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        # 回归问题采用基础的平方损失，分类问题采用交叉熵损失
        self.loss = SquareLoss()
        if not self.is_regression:
            self.loss = CrossEntropyLoss()
    
    def fit(self, X, Y):
        # 分类问题将 Y 转化为 one-hot 编码
        if not self.is_regression:
            Y = to_categorical(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        self.out_dims = Y.shape[1]
        self.trees = np.empty((self.n_estimators, self.out_dims), dtype=object)
        Y_pred = np.zeros(np.shape(Y))
        self.weights = np.ones((self.n_estimators, self.out_dims))
        self.weights[1:, :] *= self.learning_rate
        # 迭代过程
        for i in self.progressbar(range(self.n_estimators)):
            for c in range(self.out_dims):
                tree = XGBoostRegressionTree(
                        min_samples_split=self.min_samples_split,
                        min_impurity=self.min_impurity,
                        max_depth=self.max_depth,
                        loss=self.loss,
                        gamma=self.gamma,
                        lambd=self.lambd)
                # 计算损失的梯度，并用梯度进行训练
                if not self.is_regression:   
                    Y_hat = softmax(Y_pred)
                    y, y_pred = Y[:, c], Y_hat[:, c]
                else:
                    y, y_pred = Y[:, c], Y_pred[:, c]

                y, y_pred = y.reshape(-1, 1), y_pred.reshape(-1, 1)
                y_and_ypred = np.concatenate((y, y_pred), axis=1)
                tree.fit(X, y_and_ypred)
                # 用新的基学习器进行预测
                h_pred = tree.predict(X)
                # 加法模型中添加基学习器的预测，得到最新迭代下的加法模型预测
                Y_pred[:, c] += np.multiply(self.weights[i, c], h_pred)
                self.trees[i, c] = tree
                
    def predict(self, X):
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        # 生成预测
        for c in range(self.out_dims):
            y_pred = np.array([])
            for i in range(self.n_estimators):
                update = np.multiply(self.weights[i, c], self.trees[i, c].predict(X))
                y_pred = update if not y_pred.any() else y_pred + update
            Y_pred[:, c] = y_pred
        if not self.is_regression: 
            # 分类问题输出最可能类别
            Y_pred = Y_pred.argmax(axis=1)
        return Y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy
    
class XGBRegressor(XGBoost):
    
    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2,
                 min_impurity=1e-7, max_depth=float("inf"), is_regression=True,
                 gamma=0., lambd=0.):
        super(XGBRegressor, self).__init__(n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            min_samples_split=min_samples_split, 
            min_impurity=min_impurity,
            max_depth=max_depth,
            is_regression=is_regression,
            gamma=gamma,
            lambd=lambd)
        
xgb = XGBRegressor(n_estimators=30,learning_rate=0.18,max_depth=3)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
xgb.fit(X_train,y_train)
pre = xgb.predict(X_test)
mae = np.sqrt(mean_absolute_error(pre,y_test))
rmse = np.sqrt(mean_squared_error(y_test,pre))

print("mae = ",mae)
print("rmse = ",rmse)
def score(pre,true):
    u = ((true - pre)**2).sum()
    v = ((true - true.mean())**2).sum()
    return 1 - u/v
print("score = ",score(pre,y_test))
# print(xgb.score(X_test,y_test))
