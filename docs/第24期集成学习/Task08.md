# Task08 Bagging 原理与实战

# 1 集成学习

* 根据个体学习器的生成方式，集成学习分为三类学习框架
  * Bagging：**同质**基学习器之间**不存在强依赖**关系，可**并行**化的方法，**随机森林**为其变体
  * Boosting：**同质**个体学习器间存在**强依赖**关系，必须**串行**生成的序列化方法
  * Stacking：**异质**个体学习器**并行**学习，最后通过一个**元模型**结合各个体学习器的结果

## 1.1 Bagging（Bootstrap Aggregating ）

* **工作机制**
  1. Bootstrap（有放回随机采样）产生不同的训练数据集
  2. 分别基于这些训练数据集得到多个基分类器
  3. 对基分类器的分类结果进行结合得到一个相对更优的模型
* **应用场景与结合策略**
  * 回归：简单平均法$\Rightarrow$ 回归预测结果
  * 分类：简单投票法$\Rightarrow$ 分类结果
* **缺点**：不能用单棵树做预测，不知具体是哪个变量有重要作用，即**改进**了**预测准确率**但损失了**解释性**

### 1.1.1 随机森林（Random Forest）

<center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/v2-7e36a623694fda282f1bd6c68f2b4e70_b.webp" alt="" style="zoom: 67%;" /></center>

* **定义**：一种基于if-then-else规则的有监督学习算法；由许多CART决策树（分类回归树）构成，不同决策树之间没有关联
  * 随机森林源码[^1]
* **工作机制**：以决策树作为基学习器的bagging 学习方法
  1. 随机抽样，训练决策树
  2. 随机选取属性，采用某种策略（信息增益，信息增益率，**基尼指数**）做节点分裂属性
  3. 重复步骤2，直到不能再分裂
  4. 建立大量决策树形成森林[^2]
* **应用场景**
  * （离散值）分类
  * （连续值）回归
  * （无监督学习）聚类
  * 异常检测
* **影响随机森林性能的两个因素**
  * 任意两棵树的**相关性**：相关性越大，性能越差；即需要选择最优的特征个数
  * **每棵树的性能**：每棵树性能越强，整个森林的性能越好
* **优点**：（“随机”：抗过拟合，“森林”：准确度高）
  * 训练可以高度**并行化**，对于大数据时代的大样本训练速度有优势
  * 由于可以随机选择决策树节点划分特征，这样在样本特征维度很高的时候，仍然能高效的训练模型
  * 对部分特征缺失不敏感
  * 随机森林的训练效率一般优于bagging，因为随机森林多了一个属性扰动，训练选择最优属性的时候只考察了属性子集而非全集
    * 提高随机森林性能的两大扰动
      * 样本扰动：初始训练集采样
      * 属性扰动：初始属性集采样
  * 不需要进行交叉验证或者额外的测试集进行误差的无偏估计（在生成森林时对误差建立了一个无偏估计）
  * 在训练后，可以给出各个特征对于输出的重要性(**可解释性)**
  * 由于采用了**随机采样**，训练出的模型的方差小，**泛化能力强**
  * 相对于Boosting系列的Adaboost和GBDT， RF实现比较**简单**
* **缺点**：
  * 在某些噪音比较大的样本集上，RF模型容易陷入**过拟合**。
  * 取值划分比较多的**特征**容易对RF的决策产生更大的影响，从而影响拟合的模型的效果

## 1.2 Bagging 与 RF 对比

|            | Bagging  | RF                |
| ---------- | -------- | ----------------- |
| 扰动性     | 样本扰动 | 样本扰动+属性扰动 |
| 结果解释性 | 不可解释 | 可解释            |

# 2 实战练习

## 1 Bagging 理解实例

```python
# 引入所需要的包
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
```

* ####  创建1000各样本20维特征的随机分类数据集

```python
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15,n_redundant=5, random_state=5)
print(X.shape, y.shape)
```

```markdown
(1000, 20) (1000,)
```

* #### 重复分层K-fold 交叉验证评估模型-模型为bagging

```python
# 一共重复3次，每次10fold,评估模型在所有重复交叉验证中性能的平均值和标准差
# define model
model = BaggingClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# repeat performance
print("Accuracy: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))
```

```markdown
Accuracy: 0.867 (0.039)
```

```markdown
函数注解：
* RepeatedStratifiedKFold(n_splits=K, n_repeats=n, random_state):重复分层k折交叉验证器,重复K折n次
    * n_splits: 折数 (>=2)
    * n_repeats: 交叉验证器重复的次数
    * random_state: 随机数种子
    
    
* sklearn.cross_validation.cross_val_score(estimator, X, y=None, scoring=None,cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’)
* 作用：交叉验证
    * estimator:估计方法对象(分类器)
    * X：数据特征(Features)
    * y：数据标签(Labels)
    * soring：调用方法(包括accuracy和mean_squared_error等等)
    * cv：几折交叉验证
    * n_jobs：同时工作的cpu个数（-1代表全部）
    * error_score：如果在估计器拟合中发生错误，要分配给该分数的值（一般不需要指定）
```

* #### 示例-理解RepeatedStratifiedKFold()

```python
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
X = np.array([[1,2], [3, 4], [ 1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
repeat = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=4) # 重复2次2折划分数据
for train_index, test_index in repeat.split(X, y): # split:生成索引以将数据分为训练和测试集
    print("Train:", train_index, "Test：", test_index)
    X_train, X_test = X[train_index], X[test_index]
    print("X_Train","\n", X_train,"\n", "X_Test","\n", X_test)
    y_train, y_test = y[train_index], y[test_index]
    print("y_train","\n", y_train, "\n","y_Test","\n", y_test)
```

```markdown
Train: [0 2] Test： [1 3]
X_Train 
 [[1 2]
 [1 2]] 
 X_Test 
 [[3 4]
 [3 4]]
y_train 
 [0 1] 
 y_Test 
 [0 1]
Train: [1 3] Test： [0 2]
X_Train 
 [[3 4]
 [3 4]] 
 X_Test 
 [[1 2]
 [1 2]]
y_train 
 [0 1] 
 y_Test 
 [0 1]
Train: [1 3] Test： [0 2]
X_Train 
 [[3 4]
 [3 4]] 
 X_Test 
 [[1 2]
 [1 2]]
y_train 
 [0 1] 
 y_Test 
 [0 1]
Train: [0 2] Test： [1 3]
X_Train 
 [[1 2]
 [1 2]] 
 X_Test 
 [[3 4]
 [3 4]]
y_train 
 [0 1] 
 y_Test 
 [0 1]
```

* #### 鸢尾花分类示例-理解cross_val_validation()-模型为SVC

```python
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
SVC = svm.SVC()
scores = cross_val_score(SVC, X, y, cv=5, scoring='accuracy') # 简单5折交叉验证
print("No Repeatkfold Accuracy mean: %.3f (std:%.3f)" %(mean(scores), std(scores)))

# 增加重复分层5折后再交叉验证
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=7,n_repeats=3, random_state=2)
n_scores = cross_val_score(SVC,X=X, y=y,scoring='accuracy', cv=cv, n_jobs=-1)
print("Repeatkfold Accuracy mean: %.3f (std:%.3f)" %(mean(n_scores), std(n_scores)))
```

```markdown
No Repeatkfold Accuracy mean: 0.967 (std:0.021)
Repeatkfold Accuracy mean: 0.969 (std:0.033)
```

## 2 随机森林源码及理解实例

随机森林源码[^1]

* ### python3 版 随机森林源码

```python
import numpy as np
from decision_tree.decision_tree_model import ClassificationTree

Class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    ---------------------------------------------------------------------------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    max_features: int
        每棵树选用数据集中的最大的特征数
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        每棵树中最小的分割数，比如 min_samples_split = 2表示树切到还剩下两个数据集时就停止
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        每棵树切到小于min_gain后停止
        The minimum impurity required to split the tree further.
    max_depth: int
        每棵树的最大层数
        The maximum depth of a tree.
    """
    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features
        
        self.trees = []
        # build forest
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, 
                                      min_impurity=self.min_gain, max_depth=self.max_depth)
            self.trees.append(tree)
        
    def fit(self, X, y):
        # 训练，每棵树使用随机的数据集和随机的特征
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1] # 列数
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features)) # 只取总特征的一半用于训练
        for i in range(self.n_estimators):
            # 生成随机的特征
            # get random feature
            sub_X,sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, repalce=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")
        
    def predict(self, X):
        y_pres = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred
        
    
    def get_bootstrap_data(self, X, Y):
        # 通过bootstrap获得n_estimators组数据
        # get int(n_estimators) datas by bootstrap
        
        m = X.shape[0]
        Y = Y.reshape(m, 1)
        
        # 合并X和Y，方便bootstrap( combine X and Y)
        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)
        
        data_sets = []
        for _ in range(self.n_estimators):
            # 每次随机选取m个索引
            idm = np.random.choice(m, size=m, replace=True) 
            # 获取XY组合数据中索引为 idm 的数据
            bootstrap_X_Y = X_Y(idm, :)
            # 第0行到倒数第二行为X 注意，索引为[0, m)
            bootstrap_X = bootstrap_X_Y[:, :-1]
            # 最后一行为Y
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            # n_estimators 个维度为[样本数，n_features]的列表，即训练集个数为n_estimators
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets
```

注解：[^3]

* numpy.random.choice(a, size=None, replace=True, p=None)
  *  表示：从a(**必须是一维的**ndarray)中随机抽取数字，并组成指定大小(size)的数组
  * replace: True表示可以取相同数字，False表示不可以取相同数字
  * 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

### 实例-理解随机森林-手写数据集分类-可视化分类混淆矩阵

```python
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main():
    data = datasets.load_digits() # 手写数据集
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", y_train.shape)
    
    clf_model = RandomForestClassifier(n_estimators=100)
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)
    
    # classification_report: 用于显示主要分类指标的文本报告，在报告中显示每个类的精确度，召回率，F1值等信息
    # confusion_matrix: 混淆矩阵
    accuracy = metrics.classification_report(y_test, y_pred)
    print("分类指标文本报告:", accuracy)
    
    f, ax= plt.subplots()
    confusion = metrics.confusion_matrix(y_test, y_pred)
    print("混淆矩阵：\n", confusion)
    sns.heatmap(confusion, cmap="tab10",annot=True)
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    
if __name__ == "__main__":
    main()
```

```markdown
X_train.shape: (1078, 64)
Y_train.shape: (1078,)
分类指标文本报告:               precision    recall  f1-score   support

           0       1.00      0.99      0.99        77
           1       0.92      0.97      0.95        75
           2       0.98      1.00      0.99        65
           3       0.99      0.92      0.95        75
           4       0.96      0.94      0.95        69
           5       0.95      0.94      0.94        79
           6       0.99      0.97      0.98        77
           7       0.95      1.00      0.97        71
           8       0.90      0.87      0.89        63
           9       0.93      0.96      0.94        68

    accuracy                           0.96       719
   macro avg       0.96      0.96      0.96       719
weighted avg       0.96      0.96      0.96       719

混淆矩阵：
 [[76  0  0  0  1  0  0  0  0  0]
 [ 0 73  0  0  0  1  1  0  0  0]
 [ 0  0 65  0  0  0  0  0  0  0]
 [ 0  0  0 69  0  1  0  1  4  0]
 [ 0  0  0  0 65  0  0  2  1  1]
 [ 0  0  0  0  1 74  0  0  1  3]
 [ 0  0  0  0  1  1 75  0  0  0]
 [ 0  0  0  0  0  0  0 71  0  0]
 [ 0  6  1  0  0  0  0  0 55  1]
 [ 0  0  0  1  0  1  0  1  0 65]]
```

<center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210415222719520.png" alt="" style="zoom:80%;" /></center>

注解：[^4]
* classification_report(y_true, y_pred)
     * y_true 为样本真实标签，y_pred 为样本预测标签； 
     * support：当前行的类别在测试数据中的样本总量，如上表就是，在class 0 类别在测试集中总数量为1
     * precision：精度=正确预测的个数(TP)/被预测正确的个数(TP+FP)；人话也就是模型预测的结果中有多少是预测正确的
     * recall:召回率=正确预测的个数(TP)/预测个数(TP+FN)；人话也就是某个类别测试集中的总量，有多少样本预测正确了；
     * f1-score:F1 = 2*精度*召回率/(精度+召回率)
     * micro avg：计算所有数据下的指标值，假设全部数据 5 个样本中有 3 个预测正确，所以 micro avg 为 3/5=0.6
     * macro avg：每个类别评估指标未加权的平均值，比如准确率的 macro avg，(0.50+0.00+1.00)/3=0.5
     * weighted avg：加权平均，就是测试集中样本量大的，我认为它更重要，给他设置的权重大点；比如第一个值的计算方法，(0.50*1 + 0.0*1 + 1.0*3)/5 = 0.70

# 3 补充知识

## 基本概念

1. 同质和异质集成
   * 同质集成：同质集成中的个体学习器也叫基学习器（base learner）
   * 异质集成：异质集成中的个体学习器一般称组件学习器（component learner）
   
2. 弱学习器（weak learner）和强学习器：

   * 弱学习器：指泛化性能略优于随机猜测的学习器
   * 强学习器：识别准确率很高并能在多项式时间内完成的学习算法

3. 随机采样：从训练集中采集固定个数的样本，每采集完一个样本后，都将样本放回。在随机采样m次后，m个样本均不同的采样方式

   * 有放回的作用：实现集成学习的”求同”，即避免基分类器都是有偏估计，因为最终需要结合各分类器的结果

   * 随机采样的作用：实现集成学习的“存异”

   以上两点表明：数据集需要不一样，但又要有一定的关联性

4. 分层交叉验证（Stratified k-fold cross validation）：属于交叉验证，分层的意思是说在每一折中都保持着原始数据中**各个类别的比例关系**。比如：原始数据有3类，比例为1:2:1，采用3折分层交叉验证，那么划分的3折中，每一折中的数据类别保持着1:2:1的比例。

   * 作用：避免出现简单交叉验证中可能抽取到数据都是同一类的情况，比如共有3类，第一折全为0类，第二折全为1类等，这将导致模型训练时不能学习到数据的特点

   简单交叉验证与分层交叉验证的对比

   <center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/20200824110344235.png" alt="" style="zoom:67%;" /></center>

# 参考资料

[^1]: https://github.com/RRdmlearning/Machine-Learning-From-Scratch/blob/master/random_forest/random_forest_model.py
[^2]: https://easyai.tech/ai-definition/random-forest/#waht 
[^3]:https://blog.csdn.net/ImwaterP/article/details/96282230
[^4]:https://blog.csdn.net/comway_Li/article/details/102758972