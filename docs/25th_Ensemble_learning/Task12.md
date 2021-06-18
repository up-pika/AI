# Task12 Blending 算法与实战

# 1 集成学习/模型融合回顾

## 1.1 集成学习类型

1. 简单加权融合

- 回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）
- 分类：投票（Voting）
- 综合：排序融合(Rank averaging)，log融合

2. stacking/blending

- 构建多层模型，并利用预测结果再拟合预测

3. boosting/bagging

- 多树的提升方法，boosting(xgboost，Adaboost, GBDT )/  bagging(RF)

本节主要讲blending，其余知识请翻阅往期或后期博客。

# 2 Blending

* 核心：简化版的stacking
* **基本思想：**Blending采用了和stacking同样的方法，不过只从训练集中选择一个fold的结果，再和原始特征进行concat作为元学习器meta learner的特征，在测试集上两者进行同样的操作
* 理解实例：
  * 把原始的训练集先分成两部分，如70%的数据作为新的训练集，剩下30%的数据作为测试集
  * 第一层，在70%的数据上训练多个模型，然后去预测剩下30%数据的label，同时也预测test测试集的label
  * 在第二层，直接用上面30%数据在第一层预测的结果做为新特征继续训练模型，然后用test测试集第一层预测的label做特征，用第二层训练的模型做进一步预测

- 训练过程

  1. 将数据划分为训练集和测试集(test_set)，其中训练集需要再次划分为训练集 (train_set)和验证集 (val_set)
  2. 创建第一层的多个模型，这些模型可同质也可异质
  3. 使用 train_set 训练步骤 2 中的多个模型，然后用训练好的模型预测验证集 val_set 和测试集 test_set 得到预测结果val_predict 和 test_predict
  4. 创建第二层的模型（元学习器，meta learner）,使用 val_predict 作为meta learner的训练集，进行meta learner 的训练
  5. 使用第二层训练好的模型对第二层测试集 test_predict 进行预测，该结果为整个测试集的结果

![blending集成](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/2.png)

图片来源[^1]


# 3 实战练习

### 3.1 Blending

```python
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
import seaborn as sns

# 创建数据
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

data, target = make_blobs(n_samples=10000, centers=2, random_state=1, cluster_std=1.0)

## 划分训练集和测试集
X_train_origin, X_test, y_train_origin, y_test = train_test_split(data, target, 
                                                                  test_size=0.2, random_state=2021)

## 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_origin, 
                                                   y_train_origin, 
                                                   test_size=0.3, 
                                                   random_state=2021)

print("The shape of training X:", X_train.shape)
print("The shape of training y:", y_train.shape)
print("The shape of test X:", X_test.shape)
print("The shape of test y:", y_test.shape)
print("The shape of validation X:", X_val.shape)
print("The shape of validation y:", y_val.shape)
```

```markdown
output:
The shape of training X: (5600, 2)
The shape of training y: (5600,)
The shape of test X: (2000, 2)
The shape of test y: (2000,)
The shape of validation X: (2400, 2)
The shape of validation y: (2400,)
```

```python
## 设置第一层分类器
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clfs = [SVC(probability=True), 
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        KNeighborsClassifier()]

## 设置第二层分类器
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

## 输出第一层的验证集结果与测试集结果
val_features = np.zeros((X_val.shape[0], len(clfs))) # 初始化验证集结果
test_features = np.zeros((X_test.shape[0], len(clfs))) # 初始化测试机结果

for i, clf in enumerate(clfs):
    clf.fit(X_train, y_train)
    val_feature = clf.predict_proba(X_val)[:, 1]
    test_feature = clf.predict_proba(X_test)[:, 1]
    val_features[:, i] = val_feature
    test_features[:, i] = test_feature
```

```python
## 将第一层的验证集的结果输入第二层训练第二层分类器
lr.fit(val_features, y_val)
## 输出预测的结果
from sklearn.model_selection import cross_val_score

cross_val_score(lr, test_features, y_test, cv=5)
```

```markdown
output: array([1., 1., 1., 1., 1.])
```

### 3.2 Blending 用在iris数据集上

```python
from sklearn import datasets

iris = datasets.load_iris()
data, target = iris.data, iris.target

## 划分训练集和测试集
X_train_origin, X_test, y_train_origin, y_test = train_test_split(data, target, 
                                                                  test_size=0.2, random_state=2021)

## 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_origin, 
                                                   y_train_origin, 
                                                   test_size=0.3, 
                                                   random_state=2021)

print("The shape of training X:", X_train.shape)
print("The shape of training y:", y_train.shape)
print("The shape of test X:", X_test.shape)
print("The shape of test y:", y_test.shape)
print("The shape of validation X:", X_val.shape)
print("The shape of validation y:", y_val.shape)
```

```markdown
output:
The shape of training X: (84, 4)
The shape of training y: (84,)
The shape of test X: (30, 4)
The shape of test y: (30,)
The shape of validation X: (36, 4)
The shape of validation y: (36,)
```

```python
## 设置第一层分类器
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clfs = [SVC(probability=True), 
        RandomForestClassifier(n_estimators=3, n_jobs=-1, criterion='gini'),
        KNeighborsClassifier()]

## 设置第二层分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

## 输出第一层的验证集结果与测试集结果
val_features = np.zeros((X_val.shape[0], len(clfs))) # 初始化验证集结果
test_features = np.zeros((X_test.shape[0], len(clfs))) # 初始化测试集结果

for i, clf in enumerate(clfs):
    clf.fit(X_train, y_train)
    val_feature = clf.predict_proba(X_val)[:, 1]
    test_feature = clf.predict_proba(X_test)[:, 1]
    val_features[:, i] = val_feature
    test_features[:, i] = test_feature

## 将第一层的验证集的结果输入第二层训练第二层分类器
lr.fit(val_features, y_val)

## 输出预测的结果
from sklearn.model_selection import cross_val_score

y_predict = lr.predict(test_features)

print('5次交叉验证的结果：',cross_val_score(lr, test_features, y_test, cv=5))
score = accuracy_score(y_test, y_predict)
print('Blending Accuracy: %.3f' % (score * 100))
```

```markdown
output: 5次交叉验证的结果： [0.83333333 0.83333333 0.83333333 0.83333333 0.66666667]
Blending Accuracy: 95.000
```

### 3.3 自建Blending类 实现iris分类+决策边界-未完待续

# 参考资源

[^1]: [(14条消息) 图解Blending&Stacking_学如不及,犹恐失之-CSDN博客_blending](https://blog.csdn.net/sinat_35821976/article/details/83622594)

