# Task09 Boosting与Adaboost算法

# 1 Boosting(提升法)

* **思想**：利用基学习器不断调整样本的权重从而调整**误差**，迭代学习出M个弱学习器，再通过组合策略进行融合，是一种迭代类算法

* **工作机制**
  1. 从初始训练集训练一个基学习器
  2. 根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的样本在后续受到更多关注
  3. 基于调整后的样本分布来训练下一个基学习器，训练过程为阶梯状
  4. 重复**迭代**进行，直至基学习器数目达到指定的值，最终将这些基学习器通过结合策略组合起来得到最终的强学习器
* **boosting三要素**
  * 函数模型：叠加型，即$F(x)=\sum^k_{i=1}f\_i(x;w\_i)$
  * 目标函数（损失函数）
  * 优化算法：前向分布算法
* **需要解决的问题**
  * 如何调整数据集，分布到各弱学习器进行训练[^1]
  * 如何将训练得到的各个弱学习器联合起来形成强学习器
* **分类**（三要素选取不同）
  * 自适应提升 Adaboost（Adaptive Boosting）
  * 梯度提升 Gradient Boosting  Decision Tree (GBDT)

### 1.1 Adaboost

* **所选取的三要素**
  
  * 基学习器：叠加**CART**决策树（分为CART回归树或者CART分类树）或者神经网络（理论上可以为任何学习器）
  * 损失函数：指数损失函数（分类问题）或者均方误差函数(MSE)(回归问题)
    * **注意**：Adaboost是一个集成算法，框架本身并没有损失函数的概念，它只定义了如何更新基分类器的权重和样本权重，这种定义方法使得它等价于使用指数损失为损失函数，基分类器为加法模型的二分类方法）
  * 优化算法：前向分布算法
  
* **工作机制**[^2]
  
  * 先对N个训练数据的学习得到第一个弱学习器`c1`；
  * 然后将`c1`分错的数据和其他的新数据一起构成一个新的有N个训练数据的样本，通过对这个样本的学习得到第二个弱学习器`c2`；
  * 接着将`c1`和`c2`都分错了的数据加上其他的新数据构成另一个新的有N个训练数据的样本，通过对这个样本的学习得到第三个弱学习器`c3`；
  * 最终经过提升的强学习器`c_final=Majority Vote(c1,c2,c3)`。即某个数据被分为哪一类要通过`c1,c2,c3`的**加权投票法**表决
  
* **Adaboost 算法步骤**（《统计学习方法》李航）

  **假设**：

  1. 给定一个二分类的训练数据集，其中每个样本点由特征与类别组成:
     $$
     T=\left\{\left(x\_{1}, y\_{1}\right),\left(x\_{2}, y\_{2}\right), \cdots,\left(x\_{N}, y\_{N}\right)\right\},\ \ \ 
     x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}
     $$

  2. 类别$y_{i} \in \mathcal{Y}=\{-1,+1\}$，$\mathcal{X}$是特征空间，$ \mathcal{Y}$是类别集合，输出最终分类器$G(x)$

  **步骤**：

  (1) 初始化训练数据的分布（均匀分布），保证每个训练样本在基学习器的学习作用相同：
  $$
  D_{1}=\left(w_{11}, \cdots, w_{1 i}, \cdots, w_{1 N}\right), \quad w_{1 i}=\frac{1}{N}, \quad i=1,2, \cdots, N
  $$
  (2) 对于$m=1,2,...,M$            

     - 使用具有权值分布$D\_m$的训练数据集进行学习，得到基本分类器：
       $$
       G\_{m}(x): \mathcal{X} \rightarrow\{-1,+1\}
       $$

     - 计算$G\_m(x)$在训练集上的分类误差率
       $$
       e_{m}=\sum_{i=1}^{N} P\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{i=1}^{N} w_{m i} I\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)
       $$

     - 计算$G\_m(x)$的系数$\alpha\_{m}=\frac{1}{2} \log \frac{1-e\_{m}}{e\_{m}}$，这里的`log`是自然对数`ln  `                     

     - 更新训练数据集的权重分布                
  $$
  \begin{array}{c}
     D_{m+1}=\left(w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N}\right) \\
     w_{m+1, i}=\frac{w_{m i}}{Z_{m}} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right), \quad i=1,2, \cdots, N
     \end{array}
  $$
     这里的$Z\_m$是规范化因子，使得$D\_{m+1}$称为概率分布
  $$
  Z_{m}=\sum_{i=1}^{N} w_{m i} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)
  $$
  (3) 构建基本分类器的线性组合$f(x)=\sum\_{m=1}^{M} \alpha\_{m} G\_{m}(x)$，得到最终的分类器                       

  $$
  \begin{aligned}
  G(x) &=\operatorname{sign}(f(x)) \\
  &=\operatorname{sign}\left(\sum_{m=1}^{M} \alpha_{m} G_{m}(x)\right)
  \end{aligned}
  $$

* **针对boosting两个问题的解决办法，Adaboost算法采用的策略**

  1. **数据加权**：使用**加权后**选取的训练数据**代替**随机选取的训练数据，这样将训练的焦点集中在比较难分的训练数据上
  2. **弱学习器加权**：使用**加权的投票机制**代替平均投票机制。
     * 让分类或回归**效果好**（**误差小**）的弱学习器具有**较大的权重**, 使得它在表决中起较大的作用
     * 让分类或回归**效果差**的分类器具有**较小**的权重，使得它在表决中起较小的作用

* **优点**
  
  * **级联**弱学习器
  * **分类和回归精度高**
  * 弱学习器可以为**不同的**分类或回归算法
  * 充分考虑每个学习器的权重
  
* **缺点**
  
  * 弱学习器数目不好设定，需交叉验证
  * 对异常值敏感，数据不平衡时会导致模型精度下降；
  * 训练耗时
  * Adaboost只能做二分类问题，要做多分类问需要做其他的变通（Adaboost只能做二分类问题是因为最终的输出是用sign函数决定的）

#  2 实战练习

## 2.1 使用葡萄酒数据对比单个决策树和Adaboost的分类性能

```python
# 引入所需要的的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
import seaborn as sns
```

```python
# load dataset
wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', 
                'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
# 查看数据
print("Class labels",np.unique(wine["Class label"]))
wine.head()
```

```markdown
Class labels [1 2 3]
```

|      | Class label | Alcohol | Malic acid |  Ash | Alcalinity of ash | Magnesium | Total phenols | Flavanoids | Nonflavanoid phenols | Proanthocyanins | Color intensity |  Hue | OD280/OD315 of diluted wines | Proline |
| ---: | ----------: | ------: | ---------: | ---: | ----------------: | --------: | ------------: | ---------: | -------------------: | --------------: | --------------: | ---: | ---------------------------: | ------: |
|    0 |           1 |   14.23 |       1.71 | 2.43 |              15.6 |       127 |          2.80 |       3.06 |                 0.28 |            2.29 |            5.64 | 1.04 |                         3.92 |    1065 |
|    1 |           1 |   13.20 |       1.78 | 2.14 |              11.2 |       100 |          2.65 |       2.76 |                 0.26 |            1.28 |            4.38 | 1.05 |                         3.40 |    1050 |
|    2 |           1 |   13.16 |       2.36 | 2.67 |              18.6 |       101 |          2.80 |       3.24 |                 0.30 |            2.81 |            5.68 | 1.03 |                         3.17 |    1185 |
|    3 |           1 |   14.37 |       1.95 | 2.50 |              16.8 |       113 |          3.85 |       3.49 |                 0.24 |            2.18 |            7.80 | 0.86 |                         3.45 |    1480 |
|    4 |           1 |   13.24 |       2.59 | 2.87 |              21.0 |       118 |          2.80 |       2.69 |                 0.39 |            1.82 |            4.32 | 1.04 |                              |         |

数据集说明：

Class label：分类标签

- Alcohol：酒精
- Malic acid：苹果酸
- Ash：灰
- Alcalinity of ash：灰的碱度
- Magnesium：镁
- Total phenols：总酚
- Flavanoids：黄酮类化合物
- Nonflavanoid phenols：非黄烷类酚类
- Proanthocyanins：原花青素
- Color intensity：色彩强度
- Hue：色调
- OD280/OD315 of diluted wines：稀释酒OD280 OD350
- Proline：脯氨酸

```python
# 数据预处理-只考虑2,3类葡萄酒
wine = wine[wine['Class label'] != 1]
y = wine['Class label'].values
X = wine[['Alcohol','Hue']].values

# 将分类标签转成二进制编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# 按8:2分割训练集和测试集
from sklearn.model_selection import train_test_split
# stratify参数代表了按照y的类别等比例抽样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y) 
```

```python
# 使用单一决策树建模
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1)
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train,tree_test))
```

```python
# 使用sklearn 实现Adaboost （基分类树为决策树）
'''
AdaBoostClassifier相关参数：
base_estimator：基本分类器，默认为DecisionTreeClassifier(max_depth=1)
n_estimators：终止迭代的次数
learning_rate：学习率
algorithm：训练的相关算法，{'SAMME'，'SAMME.R'}，默认='SAMME.R'
random_state：随机种子
'''
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(base_estimator=tree, 
                           n_estimators=500,
                           learning_rate=0.1, 
                           random_state=1)
model = model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
model_train = accuracy_score(y_train, y_train_pred)
model_test = accuracy_score(y_test, y_test_pred)
print('Adaboost train/test accuracies %.3f/%.3f' % (model_train,model_test))
```

```markdown
单层决策树和Adaboost的性能：
Decision tree train/test accuracies 0.916/0.875
Adaboost train/test accuracies 1.000/0.917

结果分析：单层决策树似乎对训练数据欠拟合，而Adaboost模型正确地预测了训练数据的所有分类标签，而且与单层决策树相比，Adaboost的测试性能也略有提高。
```

* 可视化单层决策树和Adaboost决策树边界-对比两种方法的性能

```python
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(12,6))
for idx, clf, tt in zip([0, 1], [tree, model], ['Decision tree', 'Adaboost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],c='red', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2, s='Hue',ha='center',
         va='center',fontsize=12,transform=axarr[1].transAxes)
plt.show()
```

![image-20210419095225061](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210419095225061.png)

```markdown
注解笔记： 
* 1.np.c_与np.r_
np.r_()是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
np.c_()是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
从图中可以看出，单层决策树的决策边界为直线，而Adaboost的决策边界为曲线，Adaboost的分类性能直观上有所提升
```

# 3 拓展知识

##  3.1 集成学习个体分类器多样性

* 分析方法：误差-分歧分解 
  $$
  E = \overline{E}-\overline{A}\\\overline{E}:个体泛化误差加权均值，\overline{A}：个体学习器的加权分歧值
  $$

  * 意义：个体学习器准确性越高，多样性越大，则集成越好；证明集成学习追求好而不同

* 多样性度量：估算个体学习器的多样性程度，主要通过考虑分类器的两两相似/不相似性，有如下指标：

  * 不合度量（disagreement measure）（越大越好）
  * 相关系数（correlation coefficient）（越小越好）
  * Q-统计量（Q-statistics）
  * $K$-统计量（越小越好）

* 多样性增强的方法：

  * 数据样本扰动：如bagging 自助采样，Adboost 序列采样
    * 决策树、神经网络对数据敏感，适合采用数据样本扰动
    * **线性学习器、支持向量机、朴素贝叶斯、KNN**常被称为**稳定基学习器**（stable base learner），对这类数据样本扰动效果不明显
  * 输入属性扰动：适合属性和冗余属性较多的数据
  * 输出表示扰动：如翻转；输出调制发
  * 算法参数扰动：如调整算法的超参数

[^1]: https://blog.csdn.net/starter_____/article/details/79328749
[^2]: [机器学习算法八：Boosting及AdaBoost，GDBT，XGBoost算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/70516721)

