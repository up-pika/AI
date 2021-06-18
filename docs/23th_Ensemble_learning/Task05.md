# Task05 机器学习分类项目流程

## 1 知识梳理

### 1.1 分类模型常用度量指标

* 分类与回归的区别





| 序号 | 回归                           | 分类                               |
| ---- | ------------------------------ | ---------------------------------- |
| 1    | 因变量为连续变量               | 因变量为离散变量                   |
| 2    | 目标为获得近似拟合因变量的模型 | 目标为获得因变量所属分类类别的模型 |

* 分类情况

  * 真阳性 TP：预测值和真实值都为正例

  * 真阴性 TN：预测值和真实值都为负例

  * 假阳性 FP：预测值为正，实际值为负

  * 假阴性 FN：预测值为负，实际值为正

    



|                          | **P **预测的分类结果 | **N**     |
| ------------------------ | -------------------- | --------- |
| **P** **实际的分类结果** | 真阳性 TP            | 假阴性 FN |
| **N**                    | 假阳性 FP            | 真阴性 TN |

* 分类模型的指标

  * 准确率： **分类正确**的样本数占**总**样本的比例，即 $ACC=\frac{TP+TN}{TP+TN+FP+FN}$
  * 精度：预测为**正**且**分类正确**的样本占**预测值为正**的比例，即 $PRE=\frac{TP}{TP+FP}$
  * 召回率：预测为**正**且**分类正确**的样本占**类别为正**的比例，即 $REC = \frac{TP}{TP+FN}$
  * $F1$ 值：综合衡量**精度和召回率**，即 $F1=2\frac{PRE\times REC}{PRE+REC}$
  * $ROC$曲线：以**假阳率**为**横**轴，**真阳率**为**纵**轴画出来的曲线，曲线下方的面积越大越好
* 分类型模型常用的三个评价指标
  * 混淆矩阵 (Confusion Matrix)
  * ROC 曲线
  * AUC 面积

sklearn 中分类指标的调用函数见[^1]

### 1.2 分类模型

#### 1.2.1 逻辑回归 （Logistic Regression)

* 思想：将线性回归的结果转换到区间为[0,1] 上的概率值。即：
  $$
  \begin{aligned}
  Linear Regression: &Y =\beta_0+\beta_1X \\
  \Downarrow \\
  Logistic regression:p(X) &= \frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}
  \end{aligned}
  $$
  
* 函数图像对比如下图：

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210411101856842.png)

* 逻辑回归目标：

  * 假设数据服从0-1分布（伯努利分布）
  
  * 假设逻辑回归模型为$p(y=1|x)=\frac{1}{1+e^{-w^Tx}}$
  
* 极大似然估计 MLE 估计 $\hat{w}$, 即 
    $$
    \begin{aligned}
    \hat{w}&=argmax_wlogP(Y|X)\\
    &=argmax_x\sum^N_{i=1}(y_ilogp_1+(1-y_i)log(1-p_1)
    \end{aligned}
    $$
  
  * 梯度下降法求解更新参数：
    $$
    \begin{aligned}
    &w^{(t+1)}_k\leftarrow w^{(t)}_k-\eta\sum^N_{i=1}(y_i-\sigma(z_i))x^{(k)}_i，\\
    &其中 x^{k}_i 为第i个样本的第k个特征
    \end{aligned}
    $$
    
  
  补充：牛顿法求解梯度[^2]

#### 1.2.2 基于概率的分类模型

##### 1 线性判别分析(Linear Discriminative Analysis)

* **贝叶斯理解**

  * 贝叶斯公式：$p(Y=k|X=x)=\frac{\pi\_kf\_k(x)}{\sum^k\_{l=1}\pi\_lf\_l(x)}$ 

    另一种熟悉的写法：
    $$
    \begin{aligned}
    P(B_i|A)=\frac{P(B_i)P(A|B_i)}{\sum^N_{j=1}P(B_j)P(A|B_j)}\\
    分母可写为\ P(A)=\sum^N_{j=1}P(B_j)P(A|B_j)
    \end{aligned}
    $$
  
* **思想**：通过贝叶斯定理计算贝叶斯定理的**分子**，比较**分子最大**的那个类别为**最终类别**
  
* 求解过程如下：
    $$
    \begin{cases}
    \delta_k(x)=ln(g_k(x))=ln\pi_k+\frac{\mu}{\sigma^2}x-\frac{\mu^2}{2\sigma^2}\\\\
    \hat{\mu}_k=\frac{1}{n_k}\sum_{i:y_i=k}x_i\\\\
    \hat{\sigma}^2=\frac{1}{n-K}\sum_{k=1}^K\sum_{i:y_i=k}(x_i-\hat{\mu}_k)^2
    \end{cases}
    $$
    然后代入数据求出 $\delta_k(x)$，那个 $k$ 对一个的$\delta_k$ 大，就属于哪一类
  
* **降维分类理解**

  * 思想：将高维的数据降维到一维，然后使用某个阈值将各个类别分开

    <center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210411104925500.png" style="zoom: 50%;" /></center>

  * 目标：希望降维后的数据同一个类别自身内部方差小，不同类别之间的方差要尽可能大。简称为：**类内方差小，类间方差大**

##### 2 朴素贝叶斯

* 假设每种分类类别下的特征遵循同一个**协方差矩阵**，每两个特征之间是存在**协方差**的，因此在线性判别分析中各种特征是**不是独立**的。
* 思想：简化线性判别分析，将线性判别分析中的协方差矩阵中的**协方差**全部变成**0**，只保留各自特征的方差，也就是朴素贝叶斯假设各个特征之间是**不相关**的
* 优缺点：
  * 优点：方差减少，偏差增加（偏差-方差理论）；

#### 1.2.3 决策树

* 决策树在回归与分类问题的应用区别：

  * 回归指标：分割点为均方误差 （区域内数据的平均值）
  * 分类指标：分类错误率（不够敏感）；基尼系数；交叉熵

*  分类错误率：此区域内训练集中**非 常见类**所占的类别，即：
  $$
  \begin{aligned}
  E&=1-max_k(\hat{p}_{mk})\\
其中，&\hat{p}_{mk} 代表第m 个区域的训练集中第 k 类所占的比例
  \end{aligned}
  $$
  
* 基尼系数：衡量 $K$ 个类别的总方差，是一种衡量**结点纯度**的指标。**基尼系数越小，代表某个结点包含的观测者几乎来自同一个类别**。由基尼系数作为指标得到的分类树叫**CART**
  $$
  G=\sum^K_{k=1}\hat{p}_{mk}(1-\hat{p}_{mk})
  $$

* 交叉熵：**交叉熵越小，结点纯度越高**
  $$
  D=-\sum^K_{k=1}\hat{p}_{mk}log\hat{p}_{mk}
  $$

注意：基尼系数和交叉熵在数值上非常接近

#### 1.2.4 支持向量机 （SVM)

* 思想：找到**最大间隔超平面**，这个分割面距离最近的观测点最**远**

<center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210411112129755.png" alt="" style="zoom: 50%;" /></center>

* 公式推导：

  1. 根据距离超平面最近的点（支持向量），同时缩放w 和 b得到 SVM 的 具体形式：
     $$
     min_{w, b}\frac{1}{2}||w||^2\\
     s.t.\ y^{(i)}(w^Tx^{(i)}+b)\geq1, i=1,...,n
     $$

  2. 优化问题拉格朗日化（添加拉格朗日乘子）
     $$
     L(w,b,\alpha)=\frac12||w||^2-\sum^n_{i=1}\alpha_i\big[y^{(i)}(w^Tx^{(i)}+b)-1\big]
     $$

  3. 求解 w, b, 代入上式 （欲求b，对b求梯度然后令梯度为0，w同理）
     $$
     L(W,b,\alpha)=\sum^n_{i=1}\alpha_i-\frac1 2\sum^n_{i,j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j(x^{(i)})^Tx^{(j)}
     $$

  4. 构造对偶问题
     $$
     \begin{aligned}
     max_\alpha W(\alpha)&=\sum_{i=1}^n\alpha_i-\frac1 2\sum^n_{i,j=1}y^{(i)}y^{(j)}\alpha_i\alpha_j\big<x^{(i)},x^{(j)}\big> \\
     s.t.& \ \alpha_i\geq0, i=1,...,n\\
     &\sum^n_{i=1}\alpha_iy^{(i)}=0
\end{aligned}
     $$
     
  5. 求得 w, b 的值
     $$
     \begin{aligned}
     w^\ast &= \sum^n_{i=1}\alpha_i y^{(i)}x^{(i)}\\
   b^\ast &= -\frac{max_{i:y^{(i)}=-1}w^{\ast T}x^{(i)}+min_{i:y^{(i)}=1}w^{\ast T}x^{(i)}}{2}
     \end{aligned}
     $$
     
  6. 分离超平面和分类决策函数
  
  $$
  w^{\ast T}x+b^\ast=\sum^n_{i=1}\alpha_i y^{(i)}\big<x^{(i)},x\big> + b
  $$

#### 1.2.5 非线性支持向量机

* 思想：针对非线性问题，将数据投影到更高的维度。引入核函数避免维度爆炸

* 核函数：

  * 多项式核函数 （Polynomial Kernel) ：(常用)
    $$
    \begin{aligned}
    &K(x_i, x_j)=(<x_i, x_j>+c)^d\\
    &式中\  c\  控制低阶项的强度
\end{aligned}
    $$
  
  * 高斯核函数 (Gaussian Kernel)
  
    也称径向基核函数 （Radial Basis Function, RBF)，最主流，libsvm 默认的核函数，表达式：$K(x_i, x_j)=exp(-\frac{||x_i-x_j||_2^2}{2\sigma^2})$
    
    **注意**：使用高斯核函数之前需要将特征**标准化**，可以用于衡量样本之间的**相似度**
    
  * Sigmoid 核函数: $  K(x_i, x_j)=tanh(\alpha x_i^Tx_j+c)$
  
    此时的SVM 相当于没有隐藏层的简单神经网络
  
  * 余弦相似度核:$ K(x_i,x_j)=\frac{x_i^Tx_j}{||x_i||\ ||x_j||}$
  
    常用于衡量两端文字的余弦相似度

## 2  实战练习

### 2.1 确定数据集与特征选择

```python
## 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
import seaborn as sns
```

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature = iris.feature_names
data = pd.DataFrame(X, columns=feature)
data['target'] = y
data.head()
```

|      | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target |
| ---: | ----------------: | ---------------: | ----------------: | ---------------: | -----: |
|    0 |               5.1 |              3.5 |               1.4 |              0.2 |      0 |
|    1 |               4.9 |              3.0 |               1.4 |              0.2 |      0 |
|    2 |               4.7 |              3.2 |               1.3 |              0.2 |      0 |
|    3 |               4.6 |              3.1 |               1.5 |              0.2 |      0 |
|    4 |               5.0 |              3.6 |               1.4 |              0.2 |      0 |

* sepal length (cm)：花萼长度(厘米)
* sepal width (cm)：花萼宽度(厘米)
* petal length (cm)：花瓣长度(厘米)
* petal width (cm)：花瓣宽度(厘米)

### 2.2 逻辑回归分类

调用函数：`from sklearn.linear_model import LogisticRegression`

```python
'''
penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’正则化方式
dual bool, default=False 是否使用对偶形式，当n_samples> n_features时，默认dual = Fal
C float, default=1.0
solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs
l1_ratio float, default=None
'''
from sklearn.linear_model import LogisticRegression
log_iris = LogisticRegression(solver='sag') 
log_iris.fit(X, y)
log_iris.score(X, y)
```

```markdown
0.9866666666666667
```

### 2.3 线性判别分析-基于概率的分类方法

调用函数：`from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`

```python
'''
参数：
solver:{'svd'，'lsqr'，'eigen'}，默认='svd'
solver的使用，可能的值：
'svd'：奇异值分解（默认）。不计算协方差矩阵，因此建议将此求解器用于具有大量特征的数据。
'lsqr'：最小二乘解，可以与收缩结合使用。
'eigen'：特征值分解，可以与收缩结合使用。
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_iris = LinearDiscriminantAnalysis()
lda_iris.fit(X, y)
lda_iris.score(X, y)
```

```markdown
0.98
```

### 2.4 朴素贝叶斯-基于概率的分类方法

调用函数：`from sklearn.naive_bayes import GaussianNB`

```
from sklearn.naive_bayes import GaussianNB
NB_iris = GaussianNB()
NB_iris.fit(X, y)
NB_iris.score(X, y)
```

```markdown
0.96
```

### 2.5 决策树

调用函数：`from sklearn.tree import DecisionTreeClassifier`

```python
'''
criterion:{“gini”, “entropy”}, default=”gini”
max_depth:树的最大深度。
min_samples_split:拆分内部节点所需的最少样本数
min_samples_leaf :在叶节点处需要的最小样本数。
'''
from sklearn.tree import DecisionTreeClassifier
tree_iris = DecisionTreeClassifier(min_samples_leaf=5)
tree_iris.fit(X, y)
tree_iris.score(X, y)
```

```markdown
0.9733333333333334
```

### 2.6 SVM

调用函数：`from sklearn.svm import SVC`

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
'''
C:正则化参数。正则化的强度与C成反比。必须严格为正。惩罚是平方的l2惩罚。
kernel:{'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'}，默认='rbf'
degree:多项式和的阶数
gamma:“ rbf”，“ poly”和“ Sigmoid”的内核系数。
shrinking:是否软间隔分类，默认true
'''
svc_iris = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_iris.fit(X, y)
svc_iris.score(X, y)
```

```markdown
0.9733333333333334
```

## 参考资料

[^1]:https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
[^2]: https://zhuanlan.zhihu.com/p/165914126