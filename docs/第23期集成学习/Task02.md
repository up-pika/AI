# Task02 使用sklearn 构建完整的机器学习项目流程

一般来说，一个完整的机器学习项目流程分为以下步骤：

1. **明确**项目任务：分类 / 回归
2. 收集**数据集**并选择合适的**特征**
3. 选择**度量**模型性能的指标
4. 选择具体的**模型**并进行训练以优化模型
5. **评估**模型的性能并**调参**

## 1 回归问题常用的模型度量指标

1. MSE (均方误差) : $MSE(y,\hat{y}) = \frac{1}{n\_{samples} }\sum\_{ {i=0} }^{n_{samples} }(y_i-\hat{y_i})^2$
2. MAE (平均绝对误差)： $MAE(y, \hat{y})= \frac{1}{n\_{samples} }\sum\_{ {i=0} }^{n\_{samples} }|y\_i-\hat{y\_i}|$
3. $R^2$决定系数：$R^2(y,\hat{y} )=1-\frac{\sum^n\_{i=1}(y\_i-\hat{y}\_i)^2} {\sum^n\_{i=1} (y\_i-\overline{y}\_i)^2}$ 
4. 解释方差得分：$explained\_{variance}(y, \hat{y})= 1- \frac{Var(y-\hat{y})}{Var(y)}$

Tips: sklearn 中的回归问题度量指标[^1]

## 2 回归模型

**回归分析**研究的是因变量（目标）和自变量（特征）之间的关系。常用于**预测分析，时间序列模型**以及发现变量之间的因果关系。通常使用曲线/线来拟合数据点，目标是使曲线到数据点的距离差异最小。

常见的回归模型：

* 线性回归模型
  * 多项式回归
  * 广义可加模型 （GAM)
* 回归树
* 支持向量机回归（SVR)

### 2.1 线性回归

* **线性回归模型**：假设目标值与特征之间线性相关，即满足一个多元一次方程。通过构建损失函数，求解损失函数最小时的参数w;

* 数据集： $D={(x_1, y_1),...,(x_N,y_N)}$

  $X=(x_1, x_2,..., x_N)^T \\ Y = (y_1,y_2,...,y_N)^T$

* 模型表达式：$\hat{y} = f(w) = w^Tx$

* 估计求解参数w的方法
  * 最小二乘估计 ：**衡量**真实值 $y_i$ 与线性回归模型的预测值 $w^Tx$ 之间的**差距**。常用于**回归分析和参数估计**。表示为误差的二范数的平方和也就是平方误差和

    * 目标公式与求解过程：

    $$
    \begin{aligned}
    L(w) &= \sum^N_{i=1}||w^Tx_i-y_i||^2\\
    &=\sum^N_{i=1}(w^Tx_i-y_i)^2\\
    &=(w^TX^T-Y^T)(w^TX^T-Y^T)^T \\
    &=w^TX^TXw-2w^TX^TY+YY^T\\
    &\Downarrow\\
    &Goal: \hat{w} = argmin L(w) \\
    &Solution: \\
    &\Downarrow\\
    \frac{\partial L(w)}{\partial w} &= 2X^TXw-2X^TY=0 \\
    &\Downarrow\\
    w &=(X^TX)^{-1}X^TY
    \end{aligned}
    $$

  * 几何解释

    已知：两个向量a和b相互垂直$\Rightarrow $`<a,b>`$= a.b =a^Tb=0$

    平面的法向量为 $Y-Xw$, 与平面 $X$ 垂直，则有：$X^T(Y-Xw)=0\Rightarrow w= (X^TX)^{-1}X^TY$

  * 概率解释

    假设噪声 $\epsilon$~$N(0, \sigma^2)$, $y=f(w)+\epsilon = w^Tx+\epsilon \Rightarrow y|x_i, w \ N(w^Tx,\sigma^2)$

    用极大似然估计MLE 对参数 w 进行估计：
    $$
    \begin{aligned}
    L(w) &= log P(Y|X;w) \\
    &= log\prod^N_{i=1}P(y_i|x_i;w) \\
    &=\sum^N_{i=1}log P(y_i|x_i;w)\\
    &=\sum^N_{i=1}log(\frac{1}{\sqrt{2\pi\sigma}}exp(-\frac{(y_i-w^Tx_i)^2}{2\sigma^2})) \\
    &=\sum^N_{i=1}\big[log(\frac{1}{\sqrt{2\pi\sigma}})-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2\big]\\
    &\Downarrow \\
    argmax_wL(w) &= argmin_w\Big[l(w)=\sum^N_{i=1}(y_i-w^Tx_i)^2\Big]
    \end{aligned}
    $$
    于是，线性回归的最小二乘估计$\Leftrightarrow$ 噪声  $\epsilon$~$N(0, \sigma^2)$ 的极大似然估计

#### 1 线性回归推广---多项式回归

* 为体现因变量和特征的非线性关系，将标准的线性回归函数换成多项式函数：

$$
\begin{aligned}
y_i&=w_0+w_1x_i+\epsilon_i \\
\Downarrow \\
y_i &= w_0+w_1x_i+w_2x_i^2+...+w_dx_i^d+\epsilon
\end{aligned}
$$

**注意：**多项式的阶数d不能取过大，d越大，多项式曲线越光滑，在X的边界处有异常的波动，导致预测效果下降。一般 $d\le 3 $或者 $d\le4$。

#### 2 线性回归推广----广义可加模型 （GAM)

* GAM 是一个将线性模型推广至非线性模型的框架，每个变量都用**非线性函数**替代，但是模型仍然**保持整体可加性**。

* 既适用于**回归模型**的推广，也适用于**分类模型**的推广。

* 标准的线性回归模型换成GAM模型框架：
  $$
  \begin{aligned}
  y_i &= w_0+w_1x_{i1}+...+w_px_{ip}+\epsilon_i \\
  \Downarrow \\
  y_i&=w_0+\sum^p_{j=1}f_j(x_{ij}) + \epsilon_i
  \end{aligned}
  $$

* 优缺点：
  * 优点：简单易操作
  * 缺点：忽略一些有意义的交互作用。

### 2.2 回归树 （决策树）

* 思想：依据**分层和分割**的方式将**特征空间**划分为一系列**简单**的区域。对某个给定的待预测的自变量，用它所属区域中**训练集的平均数或众数**对其进行预测。

* 决策树由结点（node) 和有向边(directed edge) 组成。

  * 结点类型：内部结点（internal node）和叶结点(leaf node)。
    * 内部结点：特征或属性
    * 叶结点：类别或者待预测的目标值

* 建立回归树的**步骤**：

  1. 将自变量的特征空间的可能取值构成的集合分割成 N 个互斥的区域 $R_1, R_2,...,R_N$
  2. 对落入同一个区域 $R_i$的每个观测值作相同的预测，预测值等于 $R_i$ 上训练集的因变量的简单算术平均。

* 回归树表达式：
  $$
  f(x) = \sum^N_{i=1}\hat{c}I
  $$

* 优缺点：

  * 优点：解释性强；接近于人的决策方式；图的表示方式更直观；直接做定性的特征；对异常值不敏感，能很好地处理缺失值和异常值；
  * 缺点：准确性有待提升；不支持在线学习，新样本加入需要重建树；容易过拟合；忽略特征间的相关性；

### 2.3 支持向量机回归 （SVR)

* 思想：落在f(x) 的 $\epsilon$ 邻域空间中的样本点不需要计算损失，只有落在 $\epsilon$ 领域空间外的样本才计算损失。

## 3 实战练习

#### 1. 使用sklearn 的boston 房价预测数据集

```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
import seaborn as sns

# download boston dataset
from sklearn import datasets
boston = datasets.load_boston() # 返回一个类似字典的类
X = boston.data
y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X, columns=features) # 构建具有特征名称的向量
boston_data["Price"] = y
boston_data.head()
```

|      |    CRIM |   ZN | INDUS | CHAS |   NOX |    RM |  AGE |    DIS |  RAD |   TAX | PTRATIO |      B | LSTAT |
| ---: | ------: | ---: | ----: | ---: | ----: | ----: | ---: | -----: | ---: | ----: | ------: | -----: | ----: |
|    0 | 0.00632 | 18.0 |  2.31 |  0.0 | 0.538 | 6.575 | 65.2 | 4.0900 |  1.0 | 296.0 |    15.3 | 396.90 |  4.98 |
|    1 | 0.02731 |  0.0 |  7.07 |  0.0 | 0.469 | 6.421 | 78.9 | 4.9671 |  2.0 | 242.0 |    17.8 | 396.90 |  9.14 |
|    2 | 0.02729 |  0.0 |  7.07 |  0.0 | 0.469 | 7.185 | 61.1 | 4.9671 |  2.0 | 242.0 |    17.8 | 392.83 |  4.03 |
|    3 | 0.03237 |  0.0 |  2.18 |  0.0 | 0.458 | 6.998 | 45.8 | 6.0622 |  3.0 | 222.0 |    18.7 | 394.63 |  2.94 |
|    4 | 0.06905 |  0.0 |  2.18 |  0.0 | 0.458 | 7.147 | 54.2 | 6.0622 |  3.0 | 222.0 |    18.7 | 396.90 |  5.33 |

各个特征的相关解释：
CRIM：各城镇的人均犯罪率
ZN：规划地段超过25,000平方英尺的住宅用地比例
INDUS：城镇非零售商业用地比例
CHAS：是否在查尔斯河边(=1是)
NOX：一氧化氮浓度(/千万分之一)
RM：每个住宅的平均房间数
AGE：1940年以前建造的自住房屋的比例
DIS：到波士顿五个就业中心的加权距离
RAD：放射状公路的可达性指数
TAX：全部价值的房产税率(每1万美元)
PTRATIO：按城镇分配的学生与教师比例
B：1000(Bk - 0.63)^2其中Bk是每个城镇的黑人比例
LSTAT：较低地位人口
Price：房价

#### 2. 线性回归 

```python
from sklearn import linear_model # 线性回归模型
lin_reg = linear_model.LinearRegression() # 实例化线性回归类
lin_reg.fit(X,y) # 训练模型
print('Coef of linear regression model：', lin_reg.coef_) 
print('Score of linear regression model:', lin_reg.score(X,y))
```

```markdown
Coef of linear regression model： [-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
 -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
  3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03
 -5.24758378e-01]
Score of linear regression model: 0.7406426641094095
```

#### 3. 多项式回归

```python
from sklearn.preprocessing import PolynomialFeatures
X_arr = np.arange(6).reshape(3, 2)
print("Original X：\n", X_arr)

# degree: 特征转换的阶数 default=2, 2次多项式->特征矩阵经转换后，相当于将一元二次转换为二元一次
poly = PolynomialFeatures(degree=2)
print("2次转化X: \n", poly.fit_transform(X_arr))

# interaction_only : 是否只含交互项，default = false
poly = PolynomialFeatures(interaction_only=True)
print("2次转化X：\n", poly.fit_transform(X_arr))
```

```markdown
Original X：
 [[0 1]
 [2 3]
 [4 5]]
2次转化X: 
 [[ 1.  0.  1.  0.  0.  1.]
 [ 1.  2.  3.  4.  6.  9.]
 [ 1.  4.  5. 16. 20. 25.]]
2次转化X：
 [[ 1.  0.  1.  0.]
 [ 1.  2.  3.  6.]
 [ 1.  4.  5. 20.]]
```

#### 4. GAM

```python
from pygam import LinearGAM
gam = LinearGAM().fit(boston_data[boston.feature_names], y)
gam.summary()
```

```markdown
LinearGAM                                                                                                 
=============================================== ==========================================================
Distribution:                        NormalDist Effective DoF:                                    103.2423
Link Function:                     IdentityLink Log Likelihood:                                 -1589.7653
Number of Samples:                          506 AIC:                                             3388.0152
                                                AICc:                                            3442.7649
                                                GCV:                                               13.7683
                                                Scale:                                              8.8269
                                                Pseudo R-Squared:                                   0.9168
==========================================================================================================
Feature Function                  Lambda               Rank         EDoF         P > x        Sig. Code   
================================= ==================== ============ ============ ============ ============
s(0)                              [0.6]                20           11.1         2.20e-11     ***         
s(1)                              [0.6]                20           12.8         8.15e-02     .           
s(2)                              [0.6]                20           13.5         2.59e-03     **          
s(3)                              [0.6]                20           3.8          2.76e-01                 
s(4)                              [0.6]                20           11.4         1.11e-16     ***         
s(5)                              [0.6]                20           10.1         1.11e-16     ***         
s(6)                              [0.6]                20           10.4         8.22e-01                 
s(7)                              [0.6]                20           8.5          4.44e-16     ***         
s(8)                              [0.6]                20           3.5          5.96e-03     **          
s(9)                              [0.6]                20           3.4          1.33e-09     ***         
s(10)                             [0.6]                20           1.8          3.26e-03     **          
s(11)                             [0.6]                20           6.4          6.25e-02     .           
s(12)                             [0.6]                20           6.5          1.11e-16     ***         
intercept                                              1            0.0          2.23e-13     ***         
==========================================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

#### 4. 决策树回归

```python
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(criterion='mse', min_samples_leaf=5)
tree_model.fit(X, y)
tree_model.score(X, y)
```

```markdown
0.9376307599929274
```

#### 5. SVR 回归

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.pipeline import make_pipeline # 使用管道，把预处理和模型形成一个流程

svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
svr_model.fit(X, y)
svr_model.score(X, y)
```

```markdown
0.7024525421955277
```

## 补充：

### 1. $R^2$ 决定系数

$R^2$ 决定系数为回归平方和与总平方和之比。

**$R^2$越大（接近于1）**，所拟合的回归方程**越优**
$$
\begin{aligned}
\overline{y}&=\frac{1}{n}\sum_{i=1}^ny_i\\
SS_T&=\sum_{i=1}^n(y_i-\overline{y})^2\\
SS_R&=\sum_{i=1}^n(y_i-\hat{y})^2\\
SS_E&=\sum_{i=1}^n(\hat{y}_i-\overline{y})^2\\
R^2&=\frac{SS_E}{SS_T} = 1-\frac{SS_R}{SS_T}
\end{aligned}
$$
式中：

$\overline{y}$：均值

$SS_T$ : 总平方和 ，度量样本的离散程度，如果再除以 n-1,得到方差

$SS_R$：回归平方和  (回归模型预测值与真实值误差的平方和)

$SS_E$：残差平方和 （残差：实际值与观察值之间的差异）

$R^2$是一个能够衡量回归拟合好坏程度的度量，而拟合程度不应受到数值离散性的影响，所以通过**“相除”**的方式来克服这个影响

## 参考资料

[^1]:(https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)