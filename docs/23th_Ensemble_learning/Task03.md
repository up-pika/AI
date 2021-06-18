
# Task03 评估与优化回归模型

## 1 知识梳理 评估与优化模型方法

### 1.1  评估方法 MSE（均方误差）

* MSE 评价指标：$MSE = \frac{1}{N}\sum^N_{i=1}(y_i-\hat{f}(x_i))^2$
* 欠拟合：训练集和测试集误差都很大，还未收敛
* 过拟合：训练集误差很小，但是测试集误差很大，模型泛化能力很差

### 1.2 偏差-方差的权衡

* **测试均方误差**的期望值可以分解为$\hat{f}(x_0)$ 的方差，$\hat{f}(x_0)$ 的偏差平方和误差项$\epsilon$的方差：

$$
E(y_0-\hat{f}(x_0))^2 = Var(\hat{f}(x_0)) +[Bias(\hat{f}(x_0))]^2 + Var(\epsilon)
$$

* 目标：**同时最小化偏差的平方和方差**，最佳情况，偏差和方差均衡
* 偏差：由简单模型去估计数据复杂的关系引起的误差
* 偏差平方和方差本身是非负的，因此测试均方误差的期望不可能会低于误差的方差，称为建模任务的难度，$Var(\epsilon)$这个量在我们的任务确定后是无法改变的，也叫做**不可约误差**
* 偏差度量的是单个模型的**学习**能力，而方差度量的是同一个模型在**不同**数据集上的**稳定**性
* 模型复杂度越高，函数 f 的方差越大
* “偏差-方差分解”说明：**泛化**性能是由**学习**算法的能力、**数据**的充分性以及学习任务**本身的难度**所共同决定的

###  1.3 特征提取

* 估计测试误差---选择测试误差达到最小的模型
  * 训练误差修正（间接估计）：通过对过拟合的模型添加惩罚项，修正训练误差
  * 交叉验证（直接估计）:  使用k 折交叉验证的均值作为测试误差的估计
* 特征提取
  * 最优子集选择、向前逐步选择

### 1.4 压缩估计（正则化）

* 思想：将回归系数往零的方向压缩
* 岭回归 （L2正则）：在线性回归的损失函数的基础上添加对系数的约束或者惩罚$\lambda$。岭回归通过牺牲线性回归的无偏性降低方差，有可能使得模型整体的测试误差较小，提高模型的泛化能力
* Lasso 回归（L1 正则）：使用系数向量的L1范数替换岭回归的L2范数。因为L1范数可以得到稀疏解，从而可以进行特征选择

### 1. 5 降维

对方差的控制已有两种方式：

1. 使用原始变量的子集 （交叉验证）
2. 将变量系数压缩至零 （正则化）
3. （新增）将原始的特征空间投影到一个低维的空间实现变量的数量变少

* 主成分分析 (PCA)
  * 思想：通过**最大投影方差**将原始空间进行**重构**，即由特征相关重构为**无关**，即落在某个方向上的点(投影)的方差最大

## 2 实战练习

使用boston 房价数据集进行回归模型优化

```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
import seaborn as sns
```

```python
from sklearn import datasets
boston = datasets.load_boston() # 返回一个类似字典的类
X = boston.data
y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X, columns=features) # 构建具有特征名称的向量
boston_data["Price"] = y
boston_data.head()
```

|      |    CRIM |   ZN | INDUS | CHAS |   NOX |    RM |  AGE |    DIS |  RAD |   TAX | PTRATIO |      B | LSTAT | Price |
| ---: | ------: | ---: | ----: | ---: | ----: | ----: | ---: | -----: | ---: | ----: | ------: | -----: | ----: | ----: |
|    0 | 0.00632 | 18.0 |  2.31 |  0.0 | 0.538 | 6.575 | 65.2 | 4.0900 |  1.0 | 296.0 |    15.3 | 396.90 |  4.98 |  24.0 |
|    1 | 0.02731 |  0.0 |  7.07 |  0.0 | 0.469 | 6.421 | 78.9 | 4.9671 |  2.0 | 242.0 |    17.8 | 396.90 |  9.14 |  21.6 |
|    2 | 0.02729 |  0.0 |  7.07 |  0.0 | 0.469 | 7.185 | 61.1 | 4.9671 |  2.0 | 242.0 |    17.8 | 392.83 |  4.03 |  34.7 |
|    3 | 0.03237 |  0.0 |  2.18 |  0.0 | 0.458 | 6.998 | 45.8 | 6.0622 |  3.0 | 222.0 |    18.7 | 394.63 |  2.94 |  33.4 |
|    4 | 0.06905 |  0.0 |  2.18 |  0.0 | 0.458 | 7.147 | 54.2 | 6.0622 |  3.0 | 222.0 |    18.7 | 396.90 |  5.33 |  36.2 |

### 2.1 向前逐步回归

```python
#定义向前逐步回归函数
def forward_select(data,target):
    variate=set(data.columns)  #将字段名转换成字典类型
    variate.remove(target)  #去掉因变量的字段名
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')  #目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    #循环筛选变量
    while variate:
        aic_with_variate=[]
        for candidate in variate:  #逐个遍历自变量
            formula="{}~{}".format(target,"+".join(selected+[candidate]))  #将自变量名连接起来
            aic=ols(formula=formula,data=data).fit().aic  #利用ols训练模型得出aic值
            aic_with_variate.append((aic,candidate))  #将第每一次的aic值放进空列表
        aic_with_variate.sort(reverse=True)  #降序排序aic值
        best_new_score,best_candidate=aic_with_variate.pop()  #最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
        if current_score>best_new_score:  #如果目前的aic值大于最好的aic值
            variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
            selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
            current_score=best_new_score  #最新的分数等于最好的分数
            print("aic is {},continuing!".format(current_score))  #输出最小的aic值
        else:
            print("for selection over!")
            break
    formula="{}~{}".format(target,"+".join(selected))  #最终的模型式子
    print("final formula is {}".format(formula))
    model=ols(formula=formula,data=data).fit()
    return(model)
```

### 2.2 最小二乘法

```python
import statsmodels.api as sm # 最小二乘
from statsmodels.formula.api import ols # 加载ols 模型
forward_select(data=boston_data, target="Price")
```

```markdown
aic is 3286.974956900157,continuing!
aic is 3171.5423142992013,continuing!
aic is 3114.0972674193326,continuing!
aic is 3097.359044862759,continuing!
aic is 3069.438633167217,continuing!
aic is 3057.9390497191152,continuing!
aic is 3048.438382711162,continuing!
aic is 3042.274993098419,continuing!
aic is 3040.154562175143,continuing!
aic is 3032.0687017003256,continuing!
aic is 3021.726387825062,continuing!
for selection over!
final formula is Price~LSTAT+RM+PTRATIO+DIS+NOX+CHAS+B+ZN+CRIM+RAD+TAX
```

```markdown
<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x214769b6df0>
```

```python
lm=ols("Price~LSTAT+RM+PTRATIO+DIS+NOX+CHAS+B+ZN+CRIM+RAD+TAX",data=boston_data).fit()
lm.summary()
```

|    Dep. Variable: |            Price | R-squared:          | 0.741     |
| ----------------: | ---------------: | ------------------- | --------- |
|            Model: |              OLS | Adj. R-squared:     | 0.735     |
|           Method: |    Least Squares | F-statistic:        | 128.2     |
|             Date: | Sat, 10 Apr 2021 | Prob (F-statistic): | 5.54e-137 |
|             Time: |         23:50:04 | Log-Likelihood:     | -1498.9   |
| No. Observations: |              506 | AIC:                | 3022.     |
|     Df Residuals: |              494 | BIC:                | 3072.     |
|         Df Model: |               11 |                     |           |
|  Covariance Type: |        nonrobust |                     |           |

|           |     coef | std err |       t | P>\|t\| |  [0.025 | 0.975]  |
| --------: | -------: | ------: | ------: | ------: | ------: | ------- |
| Intercept |  36.3411 |   5.067 |   7.171 |   0.000 |  26.385 | 46.298  |
|     LSTAT |  -0.5226 |   0.047 | -11.019 |   0.000 |  -0.616 | -0.429  |
|        RM |   3.8016 |   0.406 |   9.356 |   0.000 |   3.003 | 4.600   |
|   PTRATIO |  -0.9465 |   0.129 |  -7.334 |   0.000 |  -1.200 | -0.693  |
|       DIS |  -1.4927 |   0.186 |  -8.037 |   0.000 |  -1.858 | -1.128  |
|       NOX | -17.3760 |   3.535 |  -4.915 |   0.000 | -24.322 | -10.430 |
|      CHAS |   2.7187 |   0.854 |   3.183 |   0.002 |   1.040 | 4.397   |
|         B |   0.0093 |   0.003 |   3.475 |   0.001 |   0.004 | 0.015   |
|        ZN |   0.0458 |   0.014 |   3.390 |   0.001 |   0.019 | 0.072   |
|      CRIM |  -0.1084 |   0.033 |  -3.307 |   0.001 |  -0.173 | -0.044  |
|       RAD |   0.2996 |   0.063 |   4.726 |   0.000 |   0.175 | 0.424   |
|       TAX |  -0.0118 |   0.003 |  -3.493 |   0.001 |  -0.018 | -0.005  |

|       Omnibus: | 178.430 | Durbin-Watson:    | 1.078     |
| -------------: | ------: | ----------------- | --------- |
| Prob(Omnibus): |   0.000 | Jarque-Bera (JB): | 787.785   |
|          Skew: |   1.523 | Prob(JB):         | 8.60e-172 |
|      Kurtosis: |   8.300 | Cond. No.         | 1.47e+04  |

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.47e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

### 2.3 岭回归

```python
from sklearn import linear_model
reg_rid = linear_model.Ridge(alpha=.5)
reg_rid.fit(X,y)
print("岭回归模型得分：",reg_rid.score(X,y))   
```



### 2.4 Lasso 回归

```python
from sklearn import linear_model
reg_lasso = linear_model.Lasso(alpha = 0.5)
reg_lasso.fit(X,y)
print("岭回归模型得分：",reg_lasso.score(X,y)) 
```

