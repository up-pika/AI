# Task01 机器学习基础与三大任务
## 1 知识梳理

### 1.1 机器学习

* 目标：利用**数学模型**理解数据，发现数据中的**规律**，用于数据的**分析和预测**
* 数据：一组**向量**。每个向量代表一个样本，共有N个样本，其中，每个样本有p+1个维度，前p个维度中每个维度表示一个**特征**，最后一个维度称为**因变量**（响应变量）。

### 1.2 机器学习分类

根据数据是否有因变量，机器学习任务可以分为：

* **有监督**学习：给定特征去估计因变量
  * **回归**：因变量是连续型变量，如：房价预测
  * **分类**：因变量是离散型变量，如：是否患癌症
* **无监督**学习：给定某些特征但未给因变量，需要学习数据本身的结构和关系。 如：聚类算法

## 2 实战练习

```python
# 引入相关科学计算包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
import seaborn as sns
```

### 2.1 回归

sklearn 中的内置数据集封装在datasets对象中，返回的对象有：

- data: 特征 X 的矩阵（ndarray)
- target: 因变量的向量（ndarray)
- feature_names:特征名称（ndarray)

例子：使用sklearn 中 Boston 房价数据集进行回归

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

| CRIM |      ZN | INDUS | CHAS |  NOX |    RM |   AGE |  DIS |    RAD |  TAX | PTRATIO |    B |  LSTAT | Price |      |
| ---: | ------: | ----: | ---: | ---: | ----: | ----: | ---: | -----: | ---: | ------: | ---: | -----: | ----: | ---- |
|    0 | 0.00632 |  18.0 | 2.31 |  0.0 | 0.538 | 6.575 | 65.2 | 4.0900 |  1.0 |   296.0 | 15.3 | 396.90 |  4.98 | 24.0 |
|    1 | 0.02731 |   0.0 | 7.07 |  0.0 | 0.469 | 6.421 | 78.9 | 4.9671 |  2.0 |   242.0 | 17.8 | 396.90 |  9.14 | 21.6 |
|    2 | 0.02729 |   0.0 | 7.07 |  0.0 | 0.469 | 7.185 | 61.1 | 4.9671 |  2.0 |   242.0 | 17.8 | 392.83 |  4.03 | 34.7 |
|    3 | 0.03237 |   0.0 | 2.18 |  0.0 | 0.458 | 6.998 | 45.8 | 6.0622 |  3.0 |   222.0 | 18.7 | 394.63 |  2.94 | 33.4 |
|    4 | 0.06905 |   0.0 | 2.18 |  0.0 | 0.458 | 7.147 | 54.2 | 6.0622 |  3.0 |   222.0 | 18.7 | 396.90 |  5.33 | 36.2 |

```python
sns.scatterplot(boston_data['DIS'], boston_data['Price'], color='r', alpha=0.6)
plt.title('Price~DIS')
plt.show()
```

![image-20210410172928242](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210410172928242.png)

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

### 2.2 分类

以 iris 鸢尾花数据集为例进行分类练习：

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
features = iris.feature_names
iris_data = pd.DataFrame(X, columns=features)
iris_data['target'] = y
iris_data.head() # 打印表头
```

|      | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target |
| ---: | ----------------: | ---------------: | ----------------: | ---------------: | -----: |
|    0 |               5.1 |              3.5 |               1.4 |              0.2 |      0 |
|    1 |               4.9 |              3.0 |               1.4 |              0.2 |      0 |
|    2 |               4.7 |              3.2 |               1.3 |              0.2 |      0 |
|    3 |               4.6 |              3.1 |               1.5 |              0.2 |      0 |
|    4 |               5.0 |              3.6 |               1.4 |              0.2 |      0 |

```python
# 可视化特征
marker = ['s', 'x', 'o']
for index,c in enumerate(np.unique(y)): # unique() 去除数组中重复数字，排序后输出
# loc 取索引某行的值
plt.scatter(x=iris_data.loc[y==c, 'sepal length (cm)'], 
    # y==c 表示行数，‘sepal length’表示列数
            y=iris_data.loc[y==c, 'sepal width (cm)'],
            alpha=0.7,label=c, marker=marker[c]) # alpha：坐标轴的透明度
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend()
plt.show()   
```

![image-20210410175648384](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210410175648384.png)

分析：每种不同的颜色和符号样式表示不同的鸢尾花，数据集有三种不同类型的鸢尾花。因变量是一个类别变量，通过特征预测鸢尾花类别的问题是一个分类问题

各个特征的相关解释：

* sepal length (cm)：花萼长度(厘米)
* sepal width (cm)：花萼宽度(厘米)
* petal length (cm)：花瓣长度(厘米)
* petal width (cm)：花瓣宽度(厘米)

### 2.3 无监督学习

<font color=red>Tips: sklearn 也可产生符合**自身需求**的数据集</font>>，见下图，地址: https://scikit-learn.org/stable/modules/classes.html?highlight=datasets#module-sklearn.datasets

![image-20210410181020504](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210410181020504.png)

例子：

```python
# 生成月牙型非凸集
from sklearn import datasets
x, y = datasets.make_moons(n_samples=4000, shuffle=True, noise=0.14, random_state=None)
for index, c in enumerate(np.unique(y)):
    plt.scatter(x[y==c,0], x[y==c,1], s=7)
plt.show()
```

![image-20210410181719708](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210410181719708.png)

```python
# 生成符合正态分布的聚类数据
from sklearn import datasets
x, y = datasets.make_blobs(n_samples=5000, n_features=2, centers=6)
for index, c in enumerate(np.unique(y)):
    plt.scatter(x[y==c, 0], x[y==c, 1], s=7)
plt.show()
```

![image-20210410181951218](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210410181951218.png)