# Task07 投票的原理与实战

# 1 集成学习的三种结合策略

* 目的：提高泛化性能；降低进入局部最小点的风险；扩大假设空间；通过结合策略融合决策后，集成学习的预测结果应当优于任何一个基模型
* 分类：平均法、投票法、学习法（Stacking)

## 1.1 平均法

* 分类

  * 简单平均法（Simple Averaging）即 $y = \frac{1}{T}\sum^T\_{i=1}y\_i(x)$

  * 加权平均法（Weighted Averaging）即 $y = \sum^T\_{i=1}w\_iy\_i(x)$

    加权平均法的权重一般从训练数据中学习而得

* 简单平均法与加权平均法比较

  * 在个体学习器性能**相差较大**时宜使用**加权平均法**，而在个体学习器**相近**时采用**简单平均法**

## 1.2 投票法

### 1.2.1 定义与基本思想

* 定义：集成学习中针对**分类**问题的一种**结合**策略
* 用途：**分类**模型 和 **bagging**模型
* 思想：选择所有机器学习算法当中得票最多的那个类
* 使用条件：1. 基模型之间的效果不能相差过大；2. 预测效果近似的基模型之间应该有较小的同质性，比如树模型与线性模型的投票优于两个树模型或两个线性模型

### 1.2.2 投票法的应用与分类

#### 1 应用场景分类

* 分类投票法：取所有模型中出现最多的预测结果；
  * sklearn 调用函数：`from sklearn.ensemble import VotingClassifier`[^1]
* 回归投票法：取所有模型预测结果的平均值
  * sklearn 调用函数：`from sklearn.ensemble import VotingRegressor`[^2]

函数使用例子：

注意：使用模型需要提供一个**模型列表**，列表中每个模型采用**Tuple**的结构表示，第一个元素代表**名称**，第二个元素代表**模型**，需要保证每个模型必须拥有**唯一**的名称

```python
# 使用Logistic和svc作为基模型进行投票法选取最优模型;voting='soft'or'hard'设置软投票或者硬投票方式
models = [('lr',LogisticRegression()),('svm',SVC())]
ensemble = VotingClassifier(estimators=models, voting='soft')

# 对于需要预处理的模型，比如 svc,需要使用pipeline 完成数据预处理
models = [('lr',LogisticRegression()),('svm',make_pipeline(StandardScaler(),SVC()))]
ensemble = VotingClassifier(estimators=models)
```

#### 2 投票规则分类

* 硬投票（Hard Voting Classifier)
  * 基本思想：选择多个模型中输出最多次数的那个类；少数服从多数
  * 类标记决策：通过**直接**选择模型输出最多的**标签**，若标签数量相等，则按照升序的次序进行选择
* 软投票 （Soft Voting Classifier）
  * 定义：又称加权平均该类投票，每个分类票数**乘以权重**，最终根据各类别投票的加权和选出最大值对应的类 ；
  * 类概率决策：通过**输出类概率**实现

### 1.2.3 投票法的其他分类方式

* 多数投票法（Voting）[^3]
  * 相对多数投票法（Plurality Voting）：**少数服从多数**
  * 绝对多数投票法（Majority Voting）：**票数至少要过半**。多个分类器对某一类别的预测结果若大于总投票结果的一半，则预测为该类别；否则拒绝该预测结果
* 加权投票法（Weighted Voting）：类似**加权平均**，每个分类票数乘以权重，最终根据各类别投票的加权和选出最大值对应的类  

### 1.2.4 硬投票与软投票理解实例

| 硬投票                                                       | 软投票                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 对于某个样本：<br />模型 1 的预测结果是 类别 A<br />模型 2 的预测结果是 类别 B<br />模型 3 的预测结果是 类别 B | 对于某个样本：<br />模型 1 的预测结果是 类别 A 的概率为 99%<br />模型 2 的预测结果是 类别 A 的概率为 49%<br />模型 3 的预测结果是 类别 A 的概率为 49% |
| 有2/3的模型预测结果是B，因此硬投票法的预测结果是B            | 最终对于类别A的预测概率的平均是<br /> (99 + 49 + 49) / 3 = 65.67%，因此软投票法的预测结果是A |

## 1.3 学习法 (Stacking)

### 1.3.1 定义与基本思想

* 初级学习器：个体学习器；次级或元学习器：结合的学习器
* 思想：先从初始数据集训练出初级学习器，然后“生成”一个新数据集用于训练次级学习器

# 2 箱线图（Box-plot）

* 定义：又称为盒须图、盒式图或箱线图，用于表征一组**数据分散**情况

* 作用：

  1. 直观地识别数据中的异常值（离群点）
  2. 直观地判断数据离散情况，了解数据分布情况
  3. 判断数据的偏态和尾重；异常值越多尾部越重；偏态表示**偏离**程度，异常值集中在较小值一侧，则分布呈**左偏态**；异常值集中在较大值一侧，则分布呈**右偏态**

* 理解分析箱线图：由5个数值点构成

  <center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210412172158414.png" alt="" style="zoom:50%;" /></center>

  <center>图片来源网络！</center>

  *  上限：非异常值范围内的最大值 
  * 下限：非异常值范围内的最小值
  * 上四分位数（75% 分位数，Q3）
  * 中位数：奇数个序列取中位数；偶数个序列，取中间两个数的平均数
  * 下四分位数（25% 分位数，Q1）

* 箱线图的**绘制方法**：先找出一组数据的上限、下限、中位数和两个四分位数；然后， 连接两个四分位数画出箱体；再将上边缘和下边缘与箱体相连接，中位数在箱体中间

# 3 投票法实战

## 随机数据集-多个KNN模型投票法

* 导入所需要的包

```python
# import the needed packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
```

* 构建随机数据集

```python
# test classification dataset - 1000个样本，20个特征 随机数据集
from sklearn.datasets import make_classification
def get_dataset():
    # define dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, n_redundant=5, random_state=2)

    # summarize the dataset
    print(X.shape, y.shape)
    return X, y
```

* 基于多个KNN 模型进行投票决策

```python
# 每个 KNN 模型采用不同的邻居值 K 参数
def get_voting():
    # define the base models
    models = list()
    # append(()) 可以实现tuple的格式
    models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
    models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
    models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
    models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
    models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    return ensemble
```

* 创建模型列表

```python
# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn1'] = KNeighborsClassifier(n_neighbors=1)
    models['knn3'] = KNeighborsClassifier(n_neighbors=3)
    models['knn5'] = KNeighborsClassifier(n_neighbors=5)
    models['knn7'] = KNeighborsClassifier(n_neighbors=7)
    models['knn9'] = KNeighborsClassifier(n_neighbors=9)
    models['hard_voting'] = get_voting()
    return models
```

* 评估模型-交叉验证

```python
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    #以分层10倍交叉验证三次重复的分数列表的形式返回
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores
```

* 使用箱线图可视化投票法的决策结果

```python
# 评估每个算法的平均性能，用箱型图和须状图可视化比较每个算法的精度分数分布
# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('->%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    
# plot model performance for comparison
plt.boxplot(results, labels=names, 
            showmeans=True,#显示均值
            flierprops={'marker':'o','markerfacecolor':'red','color':'black'},#设置异常值属性，点的形状、填充颜色和边框色
            meanprops={'marker':'D','markerfacecolor':'indianred'},#设置均值点的属性，点的颜色和形状
            medianprops={"linestyle":'--','color':'orange'}#设置中位数线的属性，线的类型和颜色
           )
plt.show()
```

```markdown
(1000, 20) (1000,)
->knn1 0.873 (0.030)
->knn3 0.889 (0.038)
->knn5 0.895 (0.031)
->knn7 0.899 (0.035)
->knn9 0.900 (0.033)
->hard_voting 0.902 (0.034)
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210412184716702.png)



# 补充

## 1 为什么要采样结合策略？

从三个方面解释：见下图[^4]（西瓜书）

1. 统计角度：学习任务的假设空间往往很大，可能有多个假设在训练集上达到同等性能，单使用一个可能**泛化性能**不佳，结合多个可以降低这个风险
2. 计算角度：算法可能陷入**局部最小值**，结合多个可以降低这个风险
3. 表示角度：真实的问题可能和学习器的假设不在同一个空间，结合多个可以使得假设空间变大，更有可能接近真是假设

<center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/20170907135317287" alt="img" style="zoom:50%;" /></center>

# 参考资源

[^1]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
[^2]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
[^3]: https://hyper.ai/wiki/2364
[^4]: https://blog.csdn.net/iamxiaofeifei/article/details/77871805?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-1&spm=1001.2101.3001.4242