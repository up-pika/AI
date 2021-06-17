# Task06 评估与优化分类模型

## 1 知识梳理

### 1.1 用管道简化工作流

1. 使用`sklearn.pipeline.make_pipeline`建立工作流
2. 使用`sklearn.pipelime.Pipeline`构建工作流对象
3. 使用模型的`fit`方法

### 1.2 使用k 折交叉验证评估模型性能

* 步骤：
  1. 将数据集随机划分成 $n$ 个**互斥**子集，每次选择 $n-1$ 份作为训练集，其余作为测试集

  2. 当完成一轮时，计算验证集的均方误差$MSE$，然后重新选择 n-1 份来训练数据

  3. 重复(1)(2)  $k$ 轮后（$k\le n$)，计算 $k$ 个$MSE_i$ 的均值：
     $$
     MSE = \frac1 k \sum^k_{i=1} MSE_i
     $$

  <center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/v2-c894c9cdf08852339acf372820ef5d6c_720w.jpg" alt="" style="zoom:50%;" /></center>

* 缺点：此方法充分利用了所有样本，但是计算比较繁琐，需要训练 k 次，测试 k 次

- k折交叉验证：调用函数 `sklearn.model_selection.cross_val_score`
- 分层 $k$ 折交叉验证：调用函数 `sklearn.model_selection.StratifiedKFold`

### 1.3 使用学习和验证曲线调试算法

| 序号 | 学习曲线                                             | 验证曲线                   |
| ---- | ---------------------------------------------------- | -------------------------- |
| 1    | 绘制不同**训练集大小**时训练集和交叉验证的**准确率** | 绘制模型参数与准确率的关系 |
| 2    | X轴是数据的数量，y轴是准确率                         | x轴为参数量，y轴为准确率   |
| 3    | 评估**样本量**和指标的关系                           | 评估**参数**和指标的关系   |

- 用**学习曲线**诊断**偏差与方差**：使用`sklearn.model_selection.learning_curve`
- 用**验证曲线**解决**欠拟合和过拟合**：使用`sklearn.model_selection.validation_curve`

### 1.4 超参数调优

- 网格搜索：调用函数 `sklearn.model_selection.GridSearchCV`
- 随机网格搜索：调用函数 `sklearn.model_selection.RandomizedSearchCV`
- 嵌套交叉验证：构建网格搜索对象，并在交叉验证中传入网格搜索对象

### 1.4 比较不同的性能评估指标

#### 1.4.1 混淆矩阵

* 混淆矩阵又称误差矩阵

| 混淆矩阵             | **P **预测的分类结果 | **N** 预测的分类结果 |
| -------------------- | -------------------- | -------------------- |
| **P 实际的分类结果** | 真阳性 TP            | 假阴性 FN            |
| **N 实际的分类结果** | 假阳性 FP            | 真阴性 TN            |

* 混淆矩阵的一级指标：

  * 真阳性 TP：将正类预测为正类，又称命中
  * 真阴性 TN：将负类预测为负类，又称正确拒绝
  * 假阳性 FP：将负类预测为正类，又称假警报，第一型错误
  * 假阴性 FN：将正类预测为负类，又称未命中，第二型错误

* 混淆矩阵的二级指标：

  最基础的二级指标为准确率，精确率，灵敏度（召回率），特异度

  * 错误率：$ERR=\frac{FP+FN}{P+N}=\frac{FP+FN}{TP+FP+FN+TN}$（针对整个样本）
  * 准确率：$ACC=\frac{TP+TN}{P+N}= \frac{TP+TN}{TP+FP+FN+TN}$ （针对整个样本）
  * <font color=blue>**真阳率**</font>：$TPR=\frac{TP}{P}=\frac{TP}{TP+FN}$ （针对实际的正类，命中率，敏感度）
  * <font color=red>**假阳率**</font>：$FPR=\frac{FP}{N}=\frac{FP}{FP+TN}$  ( 针对实际的负类，错误命中率，假报警率)
  * 真阴率 (特异度)：$SPC=1-FPR=\frac{TN}{N}=\frac{TN}{FP+TN}$
  * 精度 (精确率,准确率): $PRE=\frac{TP}{TP+FP}$ （针对预测的正类，预测为**正**且**分类正确**的样本占**预测值为正**的比例）
  * 召回率( 灵敏度，=真阳率)： $REC =TPR=\frac{TP}{P}= \frac{TP}{TP+FN}$ （针对实际的正类，预测为**正**且**分类正确**的样本占**类别为正**的比例）

* 混淆矩阵的三级指标：

  * F1-score值：综合衡量**精度和召回率**(调和平均值)，即 $F1=2\frac{PRE\times REC}{PRE+REC}$
  * F1-score 取值范围为[0~1]，1表示模型的输出最好，0表示模型的输出最差

#### 1.4.2 ROC 曲线

ROC (Receiver Operating Characteristic Curve)，又称受试者工作特征曲线

* ROC 曲线是**二元分类**模型分类效果的分析工具，用来**衡量分类器的分类能力**
* $ROC$曲线：以<font color=red>**假阳率**</font>为**横**轴（针对负样本），<font color=blue>**真阳率**</font>为**纵**轴（针对正样本）画出来的曲线，曲线下方的面积（AUC面积）越大越好。
* 优点：当测试集中正负样本分布发生变化，ROC曲线保持不变

#### 1.4.3 AUC 面积

* 定义： Roc 曲线与坐标轴形成的面积，取值范围 [0, 1]
* 实际意义为模型打分时将正例分数排在反例前面的概率

#### 1.4.4 sklearn 调用函数：

准确率：`sklearn.metrics.precision_score`
召回率：`sklearn.metrics.recall_score`
F1-Score：`sklearn.metrics.f1_score`

## 2 实战练习

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

### 2.2 网格搜索法调优`GridSearchCV()`

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

start_time = time.time()
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'svc__C':param_range,'svc__kernel':['linear']},
              {'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs = gs.fit(X, y)
end_time = time.time()
print("网格搜索经历时间：%.3f S" % float(end_time-start_time))
print('最佳得分:', gs.best_score_)
print('最优参数是:', gs.best_params_)
```

```markdown
网格搜索经历时间：1.999 S
最佳得分: 0.9800000000000001
最优参数是: {'svc__C': 1.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'}
```

### 2.3 随机搜索调优 `RandomizeSearchCV()`

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import time

start_time = time.time()
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
# param_grid = [{'svc__C':param_range,'svc__kernel':['linear','rbf'],'svc__gamma':param_range}]
gs = RandomizedSearchCV(estimator=pipe_svc, param_distributions=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs = gs.fit(X,y)
end_time = time.time()
print("随机网格搜索经历时间：%.3f S" % float(end_time-start_time))
print('最佳得分:', gs.best_score_)
print('最优参数是:', gs.best_params_)
```

```markdown
随机网格搜索经历时间：0.146 S
最佳得分: 0.9533333333333334
最优参数是: {'svc__kernel': 'rbf', 'svc__gamma': 1.0, 'svc__C': 1000.0}
```

### 2.4 绘制混淆矩阵

```python
# 查看数据集的分类数，可知有3类
data['target'].unique()
data['target'].value_counts()
```

```markdown
0    50
1    50
2    50
Name: target, dtype: int64
```

```python
# 取出类别0和1作为基础数据集
df = data.query("target==[0,1]")
y = df['target'].values
X = df.iloc[:, :-1].values
```

```python
# 绘制混淆矩阵
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)
pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))

pipe_svc.fit(X_train,y_train)
y_pred = pipe_svc.predict(X_test)
# 得到混淆矩阵
confmat = confusion_matrix(y_true=y_test,y_pred=y_pred)

fig,ax = plt.subplots(figsize=(3,3))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.7)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210411215249875.png)

### 2.5 绘制ROC 曲线

```python
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import make_scorer,f1_score
scorer = make_scorer(f1_score,pos_label=0)
# 使用网格搜索
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring=scorer,cv=10)
y_pred = gs.fit(X_train,y_train).decision_function(X_test)
# 真阳率和假阳率
fpr,tpr,threshold = roc_curve(y_test, y_pred)
# 得到AUC值
roc_auc = auc(fpr,tpr)

# 绘制ROC曲线
lw = 2
plt.figure(figsize=(7,5));
# 设置假阳率为横坐标，真阳率为纵坐标
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()
```

<center><img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210411231627620.png" style="zoom:67%;" /></center>

## 补充

### 模型的误差产生的机制

• 误差（Error）：模型预测结果与真实结果之间的差异
• 偏差（bias）：模型的**训练误差**叫做偏差
• 方差（Variance）：训练误差和测试误差的差异大小叫方差

## 精度与召回率应用于分类问题

对于一个给定类，精度和召回率的不同组合如下：

- 高精度+高召回率：模型能够很好地检测该类；
- 高精度+低召回率：模型不能很好地检测该类，但是在它检测到这个类时，判断结果是高度可信的；
- 低精度+高召回率：模型能够很好地检测该类，但检测结果中也包含其他类的点；
- 低精度+低召回率：模型不能很好地检测该类。

