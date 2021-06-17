# Task04 评估与调优超参数

## 1  知识梳理

### 1.1 参数与超参数

* 参数：可以使用最小二乘法或者梯度下降法等最优化算法优化出来的数，如神经网络模型的参数，SVM的支持向量
* 超参数：无法使用最小二乘法或者梯度下降法等最优化算法优化出来的数，如模型的学习率，正则化惩罚因子

| 序号 | 参数                   | 超参数                       |
| ---- | ---------------------- | ---------------------------- |
| 1    | 定义了可使用的模型     | 用于帮助估计模型参数         |
| 2    | 由训练数据估计得到     | 其值无法从数据中估计         |
| 3    | 保存为学习模型的一部分 | 被调整为给定的预测建模问题   |
| 4    | 模型内部的配置变量     | 模型外部的配置               |
| 5    | 不由编程者手动设置     | 由人工指定，可使用启发式设置 |

### 1.2 网格搜索 GridSearchCV()

* 调用函数：`sklearn.model_selection.GridSearchCV`, 官网链接：[1](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV)

* 思想：

  * 将所有的超参数选择列出来分别做**排列组合**；
  * 然后针对每组超参数**分别建立**一个模型；
  * 选择测试误差最小的那组超参数

  从超参数空间寻找最优超参数，即从参数排列网格中找到最优的节点，因而称为网格搜索(暴力搜索)

### 1.3 随机搜索 RandomizedSearchCV()

* 调用函数：`sklearn.model_selection.RandomizedSearchCV`  ，官网链接：[2](*scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomizedsearchcv#sklearn.model_selection.RandomizedSearchCV)
* 随机搜索法结果比稀疏网格法较好（有时也会极差，需要权衡）
* 参数的随机搜索中的每个参数都是从可能的参数值的分布中采样的
* 优点：
  * 独立于参数数量和可能的值来选择计算成本
  * 添加不影响性能的参数不会降低效率

## 2 实战练习

练习内容：使用SVR结合管道进行调优

### 2.1 评价未调参的SVR

```python
# 评价未调参的SVR模型
from sklearn.svm import SVR # 引入 SVR 类
from sklearn.pipeline import make_pipeline # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler # SV基于距离计算，需要对数据进行标准化
from sklearn.model_selection import GridSearchCV # 引入网格搜索调优
from sklearn.model_selection import cross_val_score # 引入 k 折交叉验证
from sklearn import datasets

boston = datasets.load_boston() # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
pipe_SVR = make_pipeline(StandardScaler(), SVR())
score1 = cross_val_score(estimator=pipe_SVR, X=X, y=y, scoring='r2', cv=10) # 10折交叉验证
print('CV accuracy: %.3f+/-%.3f' % ((np.mean(score1)),np.std(score1)))
```

```markdown
CV accuracy: 0.187+/-0.649
```

### 2.2 基于网格搜索的SVR调参

```python
from sklearn.pipeline import Pipeline
pipe_svr = Pipeline([("StandardScaler",StandardScaler()), ("svr",SVR())])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
# 注意__是指两个下划线，一个下划线会报错
param_grid = [{"svr__C":param_range,"svr__kernel":["linear"]},  {"svr__C":param_range,"svr__gamma":param_range,"svr__kernel":["rbf"]}]
# 10折交叉验证
gs = GridSearchCV(estimator=pipe_svr, param_grid = param_grid, scoring = 'r2', cv = 10)
gs = gs.fit(X,y)
print("网格搜索最优得分：",gs.best_score_)
print("网格搜索最优参数组合：\n",gs.best_params_)
```

```mark
网格搜索最优得分： 0.6081303070817127
网格搜索最优参数组合：
 {'svr__C': 1000.0, 'svr__gamma': 0.001, 'svr__kernel': 'rbf'}
```

### 2.3 基于随机搜索的SVR调参

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform # 引入均匀分布设置参数
pipe_svr = Pipeline([("StandarScaler",StandardScaler()),("svr",SVR())])

# 构建连续参数的分布
distributions = dict(svr__C=uniform(loc=1.0, scale=4),    # 注意__是指两个下划线
                     svr__kernel=["linear","rbf"],         # svr的核                          
                     svr__gamma=uniform(loc=0, scale=4))   
# 10 折交叉验证
rs = RandomizedSearchCV(estimator=pipe_svr,
                       param_distributions=distributions,
                       scoring='r2',
                       cv=10)
rs = rs.fit(X, y)
print("随机搜索最优得分：", rs.best_score_)
print("随机搜索最优参数组合：\n", rs.best_params_)
```

```markdown
随机搜索最优得分： 0.298932330651552
随机搜索最优参数组合：
 {'svr__C': 4.844225732066931, 'svr__gamma': 0.3629509311668113, 'svr__kernel': 'linear'}
```