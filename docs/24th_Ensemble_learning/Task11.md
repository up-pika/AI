# Task11 XGBoost算法与实战

# 1 XGBoost

* **XGBoost---陈天奇**

  * GBDT的改进与工程实现
  * 本质还是GBDT，但是目标函数不同
  * 力争把速度和效率发挥到极致，所以叫X(Extreme) GBoost

* **核心思想**

  * 不断地进行特征分裂添加树（学习树模型函数），拟合上一个树模型的残差
  *  训练得到k个树模型后，用每个样本的特征在k个树模型中叶子节点的分数作为样本的预测值
  * 累加每棵树对应的分数为最终的学习结果

* **算法步骤**

  **1) 数据集**
  $$
  \mathcal{D}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}\left(|\mathcal{D}|=n, \mathbf{x}_{i} \in \mathbb{R}^{m}, y_{i} \in \mathbb{R}\right)
  $$
  **2) 构造目标函数**
  $$
  \begin{aligned}
  &\mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right)\\
  &其中，\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)为loss function，\sum_{k} \Omega\left(f_{k}\right)为正则化项
  \end{aligned}
  $$
  **3) 叠加式的训练(Additive Training)**                                      

  给定样本$x_i$，$\hat{y}_i^{(0)} = 0$(初始预测)，$\hat{y}_i^{(1)} = \hat{y}_i^{(0)} + f_1(x_i)$，$\hat{y}_i^{(2)} = \hat{y}_i^{(0)} + f_1(x_i) + f_2(x_i) = \hat{y}_i^{(1)} + f_2(x_i)$.......以此类推，可以得到：$ \hat{y}_i^{(K)} = \hat{y}_i^{(K-1)} + f_K(x_i)$  ，其中，$ \hat{y}_i^{(K-1)} $ 为前K-1棵树的预测结果，$ f_K(x_i)$ 为第K棵树的预测结果                           

  因此，目标函数可以分解为：                                        
  $$
  \mathcal{L}^{(K)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(K-1)}+f_{K}\left(\mathrm{x}_{i}\right)\right)+\sum_{k} \Omega\left(f_{k}\right)
  $$
  由于正则化项也可以分解为前K-1棵树的复杂度加第K棵树的复杂度，因此：
  $$
  \mathcal{L}^{(K)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(K-1)}+f_{K}\left(\mathrm{x}_{i}\right)\right)+\sum_{k=1} ^{K-1}\Omega\left(f_{k}\right)+\Omega\left(f_{K}\right)
  $$
  其中$\sum\_{k=1} ^{K-1}\Omega\left(f\_{k}\right)$在模型构建到第K棵树的时候已经固定，无法改变，因此是一个已知的常数，可以在最优化的时候省去，故：                     
  $$
  \mathcal{L}^{(K)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(K-1)}+f_{K}\left(\mathrm{x}_{i}\right)\right)+\Omega\left(f_{K}\right)
  $$
  **4) 泰勒二阶展开式近似目标函数**
  $$
  \mathcal{L}^{(K)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(K-1)}\right)+g_{i} f_{K}\left(\mathrm{x}_{i}\right)+\frac{1}{2} h_{i} f_{K}^{2}\left(\mathrm{x}_{i}\right)\right]+\Omega\left(f_{K}\right)
  $$
  其中，$g\_{i}=\partial\_{\hat{y}(t-1)} l\left(y\_{i}, \hat{y}^{(t-1)}\right)$和$h\_{i}=\partial\_{\hat{y}^{(t-1)}}^{2} l\left(y\_{i}, \hat{y}^{(t-1)}\right)$                                                       补充：泰勒级数；在数学中，泰勒级数（英语：Taylor series）用无限项连加式——级数来表示一个函数，这些相加的项由函数在某一点的导数求得。具体的形式如下：                          
  $$
  f(x)=\frac{f\left(x_{0}\right)}{0 !}+\frac{f^{\prime}\left(x_{0}\right)}{1 !}\left(x-x_{0}\right)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !}\left(x-x_{0}\right)^{2}+\ldots+\frac{f^{(n)}\left(x_{0}\right)}{n !}\left(x-x_{0}\right)^{n}+......
  $$
  由于$\sum\_{i=1}^{n}l\left(y\_{i}, \hat{y}^{(K-1)}\right)$在模型构建到第K棵树的时候已经固定，无法改变，因此是一个已知的常数，可以在最优化的时候省去，故：                               
  $$
  \tilde{\mathcal{L}}^{(K)}=\sum_{i=1}^{n}\left[g_{i} f_{K}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{K}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{K}\right)
  $$
  **5) 如何定义一棵树**

  * 四个基本概念

    * 令样本所在的节点位置为$q(x)$

    * 落在节点$j$上的样本记为$I\_j = ${$i|q(x\_j)=j$}

    * 每个节点的预测值为$w\_{q(x)}$

    * 树的复杂度为$\Omega(f_K)$,可以由叶子节点的个数以及节点函数值来构建，即包含两部分，一个是树里面叶子节点的个数T；一个是树上叶子节点的得分w的L2模平方（对w进行L2正则化，相当于针对每个叶结点的得分增加L2平滑，目的是为了避免过拟合），则树的复杂度可以写成：
      $$
      \Omega(f_K)=\gamma T+\frac1 2\lambda \sum^T_{j=1}w_j^2
      $$

  目标函数用以上符号替代后：                                      
  $$
  \begin{aligned}
  \tilde{\mathcal{L}}^{(K)} &=\sum_{i=1}^{n}\left[g_{i} f_{K}\left(\mathrm{x}_{i}\right)+\frac{1}{2} h_{i} f_{K}^{2}\left(\mathrm{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\
  &=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T
  \end{aligned}
  $$
  由于我们的目标就是最小化目标函数，现在的目标函数化简为一个关于w的二次函数：
  $$
  \tilde{\mathcal{L}}^{(K)}=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T
  $$
  根据二次函数求极值的公式：$y=ax^2+bx+c$求极值，对称轴在$x=-\frac{b}{2 a}$，极值为$y=\frac{4 a c-b^{2}}{4 a}$，因此：                                       
  $$
  w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}
  $$
  以及
  $$
  \tilde{\mathcal{L}}^{(K)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T
  $$
  **6) 如何寻找树得形状** 

  上面的讨论是在基于树的形状已经确定的前提下计算$w$和$\mathcal{L}$。但实际上需要借助决策树学习的方式，使用目标函数的变化来作为分裂节点的标准，找到树的形状

  分割节点的标准为$max\{\tilde{\mathcal{L}}^{(old)} - \tilde{\mathcal{L}}^{(new)} \}$，即：                               
  $$
  \mathcal{L}_{\text {split }}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma
  $$
  **6.1 精确贪心分裂算法---树的形状**

  XGBoost在生成新树的过程中，最**基本的操作**是**节点分裂**。节点分裂中最重要的环节是找到**最优特征及最优切分点**, 然后将叶子节点按照最优特征和最优切分点进行分裂。选取最优特征和最优切分点的一种思路如下：

  * 从深度为0 的树开始，对每个叶节点枚举所有的可用特征
  * 针对每个特征，把属于该节点的训练样本根据该特征值进行升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的分裂收益
  * 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，在该节点上分裂出左右两个新的叶节点，并为每个新节点关联对应的样本集
  * 回到第 1 步，递归执行到满足特定条件为止

  该算法称为精确贪心算法，它是一种启发式算法, 因为在节点分裂时只选择当前最优的分裂策略, 而非全局最优的分裂策略

  缺点：当数据不能完全加载到内存时，非常低效且耗时

  **6.2 基于直方图的近似算法---树的形状**

  * 思想：对某一特征寻找最优切分点时，首先对该特征的所有切分点按分位数 (如百分位) **分桶**, 得到一个**候选切分点集**。特征的每一个切分点都可以分到对应的分桶; 然后，对每个桶**计算特征统计G和H**得到**直方图**, G为该桶内所有样本一阶特征统计g之和, H为该桶内所有样本二阶特征统计h之和; 最后，选择所有候选特征及候选切分点中对应桶的特征统计**收益最大**的作为最优特征及最优切分点

  * 算法过程

    1) 对于每个特征 $k=1,2, \cdots, m,$ 按分位数对特征 $k$ 分桶 $\Theta,$ 可得候选切分点集和为：
    $$
    S_{k}=\{S_{k1}, S_{k2}, \cdots, S_{kl}\}^1
    $$
    2) 对于每个特征 $k=1,2, \cdots, m,$ 有：                           
    $$
    \begin{array}{l}
    G_{k v} \leftarrow=\sum_{j \in\left\{j \mid s_{k, v} \geq \mathbf{x}_{j k}>s_{k, v-1\;}\right\}} g_{j} \\
    H_{k v} \leftarrow=\sum_{j \in\left\{j \mid s_{k, v} \geq \mathbf{x}_{j k}>s_{k, v-1\;}\right\}} h_{j}
    \end{array}
    $$
    3) 类似精确贪心算法，依据梯度统计找到最大增益的候选切分点

  * 两种候选切分点的构建策略：全局策略和本地策略

    * 全局策略：学习每**棵树前**就提出候选切分点，并在每次分裂时都采用这种分割；（在树构建的初始阶段对每一个特征确定一个候选切分点的集合, 并在该树每一层的节点分裂中均采用此集合计算收益, 整个过程候选切分点集合不改变）
    * 本地策略则是在每一次节点分裂时**均重新**确定候选切分点。全局策略需要更细的分桶才能达到本地策略的精确度, 但全局策略在选取候选切分点集合时比本地策略更简单

# 1.1 XGBoost 与 GBDT的区别

GBDT是机器学习算法，XGBoost是该算法的工程实现，且具有如下优化：

1. 算法上的优化[^1]

* **正则化**：在使用CART作为基分类器时，xgboost在代价函数里加入了正则项，用于控制模型得复杂度，防止过拟合，提高模型的泛化能力。正则项里包含了树的叶子节点的个数、每个叶子节点上输出的score的L2模的平方和
* **列抽样**：XGBoost借鉴了随机森林的做法，支持列抽样，不仅防止过拟合，还能减少计算
* **缺失值处理**：对于特征的值有缺失的样本，XGBoost采用稀疏感知算法自动学习出它的分裂方向且加快节点分裂的速度
* 节点分裂的方式不同，gbdt是用gini系数，xgboost是经过优化推导后的
* **精度更高**：GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数
* **灵活性更强：**GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，（使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题））。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导

2. 工程上的优化

* xgboost工具支持**并行**。注意xgboost的并行并不是tree粒度的并行，xgboost的并行是在**特征粒度**上的。我们知道，决策树的学习最耗时的一个步骤就是对特征值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复的使用这个结构，大大减小计算量。这个block结构也使得并行成为可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行
* 有效利用硬件资源。这是通过在每个线程中分配内部缓冲区来存储梯度统计信息来实现缓存感知来实现的

XGBoost的**缺点**

1. 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集
2. 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存

# 2 LightGBM

* 提出目的：轻量级（Light）的梯度提升机（GBM），用于解决 GDBT 在海量数据中遇到的问题，以便其可以更好更快地用于工业实践中，其相对 XGBoost 具有训练速度快、内存占用低的特点
* 针对XGBoost的缺点，LightGBM的解决方案
  * 单边梯度抽样算法
  * 直方图算法
  * 互斥特征捆绑算法
  * 基于最大深度的 Leaf-wise 的垂直生长算法
  * 类别特征最优分割
  * 特征并行和数据并行
  * 缓存优化
* 相对于XGBoost的优点
  * 内存更小
    * XGBoost 使用预排序后需要记录特征值及其对应样本的统计值的索引，而 LightGBM 使用了直方图算法将特征值转变为 bin 值，且不需要记录特征到样本的索引，将空间复杂度从 $O(2*$#$data)$降低为 $O($#$bin)$，极大的减少了内存消耗
    * LightGBM 采用了直方图算法将存储特征值转变为存储 bin 值，降低了内存消耗
    * LightGBM 在训练过程中采用互斥特征捆绑算法减少了特征数量，降低了内存消
  * 速度更快
    * LightGBM 采用了直方图算法将遍历样本转变为遍历直方图，极大的降低了时间复杂度
    * LightGBM 在训练过程中采用单边梯度算法过滤掉梯度小的样本，减少了大量的计算
    * LightGBM 采用了基于 Leaf-wise 算法的增长策略构建树，减少了很多不必要的计算量
    * LightGBM 采用优化后的特征并行、数据并行方法加速计算，当数据量非常大的时候还可以采用投票并行的策略
    * LightGBM 对缓存也进行了优化，增加了 Cache hit 的命中率

# 3 实战练习

## 3.1 XGBoost 

1 分类

```python
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load datasets
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# parameters
params = {'booster': 'gbtree',
          'objective': 'multi:softmax',
          'num_class': 3,
          'gamma': 0.1, 
          'max_depth': 6,
          'lambda':2,
          'subsample': 0.7,
          'colsample_bytree': 0.75, 
          'min_child_weight': 3,
          'silent': 0,
          'eta': 0.1,
          'seed': 1,
          'nthread': 4,}

plst = list(params.items())

dtrain = xgb.DMatrix(X_train, y_train) # 生成数据集格式
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

# test
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print('accuracy: %.2f%%' % (accuracy*100.0))

# feature importance
plot_importance(model)
plt.show()
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210425174846686.png)

2 回归

```python
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

# 加载数据集
boston = load_boston()
X,y = boston.data,boston.target

# XGBoost训练过程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 300
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 显示重要特征
plot_importance(model)
plt.show()
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210425174857467.png)

3 网格搜索调参

```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

iris = load_iris()
X,y = iris.data,iris.target
col = iris.target_names 
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)   # 分训练集和验证集
parameters = {
              'max_depth': [5, 10, 15, 20, 25],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'n_estimators': [500, 1000, 2000, 3000, 5000],
              'min_child_weight': [0, 2, 5, 10, 20],
              'max_delta_step': [0, 0.2, 0.6, 1, 2],
              'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
              'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
             # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

}

xlf = xgb.XGBClassifier(max_depth=10,
            learning_rate=0.01,
            n_estimators=2000,
            # silent=True,
            objective='multi:softmax',
            num_class=3 ,          
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=0.85,
            colsample_bytree=0.7,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=0,
            missing=None)

gs = GridSearchCV(xlf, param_grid=parameters, scoring='accuracy', cv=3)
gs.fit(train_x, train_y)

print("Best score: %0.3f" % gs.best_score_)
print("Best parameters set: %s" % gs.best_params_ )
```

```markdown
Best score: 0.933
Best parameters set: {'max_depth': 5}
```

## 3.2 LightGBM-调参

```python
import lightgbm as lgb
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
canceData=load_breast_cancer()
X=canceData.data
y=canceData.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
 
### 数据转换
print('数据转换')
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,free_raw_data=False)
 
### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.1
          }
 
### 交叉验证(调参)
print('交叉验证')
max_auc = float('0')
best_params = {}
 
# 准确率
print("调参1：提高准确率")
for num_leaves in range(5,100,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
 
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=1,
                            nfold=5,
                            metrics=['auc'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
            
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
            
        if mean_auc >= max_auc:
            max_auc = mean_auc
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth
if 'num_leaves' and 'max_depth' in best_params.keys():          
    params['num_leaves'] = best_params['num_leaves']
    params['max_depth'] = best_params['max_depth']
 
# 过拟合
print("调参2：降低过拟合")
for max_bin in range(5,256,10):
    for min_data_in_leaf in range(1,102,10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf
            
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=1,
                                nfold=5,
                                metrics=['auc'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                    
            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['max_bin']= max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf
if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
    params['max_bin'] = best_params['max_bin']
 
print("调参3：降低过拟合")
for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
    for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
        for bagging_freq in range(0,50,5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq
            
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=1,
                                nfold=5,
                                metrics=['auc'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                    
            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
            if mean_auc >= max_auc:
                max_auc=mean_auc
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq
 
if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
    params['feature_fraction'] = best_params['feature_fraction']
    params['bagging_fraction'] = best_params['bagging_fraction']
    params['bagging_freq'] = best_params['bagging_freq']
 
 
print("调参4：降低过拟合")
for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
    for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]:
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=1,
                            nfold=5,
                            metrics=['auc'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
                
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
        if mean_auc >= max_auc:
            max_auc=mean_auc
            best_params['lambda_l1'] = lambda_l1
            best_params['lambda_l2'] = lambda_l2
if 'lambda_l1' and 'lambda_l2' in best_params.keys():
    params['lambda_l1'] = best_params['lambda_l1']
    params['lambda_l2'] = best_params['lambda_l2']
 
print("调参5：降低过拟合2")
for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    params['min_split_gain'] = min_split_gain
    
    cv_results = lgb.cv(
                        params,
                        lgb_train,
                        seed=1,
                        nfold=5,
                        metrics=['auc'],
                        early_stopping_rounds=10,
                        verbose_eval=True
                        )
            
    mean_auc = pd.Series(cv_results['auc-mean']).max()
    boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
    if mean_auc >= max_auc:
        max_auc=mean_auc
        
        best_params['min_split_gain'] = min_split_gain
if 'min_split_gain' in best_params.keys():
    params['min_split_gain'] = best_params['min_split_gain']
 
print(best_params)
```

```markdown
{'bagging_fraction': 0.7,
'bagging_freq': 30,
'feature_fraction': 0.8,
'lambda_l1': 0.1,
'lambda_l2': 0.0,
'max_bin': 255,
'max_depth': 4,
'min_data_in_leaf': 81,
'min_split_gain': 0.1,
'num_leaves': 10}
```

## 3.3 梯度下降法

1 单变量：$y=x^2$求最低点

```python
import matplotlib.pyplot as plt
import numpy as np
# fx的函数值
def fx(x):
    return x**2

#定义梯度下降算法
def gradient_descent():
    times = 10 # 迭代次数
    alpha = 0.1 # 学习率
    x =10# 设定x的初始值
    x_axis = np.linspace(-10, 10) #设定x轴的坐标系
    fig = plt.figure(1,figsize=(5,5)) #设定画布大小
    ax = fig.add_subplot(1,1,1) #设定画布内只有一个图
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.plot(x_axis,fx(x_axis)) #作图
    
    for i in range(times):
        x1 = x          
        y1= fx(x)  
        print("第%d次迭代：x=%f，y=%f" % (i + 1, x, y1))
        x = x - alpha * 2 * x
        y = fx(x)
        ax.plot([x1,x], [y1,y], 'ko', lw=1, ls='-', color='coral')
    plt.show()

if __name__ == "__main__":
    gradient_descent()
```

```markdown
第1次迭代：x=10.000000，y=100.000000
第2次迭代：x=8.000000，y=64.000000
第3次迭代：x=6.400000，y=40.960000
第4次迭代：x=5.120000，y=26.214400
第5次迭代：x=4.096000，y=16.777216
第6次迭代：x=3.276800，y=10.737418
第7次迭代：x=2.621440，y=6.871948
第8次迭代：x=2.097152，y=4.398047
第9次迭代：x=1.677722，y=2.814750
第10次迭代：x=1.342177，y=1.801440
```

<img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210425171125560.png" alt="" style="zoom:67%;" />

2 多变量：$z = (x-10)^2 + (y-10)^2$求最低点

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#求fx的函数值
def fx(x, y):
    return (x - 10) ** 2 + (y - 10) ** 2

def gradient_descent():
    times = 10 # 迭代次数
    alpha = 0.05 # 学习率
    x = 20 # x的初始值
    y = 20 # y的初始值

    fig = Axes3D(plt.figure()) # 将画布设置为3D
    axis_x = np.linspace(0, 20, 100)#设置X轴取值范围
    axis_y = np.linspace(0, 20, 100)#设置Y轴取值范围
    axis_x, axis_y = np.meshgrid(axis_x, axis_y) #将数据转化为网格数据
    z = fx(axis_x,axis_y)#计算Z轴数值
    fig.set_xlabel('X', fontsize=14)
    fig.set_ylabel('Y', fontsize=14)
    fig.set_zlabel('Z', fontsize=14)
    fig.view_init(elev=60,azim=300)#设置3D图的俯视角度，方便查看梯度下降曲线
    fig.plot_surface(axis_x, axis_y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow')) #作出底图
    
    for i in range(times):
        x1 = x        
        y1 = y         
        f1 = fx(x, y)  
        print("第%d次迭代：x=%f，y=%f，fxy=%f" % (i + 1, x, y, f1))
        x = x - alpha * 2 * (x - 10)
        y = y - alpha * 2 * (y - 10)
        f = fx(x, y)
        fig.plot([x1, x], [y1, y], [f1, f], 'ko', lw=2, ls='-')
    plt.show()

if __name__ == "__main__":
    gradient_descent()
```

<img src="http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210425171337557.png" alt="" style="zoom:67%;" />

# 4 补充

## 4.1 梯度下降法 牛顿法和泰勒展开式的关系

* **泰勒展开式**
  $$
  \begin{aligned}
  f(x) &= \sum^\infty_{n=0}\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n \\
  &=\frac{f(x_0)}{0!}+\frac{f^\prime(x_0)}{1!}(x-x_0)+\frac{f^{\prime\prime}(x_0)}{2!}(x-x_0)^2+...+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n
  \end{aligned}
  $$

* **梯度下降法**---利用一阶泰勒展开式

  * 设计下一个$x^{t+1}$是从上一个$x^t$沿着某一方向走一小步$\Delta x$得到的。此处的关键问题就是：这一小步的方向是朝向哪里？

  $$
  \begin{aligned}
  f(x) = f(x_0)+f^\prime(x_0)(x-x_0)\\
  (x-x_0)为极小的矢量，等于步长\eta \\
  x\leftarrow x_0-\eta f^\prime(x_0)
  \end{aligned}
  $$

* **牛顿法的两种用法**

  * **求函数的根**---利用**一阶**泰勒展开式求解

    例如，求函数$f(x)=0$的根，先随机选个初始值$f(x\_0)=x\_0$，然后利用泰勒一阶展开式迭代求$x\_{n+1}$，公式为：
    $$
    \begin{aligned}
    &f(x) = f(x_0)+f^\prime(x_0)(x-x_0) \\
    &令f(x)=0\ \Rightarrow\ x= x_0-\frac{f(x_0)}{f^\prime(x_0)}\\
    &一直迭代有一般化公式 \Rightarrow x_{n+1}= x_n-\frac{f(x_n)}{f^\prime(x_n)}\\
    当|x_{n+1}-x_n|&<\epsilon （所设的容忍误差）时，迭代结束，x_{n+1}为函数的近似解
    \end{aligned}
    $$

  * **优化算法**：利用**二阶**泰勒展开式求解

    * 问题描述：假设有一个凸优化问题$min\_xf(x)$；即要找到一个$x$最小化$f(x)$
    * 对于凸优化问题，$f(x)$的最小值点就是$f(x)$的极值点，即满足导数$f^\prime(x)=0$的点
    * 将优化问题转换为求$f^\prime(x)=0$的根的问题

  $$
  \begin{aligned}
  &f(x) = f(x_0) + f^\prime(x_0)(x-x_0)+\frac 1 2f^{\prime\prime}(x_0)(x-x_0)^2\\
  &令f^\prime(x)=0\ \Rightarrow\ f^\prime(x_0)+f^{\prime\prime}(x_0)(x-x_0)=0求解得到 \\
  &x = x_0-\frac{f^\prime(x_0)}{f^{\prime\prime}(x_0)}\\
  &一直迭代有一般化公式 \Rightarrow x_{n+1} = x_n-\frac{f^\prime(x_n)}{f^{\prime\prime}(x_n)}\\
  当|x_{n+1}-x_n|&<\epsilon （所设的容忍误差）时，迭代结束，x_{n+1}为f^\prime(x_n)=0的近似解
  \end{aligned}
  $$

  * 优点：牛顿法比梯度下降法收敛更快
  * 缺点：牛顿法求极值需要求Hessian矩阵及其逆矩阵，计算量较大

* **对比梯度下降法与牛顿法**

  * 梯度下降法为一阶优化算法，牛顿法为二阶优化算法

  * 梯度下降法简单易实现，但迭代步骤较多；牛顿法可以通过较少的迭代找到极值，但是计算复杂（需要计算hessian矩阵以及它的逆）

    * 解释为什么牛顿法相对于梯度下降法需要较少的迭代？

      比如需要找到下山的一条最短的路径，梯度下降法每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大

# 参考资料

[^1]: https://blog.csdn.net/weixin_39210914/article/details/109512504