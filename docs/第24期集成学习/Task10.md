# Task10 前向分布算法与梯度提升决策树

# 1 GBDT

* 提出该算法的**动机**：当使用一般的损失函数时，前向分步算法难以求得各基分类器的权重和参数

  * 解决方案：当使用除MSE和指数损失函数外的损失函数时，利用**最速下降**近似方法，**关键**在于利用损失函数的**负梯度**在当前学习基模型的值作为回归问题提升树算法中的**残差的近似值**，拟合一个回归树
  * 当损失函数为MSE，即GBDT为回归树模型，可以直接用残差$r=y-f\_{m-1}(x)$拟合回归树
  * 当损失函数为指数损失函数时，GBDT退化为Adaboost，可以用前向分步算法

* **定义**：GBDT是一种通过采用**加法模型**（即基函数的线性组合），以及不断减小训练过程产生的**残差**（真实值和预测值之间的差值）来达到将数据分类或者回归的算法[^1]
  
  * GBDT又称为：MART（Multiple Additive Regression Tree)，是一种迭代的决策树算法
  
* **思想**：每个新的弱学习器的建立是为了使得之前弱学习器的**残差往梯度方向减少**，然后把弱学习器进行**累加**得到强学习器

* **所选取的三要素**

  * 叠加CART决策树（**只能用回归树**；但是可以通过选取不同的损失函数用于解决分类和回归问题）
  * 损失函数：指数损失函数或者MSE（退化为Adaboost) 以及其他连续可微的损失函数(MAE, Huber，分位数损失函数)
  * 优化算法：前向分步算法

* **算法过程**

  * 初始化$f_0(x)$
  * 对 $m=1,2,...,M$ 轮中每一轮都计算残差（损失函数为MSE：残差；指数损失函数：Adaboost求法；一般损失函数：当前树模型的负梯度值）
  * 拟合残差学习一个学习树模型
  * 更新$f\_{m}(x)=f\_{m-1}(x)+T\left(x ; \Theta\_{m}\right)$
  * 得到回归树$f\_{M}(x)=\sum\_{m=1}^{M} T\left(x ; \Theta\_{m}\right)$  

* **回归问题的提升树算法步骤（损失函数为MSE）**

  * 输入：数据集
    $$
    T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R}
    $$
    输出：最终的提升树$f\_{M}(x)=T(x;\Theta\_m)$ 

    损失函数：$L(y,f(x))$
    
  * 初始化$f\_0(x) = 0$                        
       - 对$m = 1,2,...,M$：                  
          - 计算每个样本的残差:
            $$
            r_{m i}=y_{i}-f_{m-1}\left(x_{i}\right), \quad i=1,2, \cdots, N
            $$
          
          - 拟合残差$r\_{mi}$学习一棵回归树，得到$T\left(x ; \Theta\_{m}\right)$                        
          
          - 更新$f\_{m}(x)=f\_{m-1}(x)+T\left(x ; \Theta\_{m}\right)$
          
       - 得到最终的回归问题的提升树：$f\_{M}(x)=\sum\_{m=1}^{M} T\left(x ; \Theta\_{m}\right)$  

* **使用一般损失函数的提升树算法步骤**

  * 输入：数据集

  $$
  T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R}
  $$

  ​		输出：最终的提升树$\hat{f}(x)$ 

  ​		损失函数：$L(y,f(x))$

  * 初始化 $f\_0(x)=argmin\_c\sum\_{i=1}^NL(y,c)$

  * 对$m=1,2,...,M$

    * a) 对$i=1,2,..,N$,计算损失函数对树模型所求的负梯度在当前树模型的值，用于作为残差$r\_{mi}$的近似值
      $$
      r_{mi}=-\Big[\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}\Big]
      $$

    * b) 对$r\_{mi}$拟合一个回归树，得到第$m$棵树的叶结点区域$R\_{mj},j=1,2,...,J$

    * c) 对 $j=1,2,...,J$计算
      $$
      c_{mj} = argmin_c \sum_{x_i\in R_{mj}}L(y_i,f_{m-1}(x_i)+c)
      $$

    * d) 更新 
      $$
      f_m(x) = f_{m-1}(x)+\sum_{j=1}^Jc_{mj}I(x\in R_{mj})
      $$
      

  * 得到回归树
    $$
    \hat{f}(x) = f_M(x)=\sum_{m=1}^M\sum_{j=1}^Jc_{mj}I(x\in R_{mj})
    $$

* **针对Boosting需要解决的两个问题，GBDT算法采用的策略**
  
  1. 将**残差**作为下一个弱学习器的训练数据，每个新的弱学习器的建立都是为了使得之前弱学习器的残差往梯度方向减少
  2. 将弱学习器联合起来，使用**累加**机制代替平均投票机制
  
* **优点**
  *  适合低维度数据
  * 灵活处理各种类型的数据，包括连续值和离散值；几乎可以用于所有回归问题，也可用于二分类问题（设定阈值，大于阈值为正例，反之为负例）
  * 不需要归一化与标准化也能表现良好
  * 使用健壮的损失函数，对异常值的鲁棒性非常强，如Huber损失函数
  * 可以筛选特征
  
* **缺点**
  
  * 由于弱学习器件存在依赖关系，难以并行训练数据 （但可通过自采样的SGBT达到部分并行）

# 2 前向分步算法

前向分步算法：可以用于**分类**问题和**回归**问题

（1）加法模型

* 表达式：$f(x)=\sum\_{m=1}^{M} \beta\_{m} b\left(x ; \gamma\_{m}\right)$其中，$b\left(x ; \gamma\_{m}\right)$为即基本分类器，$\gamma\_{m}$为基本分类器的参数，$\beta\_m$为基本分类器的权重

* 在给定训练数据以及损失函数$L(y, f(x))$的条件下，学习加法模型$f(x)$就是：                        
  $$
  \min _{\beta_{m}, \gamma_{m}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
  $$

* 前向分步算法的**基本思路**是：因为学习的是加法模型，如果从前向后，每一步只优化一个基函数及其系数，逐步逼近目标函数，那么就可以降低优化的复杂度。

  具体而言，每一步只需要优化：

$$
\min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, \beta b\left(x_{i} ; \gamma\right)\right)
$$
（2）前向分步算法步骤

* 给定数据集
$$
   \begin{aligned}
   T=\left\{\left(x\_{1}, y\_{1}\right),\left(x\_{2}, y\_{2}\right), \cdots,\left(x\_{N}, y\_{N}\right)\right\},x_{i}& \in \mathcal{X} \subseteq \mathbf{R}^{n}\\\\
y\_{i} &\in \mathcal{Y}=\{+1,-1\}
   \end{aligned}
$$
   损失函数$L(y, f(x))$

   基函数集合$\{b(x;\gamma)\}$

   目的：输出加法模型$f(x)$                         

   - 初始化：$f_{0}(x)=0$                           
   - 对$m = 1,2,...,M$:                     
   
      (a) 极小化损失函数：
      $$
      \left(\beta_{m}, \gamma_{m}\right)=\arg \min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right)
      $$
      得到参数$\beta\_{m}$与$\gamma\_{m}$                                           
   
      (b) 更新：                          
      $$
      f_{m}(x)=f_{m-1}(x)+\beta_{m} b\left(x ; \gamma_{m}\right)
      $$
   - 得到加法模型：                           
$$
   f(x)=f_{M}(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)
$$

这样，前向分步算法将同时求解从$m=1$到$M$的所有参数$\beta\_{m}$，$\gamma\_{m}$的优化问题简化为逐次求解各个$\beta\_{m}$，$\gamma\_{m}$的问题

Adaboost是前行分步算法的特例，其损失函数为指数损失函数，且由基本分类器组成的一个加法模型

# 3 实战练习

## 3.1 GBDT源码

源码链接[^2]（搬砖工）

```python
import abc
import math
import logging
import pandas as pd
from GBDT.decision_tree import Tree
from GBDT.loss_function import SquaresError, BinomialDeviance, MultinomialDeviance
from GBDT.tree_plot import plot_tree, plot_all_trees,plot_multi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self, loss, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__()     
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.trees = {}
        self.f_0 = {}
        self.is_log = is_log
        self.is_plot = is_plot

    def fit(self, data):
        """
        :param data: pandas.DataFrame, the features data of train training   
        """
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        self.f_0 = self.loss.initialize_f_0(data)
        # 对 m = 1, 2, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees+1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/NO.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            # 计算负梯度--对于平方误差来说就是残差
            logger.info(('------------------构建第%d颗树-------------' % iter))
            self.loss.calculate_residual(data, iter)
            target_name = 'res_' + str(iter)
            self.trees[iter] = Tree(data, self.max_depth, self.min_samples_split,
                                    self.features, self.loss, target_name, logger)
            self.loss.update_f_m(data, self.trees, iter, self.learning_rate, logger)
            if self.is_plot:
                plot_tree(self.trees[iter], max_depth=self.max_depth, iter=iter)
        # print(self.trees)
        if self.is_plot:
            plot_all_trees(self.n_trees)


class GradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__(SquaresError(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees+1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_value'] = data[f_m_name]


class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__(BinomialDeviance(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_proba'] = data[f_m_name].apply(lambda x: 1 / (1 + math.exp(-x)))
        data['predict_label'] = data['predict_proba'].apply(lambda x: 1 if x >= 0.5 else 0)


class GradientBoostingMultiClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__(MultinomialDeviance(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot)

    def fit(self, data):
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]
        # 获取所有类别
        self.classes = data['label'].unique().astype(str)
        # 初始化多分类损失函数的参数 K
        self.loss.init_classes(self.classes)
        # 根据类别将‘label’列进行one-hot处理
        for class_name in self.classes:
            label_name = 'label_' + class_name
            data[label_name] = data['label'].apply(lambda x: 1 if str(x) == class_name else 0)
            # 初始化 f_0(x)
            self.f_0[class_name] = self.loss.initialize_f_0(data, class_name)
        # print(data)
        # 对 m = 1, 2, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees + 1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/NO.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.info(('-----------------------------构建第%d颗树-----------------------------' % iter))
            # 这里计算负梯度整体计算是为了计算p_sum的一致性
            self.loss.calculate_residual(data, iter)
            self.trees[iter] = {}
            for class_name in self.classes:
                target_name = 'res_' + class_name + '_' + str(iter)
                self.trees[iter][class_name] = Tree(data, self.max_depth, self.min_samples_split,
                                                    self.features, self.loss, target_name, logger)
                self.loss.update_f_m(data, self.trees, iter, class_name, self.learning_rate, logger)
            if self.is_plot:
                    plot_multi(self.trees[iter], max_depth=self.max_depth, iter=iter)
        if self.is_plot:
            plot_all_trees(self.n_trees)

    def predict(self, data):
        """
        此处的预测的实现方式和生成树的方式不同，
        生成树是需要每个类别的树的每次迭代需要一起进行，外层循环是iter，内层循环是class
        但是，预测时树已经生成，可以将class这层循环作为外循环，可以节省计算成本。
        """
        for class_name in self.classes:
            f_0_name = 'f_' + class_name + '_0'
            data[f_0_name] = self.f_0[class_name]
            for iter in range(1, self.n_trees + 1):
                f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
                f_m_name = 'f_' + class_name + '_' + str(iter)
                data[f_m_name] = \
                    data[f_prev_name] + \
                    self.learning_rate * data.apply(lambda x:
                                                    self.trees[iter][class_name].root_node.get_predict_value(x), axis=1)

        data['sum_exp'] = data.apply(lambda x:
                                     sum([math.exp(x['f_' + i + '_' + str(iter)]) for i in self.classes]), axis=1)

        for class_name in self.classes:
            proba_name = 'predict_proba_' + class_name
            f_m_name = 'f_' + class_name + '_' + str(iter)
            data[proba_name] = data.apply(lambda x: math.exp(x[f_m_name]) / x['sum_exp'], axis=1)
        # TODO: log 每一类的概率
        data['predict_label'] = data.apply(lambda x: self._get_multi_label(x), axis=1)

    def _get_multi_label(self, x):
        label = None
        max_proba = -1
        for class_name in self.classes:
            if x['predict_proba_' + class_name] > max_proba:
                max_proba = x['predict_proba_' + class_name]
                label = class_name
        return label
```

### 3.1.2 sklearn中GBDT用于回归和分类的函数参数

* GradientBoostingRegressor 

  * 函数

  ```python
  sklearn.ensemble.GradientBoostingRegressor(
  	loss=’ls’,
  	learning_rate=0.1,
  	n_estimators=100,
  	subsample=1.0,
  	criterion='friedman_mse',
  	min_samples_split=2,
  	min_samples_leaf=1,
  	min_weight_fraction_leaf=0.0,
  	max_depth=3,
  	min_impurity_decrease=0.0,
  	min_impurity_split=None,
  	init=None,
  	random_state=None,
  	max_features=None,
  	alpha=0.9,
  	verbose=0,
  	max_leaf_nodes=None,
  	warm_start=False,
  	validation_fraction=0.1, 
      n_iter_no_change=None, 
      tol=0.0001, 
      ccp_alpha=0.0)
  ```

  

  * 参数列表

| 序号 |                       | 参数中文名称         | 可选项及其意义                                               |
| ---- | --------------------- | -------------------- | ------------------------------------------------------------ |
| 1    | loss                  | loss(损失函数)       | ls： 最小平方误差（L2损失函数）<br />lad：最小绝对偏差（L1损失函数）<br />huber：两者结合，用$\alpha$参数决定侧重性；<br />quantile：分位数损失，用于区间预测<br /> |
| 2    | learning_rate         | 学习率               | 每棵决策树的权重衰减系数，也叫作步长；需要权衡树的数量和学习率 |
| 3    | n_estimators          | 基学习器个数         | 基学习器提升次数；太大容易造成过拟合                         |
| 4    | subsample             | 子采样               | 选取样本的比例，取1，则取全部样本；小于1，取部分样本         |
| 5    | criterion             | 评价指标             | friedman_mse; mse:均方误差；mae：平均绝对误差                |
| 6    | min_samples_split     | 分裂最少点           | 定义树中节点分裂所需的最少样本数，避免过拟合；限制子树继续划分； |
| 7    | min_samples_leaf      | 叶子最少样本数       | 限制叶子节点最少的样本数，防止过拟合；如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝 |
| 8    | max_depth             | 树的最大深度         | 防止过拟合，树越深越容易过拟合                               |
| 9    | min_impurity_decrease |                      | 最小的分裂属性纯度，如果增加分裂可以降低该节点的纯度         |
| 10   | min_impurity_split    | 节点划分的最小不纯度 | 限制决策树的增长；如果某节点的不纯度(基于基尼系数）小于这个阈值，则该节点不再生成子节点，即为叶子节点 |
| 11   | init                  |                      | 初始化时的基学习器；                                         |
| 12   | random_state          | 随机数种子           | 固定参数，便于复现                                           |
| 13   | max_features          | 最大特征数           | int：每次分割时考虑特征<br />float：特征百分比<br />None：考虑所有特征；<br />log2：划分时最多考虑log2N个特征<br />sqrt/auto：考虑特征数开方后的特征个数 |
| 14   | alpha                 |                      | 只有*GradientBoostingRegressor*有；当使用Huber or Quantile时，需指定分位数的值 |
| 15   | verbose               | 输出的打印方式       | 0：不输出<br />1：打印特定区域的数的结果<br />>1：打印所有结果 |
| 16   | max_leaf_nodes        | 最大叶子节点数       | 防止过拟合                                                   |
| 17   | warm_start            |                      | 暖启动（预训练）；True：调用上一个解决方案进行拟合，并添加更多的估计值 |
| 18   | validation_fraction   |                      | 验证集的比例，根据早停法迭代次数                             |
| 19   | n_iter_no_change      |                      | 早停法判断：验证集分数持续不提升的迭代次数阈值               |
| 20   | tol                   |                      | 早停法判断： 当n_iter_no_change所设置的整数迭代次数中没有tol改变，则停止训练 |
| 21   | ccp_alpha             |                      | 剪枝                                                         |



* GradientBoostingClassifier 

  ```python
  class sklearn.ensemble.GradientBoostingClassifier(*, 
                                                   loss='deviance',
                                                    learning_rate=0.1,
                                                    n_estimators=100, 
                                                    subsample=1.0,
                                                    criterion='friedman_mse',
                                                    min_samples_split=2,
                                                    min_samples_leaf=1,
                                                    min_weight_fraction_leaf=0.0,
                                                    max_depth=3,
                                                    min_impurity_decrease=0.0,
                                                    min_impurity_split=None,
                                                    init=None, 
                                                    random_state=None,
                                                    max_features=None,
                                                    verbose=0, 
                                                    max_leaf_nodes=None,
                                                    warm_start=False,
                                                    validation_fraction=0.1,
                                                    n_iter_no_change=None,
                                                    tol=0.0001, 
                                                    ccp_alpha=0.0)
  ```





| 序号 |                       | 参数中文名称         | 可选项及其意义                                               |
| ---- | --------------------- | -------------------- | ------------------------------------------------------------ |
| 1    | loss                  | loss(损失函数)       | deviance：Logistic regression分类的概率<br />exponential：指数损失函数 |
| 2    | learning_rate         | 学习率               | 每棵决策树的权重衰减系数，也叫作步长；需要权衡树的数量和学习率 |
| 3    | n_estimators          | 基学习器个数         | 基学习器提升次数；太大容易造成过拟合                         |
| 4    | subsample             | 子采样               | 选取样本的比例，取1，则取全部样本；小于1，取部分样本         |
| 5    | criterion             | 评价指标             | friedman_mse; mse:均方误差；mae：平均绝对误差                |
| 6    | min_samples_split     | 分裂最少点           | 定义树中节点分裂所需的最少样本数，避免过拟合；限制子树继续划分； |
| 7    | min_samples_leaf      | 叶子最少样本数       | 限制叶子节点最少的样本数，防止过拟合；如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝 |
| 8    | max_depth             | 树的最大深度         | 防止过拟合，树越深越容易过拟合                               |
| 9    | min_impurity_decrease |                      | 最小的分裂属性纯度，如果增加分裂可以降低该节点的纯度         |
| 10   | min_impurity_split    | 节点划分的最小不纯度 | 限制决策树的增长；如果某节点的不纯度(基于基尼系数）小于这个阈值，则该节点不再生成子节点，即为叶子节点 |
| 11   | init                  |                      | 初始化时的基学习器；                                         |
| 12   | random_state          | 随机数种子           | 固定参数，便于复现                                           |
| 13   | max_features          | 最大特征数           | int：每次分割时考虑特征<br />float：特征百分比<br />None：考虑所有特征；<br />log2：划分时最多考虑log2N个特征<br />sqrt/auto：考虑特征数开方后的特征个数 |
| 14   | verbose               | 输出的打印方式       | 0：不输出<br />1：打印特定区域的数的结果<br />>1：打印所有结果 |
| 15   | max_leaf_nodes        | 最大叶子节点数       | 防止过拟合                                                   |
| 16   | warm_start            |                      | 暖启动（预训练）；True：调用上一个解决方案进行拟合，并添加更多的估计值 |
| 17   | validation_fraction   |                      | 验证集的比例，根据早停法迭代次数                             |
| 18   | n_iter_no_change      |                      | 早停法判断：验证集分数持续不提升的迭代次数阈值               |
| 19   | tol                   |                      | 早停法判断： 当n_iter_no_change所设置的整数迭代次数中没有tol改变，则停止训练 |
| 20   | ccp_alpha             |                      | 剪枝率                                                       |



## 3.2 GBDT应用

```python
# import packages
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))                        
```

```markdown
5.009154859960321
```

```python
# import packages
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)                                              
```

```markdown
0.43848663277068134
```

# 4 补充

## 4.1 为什么GDBT只能用回归树而不能用分类树？

* **回归树与分类树定义**：目标变量可以采用一组**离散值**的树模型称为**分类树**(常用的分类树算法有ID3、C4.5、CART)，而目标变量可以采用**连续值**（通常是实数）的决策树被称为**回归树**(如CART算法)

* 解答：因为GDBT的核心是最小化残差（梯度值），残差是连续值，回归树才有残差的概念，分类树的结果为离散值；

  补充：GDBT是加法模型，对于回归树算法来说最重要的是寻找最佳的划分点，那么回归树中的可划分点包含了所有特征的所有可取的值；在分类树中最佳划分点的判别标准是熵或者基尼系数，都是用纯度来衡量的，且分类树的分类结果做加法没有实际意义

## 4.2 损失函数（LAE, LSE, Huber, Quantile Loss）

* **L1损失函数, 最小绝对值偏差（LAD)，最小绝对值误差（LAE)**

$$
L = \sum^n_{i=1}|y_i-f(x_i)|
$$

* **平均绝对误差（Mean Absolute Error，MAE）**

  * 意义：能更好地反映预测值与真实值误差的实际情况
    $$
    MAE=\frac1N∑_{i=1}^N|y_i−f(x_i)|
    $$

* **L2 损失函数，MSE，最小平方误差（LSE)**

$$
L = \sum^n_{i=1}(y_i-f(x_i))^2
$$

* **均方误差 (Mean Square Error, MSE)**

  * 意义：计算机器学习和深度学习中模型的预测值与真实值之间的距离；是各样本数据偏离真实值的距离平方和的平均值
    $$
    MSE = \frac{\sum_{i=1}^N e_i^2}{N}=\frac1N\sum (y_i-f(x_i))^2
    $$
  
* MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度
  
* **L1和L2都存在的问题**[^3]

  若数据中90%的样本对应的目标值为150，剩下10%在0到30之间

  那么使用MAE作为损失函数的模型可能会忽视10%的异常点，而对所有样本的预测值都为150，因为模型会按中位数来预测

  MSE的模型则会给出很多介于0到30的预测值，因为模型会向异常点偏移

* **Hube 损失函数，平滑的平均绝对误差**

  * 针对L1,L2损失函数的缺点提出，Huber对数据中的异常点没有LSE那么敏感
    $$
    L_\delta(y, f(x)) = \left\{\begin{matrix}
    \frac1 2 (y-f(x))^2, |y-f(x)|\le\delta\\
    \delta|y-f(x)|-\frac 1 2\delta^2,|y-f(x)|>\delta
    \end{matrix}
    \right.
    $$

  * 超参数$\delta$ 决定 Huber 对MSE 和 MAE的侧重性；超参数 δ 可以通过交叉验证选取最佳值

    * 当$|y-f(x)|\le\delta$，类似MSE
    * 当$|y-f(x)|>\delta$，类似MAE

* **Quantile Loss**

  *  用于区间预测而不仅是点预测

  * 表达式
    $$
    L_r(y,y^p)=\sum_{i:y_i<y^p_i}(1-\gamma)|y_i-y_i^p|+\sum_{i:y_i\ge y_i^p}\gamma|y_i-y_i^p|
    $$

    * y 为所需的分位数，取值范围(0, 1)

## 4.3 梯度提升与梯度下降法的关系

* 关系：殊途同归
  * 两者都是梯度下降；梯度下降法是直接更新参数来梯度下降；而梯度提升通过累加弱学习器来梯度下降；损失函数为二分之一平方误差，步长为1时，梯度下降法跟梯度提升是等价的
  * 梯度提升是指通过梯度的方式去对模型进行提升，通过前向分步算法提升模型的准确率，减小模型的偏差。梯度下降是为了将损失函数的值降到最低

## 4.4 GBDT 可并行的部分

GBDT整体是串行训练的，但是在每一轮迭代时，有以下几部分可以并行

* **计算（更新）每个样本的负梯度的时候** （第m-1轮迭代完毕时，第m轮的每个样本的负梯度就可以全部更新）

- **分裂挑选最佳特征及其分割点时，对特征计算相应的误差及均值*的时候**（在拟合残差树时某个节点分类时，求各种可能的分类情况时的增益(基尼系数)是可以并行计算的。(GBDT没有做，xgboost做到了，用一个block结构存储这些特征增益)。但对于拟合一整棵残差树，增益是无法并行计算的，下一个节点的最大增益要在它的父节点生成后才可以求）

- **最后预测的过程当中，每个样本将之前的所有树的结果累加的时候**（因为f(最终) = f(初) + f(残1) + f(残2) + …，一个测试样本最终的预测值等于各个子模型预测结果线性相加。各个子模型对于测试样本的预测是可以同时进行的，最后将各个模型的结果累加就是最终的预测值）

注意：能并行的条件是在计算时数据都是已知，不需要得到以往的数据(这里指上一轮迭代得到的数据)才能进行就算。比如，如果求特征2的增益要知道特征1的增益，那这样增益计算就无法并行了[^4]

# 参考资料

[^1]: https://www.cnblogs.com/bnuvincent/p/9693190.html
[^2]: https://github.com/Freemanzxp/GBDT_Simple_Tutorial/blob/master/GBDT/gbdt.py
[^3]: https://www.cnblogs.com/pacino12134/p/11104446.html
[^4]: https://blog.csdn.net/weixin_40363423/article/details/98878459