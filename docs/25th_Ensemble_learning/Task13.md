# Task13 Stacking 算法与实战

# 1 Stacking

* 主要思想：训练次级模型来学习使用基层学习器的预测结果

* 核心：在训练集上进行预测，从而构建更高层的学习器

* 训练过程 （以 5 折交叉验证为例）

  1. 划分训练集和测试集，并将训练集进一步随机且大致均匀的分为 5 份，交叉验证过程中，随机选4份为训练集，剩余一份为验证集

  2. 选择基模型，在划分后的训练集上进行交叉验证训练模型，同时在测试集上进行预测

     细节：在每一次交叉验证包含两个过程

     a）训练基模型。利用 4 份训练数据训练模型，用训练的模型在验证集上进行预测

     b)  在此过程进行的同时，利用相同的 4 份数据训练出的模型，在测试集上预测；如此重复 5 次，将验证集上的 5 次结果按行叠加为 1 列，将测试集上的 5 次结果取均值融合为 1 列

  3. 使用 k 个分类器重复 2 过程。将分别得到 k 列验证集的预测结果，k 列测试集的预测结果

  4. 训练 3 过程得到的数据。将 k 列验证集的预测结果和训练集真实 label 构成次级模型的训练集（次级训练集），将 k 列测试集预测结果作为测试集，训练第二层模型

  ![ 单个基模型训练过程](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi9zdGFja2luZy5qcGc)                  	                                                                    图片来源[^1]

  ![两个基模型训练过程](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210509121951847.png)

  ​                                                                 图片来源[^2]

* **基模型的选择**：基模型之间的**相关性要尽量小**，同时基模型之间的**性能**表现**不能差距太大**

* **元模型**（meta learner, 次级模型）选择[^3]

  1. 统计方法：投票法和加权平均法
  2. 经典易解释的机器学习算法：LR, 决策树（C4.5）
  3. 非线性（non-linear）机器学习算法：GBDT，XGBoost, k-NN, RF
  4. 无监督方法：聚类k-means

* stacking 调参

  * stacking模型

    * 主要思想：训练次级模型来学习使用基层学习器的预测结果

    * 核心：在训练集上进行预测，从而构建更高层的学习器

    * 训练过程 （以 5 折交叉验证为例）

      1. 划分训练集和测试集，并将训练集进一步随机且大致均匀的分为 5 份，交叉验证过程中，随机选4份为训练集，剩余一份为验证集

      2. 选择基模型，在划分后的训练集上进行交叉验证训练模型，同时在测试集上进行预测

         细节：在每一次交叉验证包含两个过程

         a）训练基模型。利用 4 份训练数据训练模型，用训练的模型在验证集上进行预测

         b)  在此过程进行的同时，利用相同的 4 份数据训练出的模型，在测试集上预测；如此重复 5 次，将验证集上的 5 次结果按行叠加为 1 列，将测试集上的 5 次结果取均值融合为 1 列

      3. 使用 k 个分类器重复 2 过程。将分别得到 k 列验证集的预测结果，k 列测试集的预测结果

      4. 训练 3 过程得到的数据。将 k 列验证集的预测结果和训练集真实 label 构成次级模型的训练集（次级训练集），将 k 列测试集预测结果作为测试集，训练第二层模型

      ![ 单个基模型训练过程](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/aHR0cHM6Ly9yYXcuZ2l0aHVidXNlcmNvbnRlbnQuY29tL0tsYXVzemhhby9waWN0dXJlL21hc3Rlci9waWN0dXJlL2NvbW1vbi9zdGFja2luZy5qcGc)                  	                                                                    图片来源[^1]

      ![两个基模型训练过程](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210509121951847.png)

      ​                                                                 图片来源[^2]

    * **基模型的选择**：基模型之间的**相关性要尽量小**，同时基模型之间的**性能**表现**不能差距太大**

    * **元模型**（meta learner, 次级模型）选择[^3]：尽量选简单的线性模型

      1. 统计方法：投票法和加权平均法
      2. 经典易解释的机器学习算法：LR, 决策树（C4.5）
      3. 非线性（non-linear）机器学习算法：GBDT，XGBoost, k-NN, RF
      4. 无监督方法：聚类k-means

    * stacking 调参

      * stacking模型调参包括对**基模型和元模型**进行调参
        * 对于**基模型**，因为在生成元特征的时候要使用相同的K折划分，所以需要使用**交叉验证+网格搜索**来调参时最好使用与生成元特征相同的 K-Fold
        * 对于**元模型**的调参，使用**交叉验证+网格搜索**来调参时，为了降低过拟合的风险，我们最好也使用与元特征生成时**同样**的 K-Fold

# 2 Blending vs Stacking

* **交叉验证方法不同**：Blending 训练基模型时，并没有使用K-Fold方法，而是使用**留出法**(Stacking使用了K-Fold)，比如说留出20%的数据，这部分数据不加入基础模型的训练，而是在基础模型都训练好了以后，去预测这部分没有参与训练的数据得到预测概率，然后以各个模型的预测概率作为最终模型的输入特征。而Stacking 是使用**K-Fold交叉验证法**，按照n折对数据迭代划分，每次划分都有n-1份数据作为训练集，1份数据作为验证集用于获得训练好的模型预测后的结果，更新到次级学习器的输入特征中。经过n次后，那么就会产生最终的预测分数，当然有很多数据都是重复使用的

* Blending 优点

  * 将集成的知识迁移到简单的分类器上
  * **比stacking简单（不用k 折交叉验证获得stacker feature）**
  * 避开了**信息泄露**的问题 （第一层中针对训练集中的训练阶段和预测阶段使用了不一样的数据集）
  * **训练时间缩短**，因为blending 在前面训练基模型的时候就只用了较少的数据，在次级模型时只使用一部分数据做留出集进行验证，总体上时间缩短了
  * 优化variance（即模型的鲁棒性）

* Blending 缺点

  * 使用很少的数据，容易造成过拟合
  * blending没有stacking稳健（因为没有K-Fold交叉验证）

* Stacking 优点

  * 使用K折交叉验证，数据量充足，能够充分利用数据，精度更高
  * 主要优化偏差（模型的精确性）

* Stacking 缺点

  * 可能会信息泄露（在K-Fold当中除了第一轮以外都是拿着用来训练好的模型去预测之前用来训练这个模型的数据，会有这个风险）


# 3 实战练习

### 3.1 Stacking

1. 实现

```python
#使用一个基模型做K折交叉验证，因为验证集的预测结果和测试集的预测结果都是多行一列的结构，这里是先存储为一行多列，最后进行转置
def get_oof(clf, x_train, y_train, x_test):
 oof_train = np.zeros((ntrain,))  
 oof_test = np.zeros((ntest,))
 oof_test_skf = np.empty((NFOLDS, ntest))  #NFOLDS行，ntest列的二维array
 for i, (train_index, test_index) in enumerate(kf): #循环NFOLDS次
     x_tr = x_train[train_index]
     y_tr = y_train[train_index]
     x_te = x_train[test_index]
     clf.fit(x_tr, y_tr)
     oof_train[test_index] = clf.predict(x_te)
     oof_test_skf[i, :] = clf.predict(x_test)  #固定行填充，循环一次，填充一行
 oof_test[:] = oof_test_skf.mean(axis=0)  #axis=0,按列求平均，最后保留一行
 return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)  #转置，从一行变为一列
```

2. 实例-iris数据集

```python
## 1. 简单堆叠 3 折交叉验证分类
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier

RANDOM_SEED = 2021

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], # 第一层分类器
                            meta_classifier=lr, # 第二层分类器
                            random_state=RANDOM_SEED)

print('3-fold cross validation: \n')

for clf, label in zip([clf1, clf2, clf3, sclf], ['KNN', 'RF', 'Naive_Bayes', 'StackingClassifier']):
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy') 
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
```

```markdown
output:
3-fold cross validation: 
Accuracy: 0.91 (+/- 0.01) [KNN]
Accuracy: 0.95 (+/- 0.02) [RF]
Accuracy: 0.91 (+/- 0.02) [Naive_Bayes]
Accuracy: 0.95 (+/- 0.02) [StackingClassifier]
```

```python
## 画出决策边界
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec # 指定放置子图的网格的规格
import itertools  # 创建循环用迭代器的函数模块
 
gs = gridspec.GridSpec(2, 2)# 分为2行2列
fig = plt.figure(figsize=(10, 8))
for clf, label, grd in zip([clf1, clf2, clf3, sclf], 
                         ['KNN', 'RF', 'Naive_Bayes',
                          'StackingClassifier'], 
                           # 笛卡尔积，repeat 指定重复生成序列的次数，在这里产生(0,0)(0,1)(1, 0)(1,1)
                           itertools.product([0, 1], repeat=2) # 在gs的四个网格坐标中画决策边界
                          ):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)
plt.show()
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210509140851045.png)

```markdown
使用第一层所有基分类器所产生的类别概率值作为meta-classfier的输入。需要在StackingClassifier 中增加一个参数设置：use_probas = True。
另外，还有一个参数设置average_probas = True,那么这些基分类器所产出的概率值将按照列被平均，否则会拼接。

例如：

基分类器1：predictions=[0.2,0.2,0.7]

基分类器2：predictions=[0.4,0.3,0.8]

基分类器3：predictions=[0.1,0.4,0.6]

1）若use_probas = True，average_probas = True，

则产生的meta-feature 为：[0.233, 0.3, 0.7]

2）若use_probas = True，average_probas = False，

则产生的meta-feature 为：
[0.2,0.2,0.7,0.4,0.3,0.8,0.1,0.4,0.6]
```

```python
# 2.使用概率作为元特征
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            use_probas=True,  # 
                            meta_classifier=lr,
                            random_state=42)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
```

```markdown
output:
3-fold cross validation:
Accuracy: 0.93 (+/- 0.02) [KNN]
Accuracy: 0.95 (+/- 0.02) [Random Forest]
Accuracy: 0.91 (+/- 0.02) [Naive Bayes]
Accuracy: 0.93 (+/- 0.02) [StackingClassifier]
```

```python
# 3. 堆叠5折CV分类与网格搜索(结合网格搜索调参优化)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier

# Initializing models

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], 
                            meta_classifier=lr,
                            random_state=42)

params = {'kneighborsclassifier__n_neighbors': [1, 5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta_classifier__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
grid.fit(X, y)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
```

```markdown
output:
0.933 +/- 0.03 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 10}
0.933 +/- 0.03 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
0.913 +/- 0.02 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 10}
0.920 +/- 0.02 {'kneighborsclassifier__n_neighbors': 1, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
0.940 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 10}
0.940 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
0.940 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 10}
0.947 +/- 0.03 {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
Best parameters: {'kneighborsclassifier__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
Accuracy: 0.95
```

```python
# 如果打算多次使用回归算法，需要在参数网格中添加一个附加的数字后缀
from sklearn.model_selection import GridSearchCV

# Initializing models

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf1, clf2, clf3], 
                            meta_classifier=lr,
                            random_state=RANDOM_SEED)

params = {'kneighborsclassifier-1__n_neighbors': [1, 5], # 数字后缀
          'kneighborsclassifier-2__n_neighbors': [1, 5],
          'randomforestclassifier__n_estimators': [10, 50],
          'meta_classifier__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
grid.fit(X, y)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
```

```markdown
output:
0.940 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 10}
0.940 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
0.913 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 10}
0.920 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
0.947 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 10}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
0.920 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 10}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
0.947 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 10}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
0.920 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 10}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 1, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 10}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
0.947 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 10}
0.953 +/- 0.02 {'kneighborsclassifier-1__n_neighbors': 5, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 10.0, 'randomforestclassifier__n_estimators': 50}
Best parameters: {'kneighborsclassifier-1__n_neighbors': 1, 'kneighborsclassifier-2__n_neighbors': 5, 'meta_classifier__C': 0.1, 'randomforestclassifier__n_estimators': 50}
Accuracy: 0.95
```

```python
# 4.在不同特征子集上运行的分类器的堆叠
##不同的1级分类器可以适合训练数据集中的不同特征子集。以下示例说明了如何使用scikit-learn管道和ColumnSelector：
from sklearn.datasets import load_iris
from mlxtend.classifier import StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X = iris.data
y = iris.target

pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)),  # 选择第0,2列
                      LogisticRegression())
pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)),  # 选择第1,2,3列
                      LogisticRegression())

sclf = StackingCVClassifier(classifiers=[pipe1, pipe2], 
                            meta_classifier=LogisticRegression(),
                            random_state=42)

sclf.fit(X, y)
```

```markdown
output:
StackingCVClassifier(classifiers=[Pipeline(steps=[('columnselector',
                                                   ColumnSelector(cols=(0, 2))),
                                                  ('logisticregression',
                                                   LogisticRegression())]),
                                  Pipeline(steps=[('columnselector',
                                                   ColumnSelector(cols=(1, 2,
                                                                        3))),
                                                  ('logisticregression',
                                                   LogisticRegression())])],
                     meta_classifier=LogisticRegression(), random_state=42)
```

```python
# 5.ROC曲线 decision_function
### 像其他scikit-learn分类器一样，它StackingCVClassifier具有decision_function可用于绘制ROC曲线的方法。
### 请注意，decision_function期望并要求元分类器实现decision_function。
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

iris = datasets.load_iris()
X, y = iris.data[:, [0, 1]], iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_SEED)

clf1 =  LogisticRegression()
clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
clf3 = SVC(random_state=RANDOM_SEED)
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(sclf)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210509141405939.png)

# 参考资料

[^1]: [(14条消息) 模型融合Blending 和 Stacking_笔记小屋-CSDN博客_stacking融合](https://blog.csdn.net/randompeople/article/details/103452483)
[^2]: [机器学习分记:利用Stacking提升模型预测性能 (qq.com)](https://mp.weixin.qq.com/s/SBIecY07n-3q4zAEQSelqQ)
[^3]: [今我来思，堆栈泛化（Stacked Generalization） - 简书 (jianshu.com)](https://www.jianshu.com/p/46ccf40222d6)