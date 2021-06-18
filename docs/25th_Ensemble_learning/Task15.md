# Task15 集成学习案例-蒸汽量预测

# 1 集成学习案例二 （蒸汽量预测）

## 1.1 背景介绍

火力发电的基本原理是：燃料在燃烧时加热水生成蒸汽，蒸汽压力推动汽轮机旋转，然后汽轮机带动发电机旋转，产生电能。在这一系列的能量转化中，影响发电效率的核心是锅炉的燃烧效率，即燃料燃烧加热水产生高温高压蒸汽。锅炉的燃烧效率的影响因素很多，包括锅炉的可调参数，如燃烧给量，一二次风，引风，返料风，给水水量；以及锅炉的工况，比如锅炉床温、床压，炉膛温度、压力，过热器的温度等。我们如何使用以上的信息，根据锅炉的工况，预测产生的蒸汽量，来为我国的工业届的产量预测贡献自己的一份力量呢？

所以，该案例是使用以上工业指标的特征，进行蒸汽量的预测问题。由于信息安全等原因，我将使用的是经脱敏后的锅炉传感器采集的数据（采集频率是分钟级别）。

## 1.2 数据信息
数据分成训练数据（train.txt）和测试数据（test.txt），其中字段”V0”-“V37”，这38个字段是作为特征变量，”target”作为目标变量。我们需要利用训练数据训练出模型，预测测试数据的目标变量。

## 1.3 评价指标
最终的评价指标为均方误差MSE，即：
$$Score = \frac{1}{n} \sum_1 ^n (y_i - y ^*)^2$$

# 2 实践练习

## baseline

1. import package

```python
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
```

1. 加载数据

```python
data_train = pd.read_csv('train.txt', sep='\t')
data_test = pd.read_csv('test.txt', sep='\t')
# 合并训练数据和测试数据
data_train["origin"] = "train" # 新增一列表示数据为train标签
data_test["origin"] = "test" 
data_all = pd.concat([data_train, data_test], axis=0, ignore_index=True) # 将test直接加到train下面的行

# 显示前5条数据
data_all.head()
```

1. 数据探索性分析

   传感器数据为连续变量，所以用kdeplot(核密度估计图）进行数据的初步分析

```python
for col in data_all.columns[0:-2]: # 第一列到倒数第3列，含第3列 col 为str 类型
    # 核密度估计
    # data_all["V0"][(data_all["oringin"] == "train")] # 只取train训练集的各个特征
    g = sns.kdeplot(data_all[col][(data_all["origin"] == "train")], color="Red", shade=True)
    g = sns.kdeplot(data_all[col][(data_all["origin"] == "test")], color="Blue", shade=True)
    g.set_xlabel(col) 
    g.set_ylabel("Frequency")
    g = g.legend(["train", "test"])
    plt.show()   
```

​		(图片略)从图中可以看出特征"V5","V9","V11","V17","V22","V28"中训练集数据分布和测试集数据分布不均，所以删除这些特征数据

```python
for col in ["V5","V9","V11","V17","V22","V28"]:
    g = sns.kdeplot(data_all[col][data_all["origin"] == "train"], color="Red", shade = True) 
    g = sns.kdeplot(data_all[col][(data_all["origin"] == "test")], ax =g, color="Blue", shade= True)
    # 有无括号都可以 (data_all["oringin"] == "test")
    g.set_xlabel(col)
    g.set_ylabel("Frequency")
    g = g.legend(["train","test"])
    plt.show()

data_all.drop(["V5","V9","V11","V17","V22","V28"],axis=1,inplace=True)
```

​	（图片略）

​		特征间相关性

```python
data_train1 = data_all[data_all["origin"] == "train"].drop("origin", axis=1) # 删除“oringin”列
plt.figure(figsize=(20, 16)) # 指定绘图对象宽度和高度
columns_all = data_train1.columns.tolist() # 列表头 列表类型
mcorr = data_train1[columns_all].corr(method="spearman") # 相关系数矩阵，即给出了任意两个变量之间的相关系数
mask = np.zeros_like(mcorr, dtype=np.bool) # 构造与mcorr同维数矩阵为bool型
mask[np.triu_indices_from(mask)] = True # 角分线右侧为True 
cmap = sns.diverging_palette(220, 10, as_cmap=True) # 返回matplotlib colormap 对象，调色板 220，10为图的正负范围的锚定色调
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
plt.show()
```

![](http://huilan-typora-picture.oss-cn-beijing.aliyuncs.com/img/image-20210516222738295.png)

​	删除低相关性特征

```python
# 只剩25个特征
threshold = 0.1
corr_matrix = data_train1.corr().abs()
drop_col=corr_matrix[corr_matrix["target"]<threshold].index
data_all.drop(drop_col,axis=1,inplace=True)
```

​	归一化

```python
cols_numeric=list(data_all.columns)
cols_numeric.remove("origin")

def scale_minmax(col):
    return (col-col.min()) / (col.max() - col.min())

scale_cols = [col for col in cols_numeric if col != "target"]
data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax, axis=0) # 按列（对每个特征进行归一化）
data_all[scale_cols].describe() # 数据统计信息
```

1. 特征工程

   Box-Cox变换

   ```python
   fcols = 6
   frows = len(cols_numeric) -1
   plt.figure(figsize=(4*fcols, 4*frows)) # 对每个特征都做4个图 
   i = 0
   
   for var in cols_numeric: # 对每个特征做6个图 原始图，qq图，相关性图
       if var != 'target':
           dat = data_all[[var,'target']].dropna() # data_all[[]] 获得含列索引的dataframe，dropna()：删除缺失值
   
           i += 1
           plt.subplot(frows, fcols, i)
           sns.distplot(dat[var], fit=stats.norm);
           plt.title(var + 'Original')
           plt.xlabel('')
           
           # qq图
           i += 1
           plt.subplot(frows,fcols,i)
           _=stats.probplot(dat[var], plot=plt)
           plt.title('skew='+'{:.4f}'.format(stats.skew(dat[var])))
           plt.xlabel('')
           plt.ylabel('')
           
           # 相关性图
           i+=1
           plt.subplot(frows,fcols,i)
           plt.plot(dat[var], dat['target'],'.',alpha=0.5)
           plt.title('corr='+'{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1])) # 相关系数
           
           i+=1
           plt.subplot(frows,fcols,i)
           trans_var, lambda_var = stats.boxcox(dat[var].dropna()+1)
           trans_var = scale_minmax(trans_var)      
           sns.distplot(trans_var , fit=stats.norm);
           plt.title(var+' Tramsformed')
           plt.xlabel('')
           
           i+=1
           plt.subplot(frows,fcols,i)
           _=stats.probplot(trans_var, plot=plt)
           plt.title('skew='+'{:.4f}'.format(stats.skew(trans_var)))
           plt.xlabel('')
           plt.ylabel('')
           
           i+=1
           plt.subplot(frows,fcols,i)
           plt.plot(trans_var, dat['target'],'.',alpha=0.5)
           plt.title('corr='+'{:.2f}'.format(np.corrcoef(trans_var,dat['target'])[0][1]))
   ```

   ```python
   # 进行Box-Cox变换
   cols_transform=data_all.columns[0:-2]
   for col in cols_transform:   
       # transform column
       data_all.loc[:,col], _ = stats.boxcox(data_all.loc[:,col]+1)
   print(data_all.target.describe())
   plt.figure(figsize=(12,4))
   plt.subplot(1,2,1)
   sns.distplot(data_all.target.dropna() , fit=stats.norm);
   plt.subplot(1,2,2)
   _=stats.probplot(data_all.target.dropna(), plot=plt)
   ```

   ```
   # 对预测值进行对数变换
   sp = data_train.target
   data_train.target1 =np.power(1.5,sp)
   print(data_train.target1.describe())
   
   plt.figure(figsize=(12,4))
   plt.subplot(1,2,1)
   sns.distplot(data_train.target1.dropna(),fit=stats.norm);
   plt.subplot(1,2,2)
   _=stats.probplot(data_train.target1.dropna(), plot=plt)
   ```

2. 模型训练与预测

   ```python
   # function to get training samples
   def get_training_data():
       # extract training samples
       # from sklearn.model_selection import train_test_split
       df_train = data_all[data_all["origin"]=="train"]
       df_train["label"]=data_train.target1
       # split SalePrice and features
       y = df_train.target
       X = df_train.drop(["origin","target","label"],axis=1)
       X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.3,random_state=100)
       return X_train,X_valid,y_train,y_valid
   
   # extract test data (without SalePrice)
   def get_test_data():
       df_test = data_all[data_all["origin"]=="test"].reset_index(drop=True)
       return df_test.drop(["origin","target"],axis=1)
       
   # mse, rmse
   from sklearn.metrics import make_scorer
   # metric for evaluation
   def rmse(y_true, y_pred):
       diff = y_pred - y_true
       sum_sq = sum(diff**2)    
       n = len(y_pred)   
       return np.sqrt(sum_sq/n)
   
   def mse(y_ture,y_pred):
       return mean_squared_error(y_ture,y_pred)
   
   # scorer to be used in sklearn model fitting
   rmse_scorer = make_scorer(rmse, greater_is_better=False) 
   
   #输入的score_func为记分函数时，该值为True（默认值）；输入函数为损失函数时，该值为False
   mse_scorer = make_scorer(mse, greater_is_better=False)
   
   # 再次对隐藏的不易于发现的异常值进行删除
   # function to detect outliers based on the predictions of a model
   def find_outliers(model, X, y, sigma=3):
   
       # predict y values using model
       model.fit(X, y)
       y_pred = pd.Series(model.predict(X), index=y.index)
           
       # calculate residuals between the model prediction and true y values
       resid = y - y_pred
       mean_resid = resid.mean()
       std_resid = resid.std()
   
       # calculate z statistic, define outliers to be where |z|>sigma
       z = (resid - mean_resid) / std_resid  # 归一化 变成正态分布
       outliers = z[abs(z) > sigma].index
       
       # print and plot the results
       print('R2=', model.score(X,y))
       print('rmse=', rmse(y, y_pred))
       print("mse=", mean_squared_error(y,y_pred))
       print('---------------------------------------')
   
       print('mean of residuals:', mean_resid)
       print('std of residuals:', std_resid)
       print('---------------------------------------')
   
       print(len(outliers),'outliers:')
       print(outliers.tolist())
   
       plt.figure(figsize=(15,5))
       ax_131 = plt.subplot(1,3,1)
       plt.plot(y,y_pred,'.')
       plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
       plt.legend(['Accepted','Outlier'])
       plt.xlabel('y')
       plt.ylabel('y_pred');
   
       ax_132=plt.subplot(1,3,2)
       plt.plot(y,y-y_pred,'.')
       plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
       plt.legend(['Accepted','Outlier'])
       plt.xlabel('y')
       plt.ylabel('y - y_pred');
   
       ax_133=plt.subplot(1,3,3)
       z.plot.hist(bins=50,ax=ax_133)
       z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
       plt.legend(['Accepted','Outlier'])
       plt.xlabel('z')
       
       return outliers
   ```

   ```python
   # get training data
   X_train, X_valid,y_train,y_valid = get_training_data()
   test=get_test_data()
   
   # find and remove outliers using a Ridge model
   outliers = find_outliers(Ridge(), X_train, y_train)
   X_outliers=X_train.loc[outliers]
   y_outliers=y_train.loc[outliers]
   X_t=X_train.drop(outliers)
   y_t=y_train.drop(outliers)
   ```

   ```python
   def get_trainning_data_omitoutliers():
       #获取训练数据省略异常值
       y=y_t.copy() # 浅拷贝
       X=X_t.copy()
       return X,y
   ```

   ```python
   def train_model(model, param_grid=[], X=[], y=[], 
                   splits=5, repeats=5):
   
       # 获取数据
       if len(y)==0:
           X,y = get_trainning_data_omitoutliers()
           
       # 交叉验证
       rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
       
       # 网格搜索最佳参数
       if len(param_grid) > 0:
           gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                                  scoring="neg_mean_squared_error",
                                  verbose=1, return_train_score=True)
   
           # 训练
           gsearch.fit(X,y)
   
           # 最好的模型
           model = gsearch.best_estimator_        
           best_idx = gsearch.best_index_
   
           # 获取交叉验证评价指标
           grid_results = pd.DataFrame(gsearch.cv_results_)
           cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
           cv_std = grid_results.loc[best_idx,'std_test_score']
   
       # 没有网格搜索  
       else:
           grid_results = []
           cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)
           cv_mean = abs(np.mean(cv_results))
           cv_std = np.std(cv_results)
       
       # 合并数据
       cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
   
       # 预测
       y_pred = model.predict(X)
       
       # 模型性能的统计数据        
       print('----------------------')
       print(model)
       print('----------------------')
       print('score=',model.score(X,y))
       print('rmse=',rmse(y, y_pred))
       print('mse=',mse(y, y_pred))
       print('cross_val: mean=',cv_mean,', std=',cv_std)
       
       # 残差分析与可视化
       y_pred = pd.Series(y_pred,index=y.index)
       resid = y - y_pred
       mean_resid = resid.mean()
       std_resid = resid.std()
       z = (resid - mean_resid)/std_resid    
       n_outliers = sum(abs(z)>3)
       outliers = z[abs(z)>3].index
       
       return model, cv_score, grid_results
   ```

   ```python
   # 定义训练变量存储数据
   opt_models = dict()
   score_models = pd.DataFrame(columns=['mean','std'])
   splits=5
   repeats=5
   
   model = 'Ridge'  #可替换，见案例分析一的各种模型
   opt_models[model] = Ridge() #可替换，见案例分析一的各种模型
   alph_range = np.arange(0.25,6,0.25)
   param_grid = {'alpha': alph_range}
   
   opt_models[model],cv_score,grid_results = train_model(opt_models[model], param_grid=param_grid, 
                                                 splits=splits, repeats=repeats)
   
   cv_score.name = model
   score_models = score_models.append(cv_score)
   
   plt.figure()
   plt.errorbar(alph_range, abs(grid_results['mean_test_score']),
                abs(grid_results['std_test_score'])/np.sqrt(splits*repeats))
   plt.xlabel('alpha')
   plt.ylabel('score')
   
   # 预测函数
   def model_predict(test_data,test_y=[]):
       i=0
       y_predict_total=np.zeros((test_data.shape[0],))
       for model in opt_models.keys():
           if model!="LinearSVR" and model!="KNeighbors":
               y_predict=opt_models[model].predict(test_data)
               y_predict_total+=y_predict
               i+=1
           if len(test_y)>0:
               print("{}_mse:".format(model),mean_squared_error(y_predict,test_y))
       y_predict_mean=np.round(y_predict_total/i,6)
       if len(test_y)>0:
           print("mean_mse:",mean_squared_error(y_predict_mean,test_y))
       else:
           y_predict_mean=pd.Series(y_predict_mean)
           return y_predict_mean
   ```

3. 保存结果

   ```python
   y_ = model_predict(test)
   y_.to_csv('predict.txt',header = None,index = False)
   ```

   
