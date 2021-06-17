# Task14 集成学习案例-幸福感预测

# 1 集成学习案例（幸福感预测）

## 1.1 案例背景

幸福感是一个古老而深刻的话题，是人类世代追求的方向。与幸福感相关的因素成千上万、因人而异，大如国计民生，小如路边烤红薯，都会对幸福感产生影响。这些错综复杂的因素中，我们能找到其中的共性，一窥幸福感的要义吗？另外，在社会科学领域，幸福感的研究占有重要的位置。这个涉及了哲学、心理学、社会学、经济学等多方学科的话题复杂而有趣；同时与大家生活息息相关，每个人对幸福感都有自己的衡量标准。如果能发现影响幸福感的共性，生活中是不是将多一些乐趣；如果能**找到影响幸福感的政策因素**，便能**优化资源配置**来**提升国民的幸福感**。目前社会科学研究注重变量的可解释性和未来政策的落地，主要采用了**线性回归和逻辑回归**的方法，在**收入、健康、职业、社交关系、休闲方式等经济人口因素；以及政府公共服务、宏观经济环境、税负**等宏观因素上有了一系列的推测和发现。
该案例为幸福感预测这一经典课题，希望在现有社会科学研究外有其他维度的算法尝试，结合多学科各自优势，挖掘潜在的影响因素，发现更多可解释、可理解的相关关系。具体来说，此案例就是一个**数据挖掘**类型的比赛——**幸福感预测的baseline**。具体来说，我们需要使用包括个体变量（性别、年龄、地域、职业、健康、婚姻与政治面貌等等）、家庭变量（父母、配偶、子女、家庭资本等等）、社会态度（公平、信用、公共服务等等）等**139维**度的信息来预测其对幸福感的影响。

数据来源于国家官方的《中国综合社会调查（CGSS）》文件中的调查结果中的数据，数据来源可靠可依赖

## 1.2 数据信息

赛题要求使用以上 139 维的特征，使用 8000 余组数据进行对于个人幸福感的预测（预测值为1，2，3，4，5，其中1代表幸福感最低，5代表幸福感最高）。因为考虑到变量个数较多，部分变量间关系复杂，数据分为完整版和精简版两类。本文使用了**完整版**的数据。赛题也给出了index文件中包含每个变量对应的问卷题目，以及变量取值的含义；survey文件中为原版问卷，作为补充以方便理解问题背景。

## 1.3 评价指标

最终的评价指标为均方误差MSE，即：
$$
Score = \frac1 n\sum^n_1(y_i - y_i^\prime)^2
$$
其中 $y\_i$ 为原来的幸福感数值，$y^\prime_i$ 为模型预测的幸福感值

# 2 特征探索性分析和特征工程

## 2.1 数据总览

* 使用describe() 和 info() 分别观察数据的统计信息，比如最大值，最小值，均值，方差，分位数（25%， 50%， 75%），和 数据各个特征的数据类型

## 2.2 数据探索性分析

1. 异常值处理

   对数据中连续出现的负数值进行处理。数据中存在-1、-2、-3、-8这几个异常值，将他们视为有问题的特征，但是不删去

2. 缺失值处理

   首先使用isnull() 使用fillna() 函数
   
3. 时间类型值处理

# 3 实战练习

## 3.1 导入package

```python
import os # 处理文件和目录
import time 
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb


# 分类器
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression  # LR
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.neighbors import KNeighborsClassifier # KNN 分类
from sklearn.naive_bayes import GaussianNB # 高斯贝叶斯分类器
from sklearn.linear_model import SGDClassifier # 随机梯度下降分类器
from sklearn.tree import DecisionTreeClassifier # 决策树

# 回归模型
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.linear_model import BayesianRidge as br # 贝叶斯回归
from sklearn.linear_model import Ridge # L2 回归
from sklearn.linear_model import Lasso # L1 回归
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import ElasticNet as en # 弹性网络具有L1 L2正则效果
from sklearn.kernel_ridge import KernelRidge as kr # 核岭回归

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, f1_score

import logging
import warnings
warnings.filterwarnings('ignore') # 消除warning
```

## 3.2 导入数据集

```python
# 指定某些列为时间类型，这个参数一般搭配下面的date_parser使用
train = pd.read_csv('train.csv', parse_dates=['survey_time'], encoding='latin-1')
test = pd.read_csv('test.csv', parse_dates=['survey_time'], encoding='latin-1') ##latin-1向下兼容ASCII
```

```python
train['happiness'].value_counts()
```

输出：

```markdown
 4    4818
 5    1410
 3    1159
 2     497
 1     104
-8      12
Name: happiness, dtype: int64
```

由结果可以看出预测值中有异常值，需要对异常样本进行样本删除（行删除）

```python
train = train[train["happiness"]!=-8].reset_index(drop=True) # 不想保留原来的index
train_data_copy = train.copy() # 获得删去“happiness”-8的数据
target_col = "happiness" # 目标列
target = train_data_copy[target_col]
del train_data_copy[target_col] # 删去目标列
data = pd.concat([train_data_copy, test], axis=0, ignore_index=True) # 合并
```

## 3.3 数据预处理

1. 对数据中的连续出现的负数进行处理。由于数据中的负数值只有-1，-2，-3，-8这几种数值，所以它们进行分别的操作

```python
## make feature +5
# 负值：-1、-2、-3、-8，将它们视为有问题的特征，但是不删去
def getres1(row):
    return len([x for x in row.values if type(x)==int and x<0]) 

def getres2(row):
    return len([x for x in row.values if type(x)==int and x==-8])

def getres3(row):
    return len([x for x in row.values if type(x)==int and x==-1])

def getres4(row):
    return len([x for x in row.values if type(x)==int and x==-2])

def getres5(row):
    return len([x for x in row.values if type(x)==int and x==-3])

#检查数据
data['neg1'] = data[data.columns].apply(lambda row:getres1(row),axis=1)
data.loc[data['neg1']>20,'neg1'] = 20  #平滑处理

data['neg2'] = data[data.columns].apply(lambda row:getres2(row),axis=1)
data['neg3'] = data[data.columns].apply(lambda row:getres3(row),axis=1)
data['neg4'] = data[data.columns].apply(lambda row:getres4(row),axis=1)
data['neg5'] = data[data.columns].apply(lambda row:getres5(row),axis=1)
```

### 缺失值填充

```python
# 填充缺失值 共25列 去掉4列 填充21列
# 以下的列都是缺省的，视情况填补
data['work_status'] = data['work_status'].fillna(0)
data['work_yr'] = data['work_yr'].fillna(0)
data['work_manage'] = data['work_manage'].fillna(0)
data['work_type'] = data['work_type'].fillna(0)

data['edu_yr'] = data['edu_yr'].fillna(0)
data['edu_status'] = data['edu_status'].fillna(0)

data['s_work_type'] = data['s_work_type'].fillna(0)
data['s_work_status'] = data['s_work_status'].fillna(0)
data['s_political'] = data['s_political'].fillna(0)
data['s_hukou'] = data['s_hukou'].fillna(0)
data['s_income'] = data['s_income'].fillna(0)
data['s_birth'] = data['s_birth'].fillna(0)
data['s_edu'] = data['s_edu'].fillna(0)
data['s_work_exper'] = data['s_work_exper'].fillna(0)

data['minor_child'] = data['minor_child'].fillna(0)
data['marital_now'] = data['marital_now'].fillna(0)
data['marital_1st'] = data['marital_1st'].fillna(0)
data['social_neighbor']=data['social_neighbor'].fillna(0)
data['social_friend']=data['social_friend'].fillna(0)
data['hukou_loc']=data['hukou_loc'].fillna(1) #最少为1，表示户口
data['family_income']=data['family_income'].fillna(66365) #删除问题值后的平均值
```

### 时间有关特征处理

1. 年龄分层处理，即划分年龄段，这里将年龄分为6个区间
2. 计算具体的年龄，需要特征交叉，将出生年月与调查事件交叉获得调查者的真实年龄

```python
## 144+1 =145
data['survey_time'] = pd.to_datetime(data['survey_time'], format='%Y-%m-%d',errors='coerce')#防止时间格式不同的报错errors='coerce‘
data['survey_time'] = data['survey_time'].dt.year # 仅仅是year，方便计算年龄
data['age'] = data['survey_time']-data['birth']
# print(data['age'],data['survey_time'],data['birth'])
#年龄分层 145+1=146
bins = [0,17,26,34,50,63,100]
data['age_bin'] = pd.cut(data['age'], bins, labels=[0,1,2,3,4,5]) 
```

```python
#对‘宗教’处理
data.loc[data['religion']<0,'religion'] = 1 #1为不信仰宗教
data.loc[data['religion_freq']<0,'religion_freq'] = 1 #1为从来没有参加过
#对‘教育程度’处理
data.loc[data['edu']<0,'edu'] = 4 #初中
data.loc[data['edu_status']<0,'edu_status'] = 0
data.loc[data['edu_yr']<0,'edu_yr'] = 0
#对‘个人收入’处理
data.loc[data['income']<0,'income'] = 0 #认为无收入
#对‘政治面貌’处理
data.loc[data['political']<0,'political'] = 1 #认为是群众
#对体重处理
data.loc[(data['weight_jin']<=80)&(data['height_cm']>=160),'weight_jin']= data['weight_jin']*2
data.loc[data['weight_jin']<=60,'weight_jin']= data['weight_jin']*2  #个人的想法，没有60斤的成年人
#对身高处理
data.loc[data['height_cm']<150,'height_cm'] = 150 #成年人的实际情况
#对‘健康’处理
data.loc[data['health']<0,'health'] = 4 #认为是比较健康
data.loc[data['health_problem']<0,'health_problem'] = 4
#对‘沮丧’处理
data.loc[data['depression']<0,'depression'] = 4 
#对‘媒体’处理
data.loc[data['media_1']<0,'media_1'] = 1 #都是从不
data.loc[data['media_2']<0,'media_2'] = 1
data.loc[data['media_3']<0,'media_3'] = 1
data.loc[data['media_4']<0,'media_4'] = 1
data.loc[data['media_5']<0,'media_5'] = 1
data.loc[data['media_6']<0,'media_6'] = 1
#对‘空闲活动’处理
data.loc[data['leisure_1']<0,'leisure_1'] = 1 #可调其他值
data.loc[data['leisure_2']<0,'leisure_2'] = 5
data.loc[data['leisure_3']<0,'leisure_3'] = 3
```

使用众数（代码中使用mode()来实现异常值的修正），由于这里的特征是空闲活动，所以采用众数对于缺失值进行处理比较合理

```python
data.loc[data['leisure_4']<0,'leisure_4'] = data['leisure_4'].mode() #取众数
data.loc[data['leisure_5']<0,'leisure_5'] = data['leisure_5'].mode()
data.loc[data['leisure_6']<0,'leisure_6'] = data['leisure_6'].mode()
data.loc[data['leisure_7']<0,'leisure_7'] = data['leisure_7'].mode()
data.loc[data['leisure_8']<0,'leisure_8'] = data['leisure_8'].mode()
data.loc[data['leisure_9']<0,'leisure_9'] = data['leisure_9'].mode()
data.loc[data['leisure_10']<0,'leisure_10'] = data['leisure_10'].mode()
data.loc[data['leisure_11']<0,'leisure_11'] = data['leisure_11'].mode()
data.loc[data['leisure_12']<0,'leisure_12'] = data['leisure_12'].mode()
data.loc[data['socialize']<0,'socialize'] = 2 #很少
data.loc[data['relax']<0,'relax'] = 4 #经常
data.loc[data['learn']<0,'learn'] = 1 #从不
#对‘社交’处理
data.loc[data['social_neighbor']<0,'social_neighbor'] = 0
data.loc[data['social_friend']<0,'social_friend'] = 0
data.loc[data['socia_outing']<0,'socia_outing'] = 1
data.loc[data['neighbor_familiarity']<0,'social_neighbor']= 4
#对‘社会公平性’处理
data.loc[data['equity']<0,'equity'] = 4
#对‘社会等级’处理
data.loc[data['class_10_before']<0,'class_10_before'] = 3
data.loc[data['class']<0,'class'] = 5
data.loc[data['class_10_after']<0,'class_10_after'] = 5
data.loc[data['class_14']<0,'class_14'] = 2
#对‘工作情况’处理
data.loc[data['work_status']<0,'work_status'] = 0
data.loc[data['work_yr']<0,'work_yr'] = 0
data.loc[data['work_manage']<0,'work_manage'] = 0
data.loc[data['work_type']<0,'work_type'] = 0
#对‘社会保障’处理
data.loc[data['insur_1']<0,'insur_1'] = 1
data.loc[data['insur_2']<0,'insur_2'] = 1
data.loc[data['insur_3']<0,'insur_3'] = 1
data.loc[data['insur_4']<0,'insur_4'] = 1
data.loc[data['insur_1']==0,'insur_1'] = 0
data.loc[data['insur_2']==0,'insur_2'] = 0
data.loc[data['insur_3']==0,'insur_3'] = 0
data.loc[data['insur_4']==0,'insur_4'] = 0
```

取均值进行缺失值的补全（代码实现为means()），因为家庭的收入是连续值，使用均值进行缺失值的补全

```python
#对家庭情况处理
family_income_mean = data['family_income'].mean()
data.loc[data['family_income']<0,'family_income'] = family_income_mean
data.loc[data['family_m']<0,'family_m'] = 2
data.loc[data['family_status']<0,'family_status'] = 3
data.loc[data['house']<0,'house'] = 1
data.loc[data['car']<0,'car'] = 0
data.loc[data['car']==2,'car'] = 0 #变为0和1
data.loc[data['son']<0,'son'] = 1
data.loc[data['daughter']<0,'daughter'] = 0
data.loc[data['minor_child']<0,'minor_child'] = 0
#对‘婚姻’处理
data.loc[data['marital_1st']<0,'marital_1st'] = 0
data.loc[data['marital_now']<0,'marital_now'] = 0
#对‘配偶’处理
data.loc[data['s_birth']<0,'s_birth'] = 0
data.loc[data['s_edu']<0,'s_edu'] = 0
data.loc[data['s_political']<0,'s_political'] = 0
data.loc[data['s_hukou']<0,'s_hukou'] = 0
data.loc[data['s_income']<0,'s_income'] = 0
data.loc[data['s_work_type']<0,'s_work_type'] = 0
data.loc[data['s_work_status']<0,'s_work_status'] = 0
data.loc[data['s_work_exper']<0,'s_work_exper'] = 0
#对‘父母情况’处理
data.loc[data['f_birth']<0,'f_birth'] = 1945
data.loc[data['f_edu']<0,'f_edu'] = 1
data.loc[data['f_political']<0,'f_political'] = 1
data.loc[data['f_work_14']<0,'f_work_14'] = 2
data.loc[data['m_birth']<0,'m_birth'] = 1940
data.loc[data['m_edu']<0,'m_edu'] = 1
data.loc[data['m_political']<0,'m_political'] = 1
data.loc[data['m_work_14']<0,'m_work_14'] = 2
#和同龄人相比社会经济地位
data.loc[data['status_peer']<0,'status_peer'] = 2
#和3年前比社会经济地位
data.loc[data['status_3_before']<0,'status_3_before'] = 2
#对‘观点’处理
data.loc[data['view']<0,'view'] = 4
#对期望年收入处理
data.loc[data['inc_ability']<=0,'inc_ability']= 2
inc_exp_mean = data['inc_exp'].mean()
data.loc[data['inc_exp']<=0,'inc_exp']= inc_exp_mean #取均值

#部分特征处理，取众数（首先去除缺失值的数据）
for i in range(1,9+1):
    data.loc[data['public_service_'+str(i)]<0,'public_service_'+str(i)] = int(data['public_service_'+str(i)].dropna().mode().values)
for i in range(1,13+1):
    data.loc[data['trust_'+str(i)]<0,'trust_'+str(i)] = int(data['trust_'+str(i)].dropna().mode().values)
```

因为家庭的收入是连续值，使用均值进行缺失值的补全。针对其他特征需要使用日常生活中的真实情况，例如“宗教信息”特征为负数的认为是“不信仰宗教”，并认为“参加宗教活动的频率”为1，即没有参加过宗教活动，主观的进行补全

### 特征交叉

```markdown
添加了如下的特征：第一次结婚年龄、最近结婚年龄、是否再婚、配偶年龄、配偶年龄差、各种收入比（与配偶之间的收入比、十年后预期收入与现在收入之比等等）、收入与住房面积比（其中也包括10年后期望收入等等各种情况）、社会阶级（10年后的社会阶级、14年后的社会阶级等等）、悠闲指数、满意指数、信任指数等等。除此之外，还考虑了对于同一省、市、县进行了归一化。例如同一省市内的收入的平均值等以及一个个体相对于同省、市、县其他人的各个指标的情况。同时也考虑了对于同龄人之间的相互比较，即在同龄人中的收入情况、健康情况等等
```

```python
#第一次结婚年龄 147
data['marital_1stbir'] = data['marital_1st'] - data['birth'] 
#最近结婚年龄 148
data['marital_nowtbir'] = data['marital_now'] - data['birth'] 
#是否再婚 149
data['mar'] = data['marital_nowtbir'] - data['marital_1stbir']
#配偶年龄 150
data['marital_sbir'] = data['marital_now']-data['s_birth']
#配偶年龄差 151
data['age_'] = data['marital_nowtbir'] - data['marital_sbir'] 

#收入比 151+7 =158
data['income/s_income'] = data['income']/(data['s_income']+1) #同居伴侣
data['income+s_income'] = data['income']+(data['s_income']+1)
data['income/family_income'] = data['income']/(data['family_income']+1)
data['all_income/family_income'] = (data['income']+data['s_income'])/(data['family_income']+1)
data['income/inc_exp'] = data['income']/(data['inc_exp']+1)
data['family_income/m'] = data['family_income']/(data['family_m']+0.01)
data['income/m'] = data['income']/(data['family_m']+0.01)

#收入/面积比 158+4=162
data['income/floor_area'] = data['income']/(data['floor_area']+0.01)
data['all_income/floor_area'] = (data['income']+data['s_income'])/(data['floor_area']+0.01)
data['family_income/floor_area'] = data['family_income']/(data['floor_area']+0.01)
data['floor_area/m'] = data['floor_area']/(data['family_m']+0.01)

#class 162+3=165
data['class_10_diff'] = (data['class_10_after'] - data['class'])
data['class_diff'] = data['class'] - data['class_10_before']
data['class_14_diff'] = data['class'] - data['class_14']
#悠闲指数 166
leisure_fea_lis = ['leisure_'+str(i) for i in range(1,13)]
data['leisure_sum'] = data[leisure_fea_lis].sum(axis=1) #skew
#满意指数 167
public_service_fea_lis = ['public_service_'+str(i) for i in range(1,10)]
data['public_service_sum'] = data[public_service_fea_lis].sum(axis=1) #skew

#信任指数 168
trust_fea_lis = ['trust_'+str(i) for i in range(1,14)]
data['trust_sum'] = data[trust_fea_lis].sum(axis=1) #skew

#province mean 168+13=181
data['province_income_mean'] = data.groupby(['province'])['income'].transform('mean').values
data['province_family_income_mean'] = data.groupby(['province'])['family_income'].transform('mean').values
data['province_equity_mean'] = data.groupby(['province'])['equity'].transform('mean').values
data['province_depression_mean'] = data.groupby(['province'])['depression'].transform('mean').values
data['province_floor_area_mean'] = data.groupby(['province'])['floor_area'].transform('mean').values
data['province_health_mean'] = data.groupby(['province'])['health'].transform('mean').values
data['province_class_10_diff_mean'] = data.groupby(['province'])['class_10_diff'].transform('mean').values
data['province_class_mean'] = data.groupby(['province'])['class'].transform('mean').values
data['province_health_problem_mean'] = data.groupby(['province'])['health_problem'].transform('mean').values
data['province_family_status_mean'] = data.groupby(['province'])['family_status'].transform('mean').values
data['province_leisure_sum_mean'] = data.groupby(['province'])['leisure_sum'].transform('mean').values
data['province_public_service_sum_mean'] = data.groupby(['province'])['public_service_sum'].transform('mean').values
data['province_trust_sum_mean'] = data.groupby(['province'])['trust_sum'].transform('mean').values

#city   mean 181+13=194
data['city_income_mean'] = data.groupby(['city'])['income'].transform('mean').values #按照city分组
data['city_family_income_mean'] = data.groupby(['city'])['family_income'].transform('mean').values
data['city_equity_mean'] = data.groupby(['city'])['equity'].transform('mean').values
data['city_depression_mean'] = data.groupby(['city'])['depression'].transform('mean').values
data['city_floor_area_mean'] = data.groupby(['city'])['floor_area'].transform('mean').values
data['city_health_mean'] = data.groupby(['city'])['health'].transform('mean').values
data['city_class_10_diff_mean'] = data.groupby(['city'])['class_10_diff'].transform('mean').values
data['city_class_mean'] = data.groupby(['city'])['class'].transform('mean').values
data['city_health_problem_mean'] = data.groupby(['city'])['health_problem'].transform('mean').values
data['city_family_status_mean'] = data.groupby(['city'])['family_status'].transform('mean').values
data['city_leisure_sum_mean'] = data.groupby(['city'])['leisure_sum'].transform('mean').values
data['city_public_service_sum_mean'] = data.groupby(['city'])['public_service_sum'].transform('mean').values
data['city_trust_sum_mean'] = data.groupby(['city'])['trust_sum'].transform('mean').values

#county  mean 194 + 13 = 207
data['county_income_mean'] = data.groupby(['county'])['income'].transform('mean').values
data['county_family_income_mean'] = data.groupby(['county'])['family_income'].transform('mean').values
data['county_equity_mean'] = data.groupby(['county'])['equity'].transform('mean').values
data['county_depression_mean'] = data.groupby(['county'])['depression'].transform('mean').values
data['county_floor_area_mean'] = data.groupby(['county'])['floor_area'].transform('mean').values
data['county_health_mean'] = data.groupby(['county'])['health'].transform('mean').values
data['county_class_10_diff_mean'] = data.groupby(['county'])['class_10_diff'].transform('mean').values
data['county_class_mean'] = data.groupby(['county'])['class'].transform('mean').values
data['county_health_problem_mean'] = data.groupby(['county'])['health_problem'].transform('mean').values
data['county_family_status_mean'] = data.groupby(['county'])['family_status'].transform('mean').values
data['county_leisure_sum_mean'] = data.groupby(['county'])['leisure_sum'].transform('mean').values
data['county_public_service_sum_mean'] = data.groupby(['county'])['public_service_sum'].transform('mean').values
data['county_trust_sum_mean'] = data.groupby(['county'])['trust_sum'].transform('mean').values

#ratio 相比同省 207 + 13 =220
data['income/province'] = data['income']/(data['province_income_mean'])                                      
data['family_income/province'] = data['family_income']/(data['province_family_income_mean'])   
data['equity/province'] = data['equity']/(data['province_equity_mean'])       
data['depression/province'] = data['depression']/(data['province_depression_mean'])                                                
data['floor_area/province'] = data['floor_area']/(data['province_floor_area_mean'])
data['health/province'] = data['health']/(data['province_health_mean'])
data['class_10_diff/province'] = data['class_10_diff']/(data['province_class_10_diff_mean'])
data['class/province'] = data['class']/(data['province_class_mean'])
data['health_problem/province'] = data['health_problem']/(data['province_health_problem_mean'])
data['family_status/province'] = data['family_status']/(data['province_family_status_mean'])
data['leisure_sum/province'] = data['leisure_sum']/(data['province_leisure_sum_mean'])
data['public_service_sum/province'] = data['public_service_sum']/(data['province_public_service_sum_mean'])
data['trust_sum/province'] = data['trust_sum']/(data['province_trust_sum_mean']+1)

#ratio 相比同市 220 + 13 =233
data['income/city'] = data['income']/(data['city_income_mean'])                                      
data['family_income/city'] = data['family_income']/(data['city_family_income_mean'])   
data['equity/city'] = data['equity']/(data['city_equity_mean'])       
data['depression/city'] = data['depression']/(data['city_depression_mean'])                                                
data['floor_area/city'] = data['floor_area']/(data['city_floor_area_mean'])
data['health/city'] = data['health']/(data['city_health_mean'])
data['class_10_diff/city'] = data['class_10_diff']/(data['city_class_10_diff_mean'])
data['class/city'] = data['class']/(data['city_class_mean'])
data['health_problem/city'] = data['health_problem']/(data['city_health_problem_mean'])
data['family_status/city'] = data['family_status']/(data['city_family_status_mean'])
data['leisure_sum/city'] = data['leisure_sum']/(data['city_leisure_sum_mean'])
data['public_service_sum/city'] = data['public_service_sum']/(data['city_public_service_sum_mean'])
data['trust_sum/city'] = data['trust_sum']/(data['city_trust_sum_mean'])

#ratio 相比同个地区 233 + 13 =246
data['income/county'] = data['income']/(data['county_income_mean'])                                      
data['family_income/county'] = data['family_income']/(data['county_family_income_mean'])   
data['equity/county'] = data['equity']/(data['county_equity_mean'])       
data['depression/county'] = data['depression']/(data['county_depression_mean'])                                                
data['floor_area/county'] = data['floor_area']/(data['county_floor_area_mean'])
data['health/county'] = data['health']/(data['county_health_mean'])
data['class_10_diff/county'] = data['class_10_diff']/(data['county_class_10_diff_mean'])
data['class/county'] = data['class']/(data['county_class_mean'])
data['health_problem/county'] = data['health_problem']/(data['county_health_problem_mean'])
data['family_status/county'] = data['family_status']/(data['county_family_status_mean'])
data['leisure_sum/county'] = data['leisure_sum']/(data['county_leisure_sum_mean'])
data['public_service_sum/county'] = data['public_service_sum']/(data['county_public_service_sum_mean'])
data['trust_sum/county'] = data['trust_sum']/(data['county_trust_sum_mean'])

#age   mean 246+ 13 =259
data['age_income_mean'] = data.groupby(['age'])['income'].transform('mean').values
data['age_family_income_mean'] = data.groupby(['age'])['family_income'].transform('mean').values
data['age_equity_mean'] = data.groupby(['age'])['equity'].transform('mean').values
data['age_depression_mean'] = data.groupby(['age'])['depression'].transform('mean').values
data['age_floor_area_mean'] = data.groupby(['age'])['floor_area'].transform('mean').values
data['age_health_mean'] = data.groupby(['age'])['health'].transform('mean').values
data['age_class_10_diff_mean'] = data.groupby(['age'])['class_10_diff'].transform('mean').values
data['age_class_mean'] = data.groupby(['age'])['class'].transform('mean').values
data['age_health_problem_mean'] = data.groupby(['age'])['health_problem'].transform('mean').values
data['age_family_status_mean'] = data.groupby(['age'])['family_status'].transform('mean').values
data['age_leisure_sum_mean'] = data.groupby(['age'])['leisure_sum'].transform('mean').values
data['age_public_service_sum_mean'] = data.groupby(['age'])['public_service_sum'].transform('mean').values
data['age_trust_sum_mean'] = data.groupby(['age'])['trust_sum'].transform('mean').values

# 和同龄人相比259 + 13 =272
data['income/age'] = data['income']/(data['age_income_mean'])                                      
data['family_income/age'] = data['family_income']/(data['age_family_income_mean'])   
data['equity/age'] = data['equity']/(data['age_equity_mean'])       
data['depression/age'] = data['depression']/(data['age_depression_mean'])                                                
data['floor_area/age'] = data['floor_area']/(data['age_floor_area_mean'])
data['health/age'] = data['health']/(data['age_health_mean'])
data['class_10_diff/age'] = data['class_10_diff']/(data['age_class_10_diff_mean'])
data['class/age'] = data['class']/(data['age_class_mean'])
data['health_problem/age'] = data['health_problem']/(data['age_health_problem_mean'])
data['family_status/age'] = data['family_status']/(data['age_family_status_mean'])
data['leisure_sum/age'] = data['leisure_sum']/(data['age_leisure_sum_mean'])
data['public_service_sum/age'] = data['public_service_sum']/(data['age_public_service_sum_mean'])
data['trust_sum/age'] = data['trust_sum']/(data['age_trust_sum_mean'])
```

```markdown
经过如上的操作后，特征从一开始的131维，扩充为了272维的特征
```

## 3.4 特征工程

```python
print('shape',data.shape)
data.head()
```

```
shape (10956, 272)
```

Out[17]:

|      |   id | survey_type | province | city | county | survey_time | gender | birth | nationality | religion |  ... | depression/age | floor_area/age | health/age | class_10_diff/age | class/age | health_problem/age | family_status/age | leisure_sum/age | public_service_sum/age | trust_sum/age |
| ---: | ---: | ----------: | -------: | ---: | -----: | ----------: | -----: | ----: | ----------: | -------: | ---: | -------------: | -------------: | ---------: | ----------------: | --------: | -----------------: | ----------------: | --------------: | ---------------------: | ------------: |
|    0 |    1 |           1 |       12 |   32 |     59 |        2015 |      1 |  1959 |           1 |        1 |  ... |       1.285211 |       0.410351 |   0.848837 |          0.000000 |  0.683307 |           0.521429 |          0.733668 |        0.724620 |               0.666638 |      0.925941 |
|    1 |    2 |           2 |       18 |   52 |     85 |        2015 |      1 |  1992 |           1 |        1 |  ... |       0.733333 |       0.952824 |   1.179337 |          1.012552 |  1.344444 |           0.891344 |          1.359551 |        1.011792 |               1.130778 |      1.188442 |
|    2 |    3 |           2 |       29 |   83 |    126 |        2015 |      2 |  1967 |           1 |        0 |  ... |       1.343537 |       0.972328 |   1.150485 |          1.190955 |  1.195762 |           1.055679 |          1.190955 |        0.966470 |               1.193204 |      0.803693 |
|    3 |    4 |           2 |       10 |   28 |     51 |        2015 |      2 |  1943 |           1 |        1 |  ... |       1.111663 |       0.642329 |   1.276353 |          4.977778 |  1.199143 |           1.188329 |          1.162630 |        0.899346 |               1.153810 |      1.300950 |
|    4 |    5 |           1 |        7 |   18 |     36 |        2015 |      2 |  1994 |           1 |        1 |  ... |       0.750000 |       0.587284 |   1.177106 |          0.000000 |  0.236957 |           1.116803 |          1.093645 |        1.045313 |               0.728161 |      1.117428 |

5 rows × 272 columns

删去有效样本数很少的特征，例如负值太多的特征或者是缺失值太多的特征，这里一共删除了包括“目前的最高教育程度”在内的9类特征，得到了最终的263维的特征

```python
#272-9=263
#删除数值特别少的和之前用过的特征
del_list=['id','survey_time','edu_other','invest_other','property_other','join_party','province','city','county']
use_feature = [clo for clo in data.columns if clo not in del_list]
data.fillna(0,inplace=True) #还是补0
train_shape = train.shape[0] #一共的数据量，训练集
features = data[use_feature].columns #删除后所有的特征
X_train_263 = data[:train_shape][use_feature].values
y_train = target
X_test_263 = data[train_shape:][use_feature].values
X_train_263.shape #最终一种263个特征
```

这里选择了最重要的49个特征，作为除了以上263维特征外的另外一组特征

```python
imp_fea_49 = ['equity','depression','health','class','family_status','health_problem','class_10_after',
           'equity/province','equity/city','equity/county',
           'depression/province','depression/city','depression/county',
           'health/province','health/city','health/county',
           'class/province','class/city','class/county',
           'family_status/province','family_status/city','family_status/county',
           'family_income/province','family_income/city','family_income/county',
           'floor_area/province','floor_area/city','floor_area/county',
           'leisure_sum/province','leisure_sum/city','leisure_sum/county',
           'public_service_sum/province','public_service_sum/city','public_service_sum/county',
           'trust_sum/province','trust_sum/city','trust_sum/county',
           'income/m','public_service_sum','class_diff','status_3_before','age_income_mean','age_floor_area_mean',
           'weight_jin','height_cm',
           'health/age','depression/age','equity/age','leisure_sum/age'
          ]
train_shape = train.shape[0]
X_train_49 = data[:train_shape][imp_fea_49].values
X_test_49 = data[train_shape:][imp_fea_49].values
X_train_49.shape #最重要的49个特征
```

选择需要进行onehot编码的离散变量进行one-hot编码，再合成为第三类特征，共383维。

```python
cat_fea = ['survey_type','gender','nationality','edu_status','political','hukou','hukou_loc','work_exper','work_status','work_type',
           'work_manage','marital','s_political','s_hukou','s_work_exper','s_work_status','s_work_type','f_political','f_work_14',
           'm_political','m_work_14'] #已经是0、1的值不需要onehot
noc_fea = [clo for clo in use_feature if clo not in cat_fea]

onehot_data = data[cat_fea].values
enc = preprocessing.OneHotEncoder(categories = 'auto')
oh_data=enc.fit_transform(onehot_data).toarray()
oh_data.shape #变为onehot编码格式

X_train_oh = oh_data[:train_shape,:]
X_test_oh = oh_data[train_shape:,:]
X_train_oh.shape #其中的训练集

X_train_383 = np.column_stack([data[:train_shape][noc_fea].values,X_train_oh])#先是noc，再是cat_fea
X_test_383 = np.column_stack([data[train_shape:][noc_fea].values,X_test_oh])
X_train_383.shape
```

```markdown
基于此，构建完成了三种特征工程（训练数据集）

其一是上面提取的最重要的49种特征，其中包括健康程度、社会阶级、在同龄人中的收入情况等等特征

其二是扩充后的263维特征（这里可以认为是初始特征）

其三是使用One-hot编码后的特征，这里要使用One-hot进行编码的原因在于，有部分特征为分离值，例如性别中男女，男为1，女为2，我们想使用One-hot将其变为男为0，女为1，来增强机器学习算法的鲁棒性能；再如民族这个特征，原本是1-56这56个数值，如果直接分类会让分类器的鲁棒性变差，所以使用One-hot编码将其变为6个特征进行非零即一的处理
```

## 3.5 特征建模

首先对于原始的263维的特征，使用lightGBM进行处理，使用5折交叉验证的方法

```python
##### lgb_263 #
#lightGBM决策树
lgb_263_param = {
'num_leaves': 7, 
'min_data_in_leaf': 20, #叶子可能具有的最小记录数
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.003,
"boosting": "gbdt", #用gbdt算法
"feature_fraction": 0.18, #例如 0.18时，意味着在每次迭代中随机选择18％的参数来建树
"bagging_freq": 1,
"bagging_fraction": 0.55, #每次迭代时用的数据比例
"bagging_seed": 14,
"metric": 'mse',
"lambda_l1": 0.1,
"lambda_l2": 0.2, 
"verbosity": -1}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)   #交叉切分：5
oof_lgb_263 = np.zeros(len(X_train_263))
predictions_lgb_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):

    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train_263[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train_263[val_idx], y_train[val_idx])#train:val=4:1

    num_round = 10000
    lgb_263 = lgb.train(lgb_263_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 800)
    oof_lgb_263[val_idx] = lgb_263.predict(X_train_263[val_idx], num_iteration=lgb_263.best_iteration)
    predictions_lgb_263 += lgb_263.predict(X_test_263, num_iteration=lgb_263.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb_263, target)))
```

```markdown
fold n°1
Training until validation scores don't improve for 800 rounds
[500]	training's l2: 0.499759	valid_1's l2: 0.532511
[1000]	training's l2: 0.451528	valid_1's l2: 0.499127
[1500]	training's l2: 0.425443	valid_1's l2: 0.485366
[2000]	training's l2: 0.407389	valid_1's l2: 0.479308
[2500]	training's l2: 0.393001	valid_1's l2: 0.475557
[3000]	training's l2: 0.380766	valid_1's l2: 0.473685
[3500]	training's l2: 0.370009	valid_1's l2: 0.47256
[4000]	training's l2: 0.36022	valid_1's l2: 0.471582
[4500]	training's l2: 0.35124	valid_1's l2: 0.470863
[5000]	training's l2: 0.342828	valid_1's l2: 0.470557
[5500]	training's l2: 0.334901	valid_1's l2: 0.470027
[6000]	training's l2: 0.327379	valid_1's l2: 0.469945
[6500]	training's l2: 0.320136	valid_1's l2: 0.469929
Early stopping, best iteration is:
[6113]	training's l2: 0.325685	valid_1's l2: 0.469866
fold n°2
Training until validation scores don't improve for 800 rounds
[500]	training's l2: 0.504322	valid_1's l2: 0.513628
[1000]	training's l2: 0.454889	valid_1's l2: 0.47926
[1500]	training's l2: 0.428782	valid_1's l2: 0.465975
[2000]	training's l2: 0.410927	valid_1's l2: 0.459213
[2500]	training's l2: 0.397259	valid_1's l2: 0.455058
[3000]	training's l2: 0.385427	valid_1's l2: 0.45243
[3500]	training's l2: 0.374843	valid_1's l2: 0.45074
[4000]	training's l2: 0.365255	valid_1's l2: 0.449343
[4500]	training's l2: 0.356341	valid_1's l2: 0.448434
[5000]	training's l2: 0.348006	valid_1's l2: 0.447479
[5500]	training's l2: 0.339997	valid_1's l2: 0.446742
[6000]	training's l2: 0.332342	valid_1's l2: 0.446239
[6500]	training's l2: 0.325096	valid_1's l2: 0.445999
[7000]	training's l2: 0.318177	valid_1's l2: 0.445795
[7500]	training's l2: 0.31147	valid_1's l2: 0.445203
[8000]	training's l2: 0.305186	valid_1's l2: 0.444816
[8500]	training's l2: 0.299023	valid_1's l2: 0.444776
[9000]	training's l2: 0.293044	valid_1's l2: 0.444549
[9500]	training's l2: 0.287267	valid_1's l2: 0.444302
[10000]	training's l2: 0.281737	valid_1's l2: 0.444043
Did not meet early stopping. Best iteration is:
[10000]	training's l2: 0.281737	valid_1's l2: 0.444043
fold n°3
Training until validation scores don't improve for 800 rounds
[500]	training's l2: 0.503169	valid_1's l2: 0.518027
[1000]	training's l2: 0.455063	valid_1's l2: 0.480538
[1500]	training's l2: 0.429865	valid_1's l2: 0.46407
[2000]	training's l2: 0.412415	valid_1's l2: 0.455411
[2500]	training's l2: 0.39818	valid_1's l2: 0.449859
[3000]	training's l2: 0.386272	valid_1's l2: 0.446564
[3500]	training's l2: 0.375497	valid_1's l2: 0.444636
[4000]	training's l2: 0.365709	valid_1's l2: 0.442973
[4500]	training's l2: 0.356724	valid_1's l2: 0.442256
[5000]	training's l2: 0.348307	valid_1's l2: 0.441686
[5500]	training's l2: 0.34018	valid_1's l2: 0.441066
[6000]	training's l2: 0.332494	valid_1's l2: 0.440792
[6500]	training's l2: 0.325101	valid_1's l2: 0.440477
[7000]	training's l2: 0.318142	valid_1's l2: 0.440624
Early stopping, best iteration is:
[6645]	training's l2: 0.323027	valid_1's l2: 0.440355
fold n°4
Training until validation scores don't improve for 800 rounds
[500]	training's l2: 0.504278	valid_1's l2: 0.512194
[1000]	training's l2: 0.455536	valid_1's l2: 0.477492
[1500]	training's l2: 0.429192	valid_1's l2: 0.465315
[2000]	training's l2: 0.411059	valid_1's l2: 0.459404
[2500]	training's l2: 0.396757	valid_1's l2: 0.45599
[3000]	training's l2: 0.384704	valid_1's l2: 0.453799
[3500]	training's l2: 0.374064	valid_1's l2: 0.452263
[4000]	training's l2: 0.364263	valid_1's l2: 0.451174
[4500]	training's l2: 0.35523	valid_1's l2: 0.450311
[5000]	training's l2: 0.346777	valid_1's l2: 0.4498
[5500]	training's l2: 0.338868	valid_1's l2: 0.449091
[6000]	training's l2: 0.331341	valid_1's l2: 0.448833
[6500]	training's l2: 0.324036	valid_1's l2: 0.448504
[7000]	training's l2: 0.317199	valid_1's l2: 0.448216
[7500]	training's l2: 0.310577	valid_1's l2: 0.448083
[8000]	training's l2: 0.304131	valid_1's l2: 0.448229
Early stopping, best iteration is:
[7370]	training's l2: 0.312295	valid_1's l2: 0.448006
fold n°5
Training until validation scores don't improve for 800 rounds
[500]	training's l2: 0.503075	valid_1's l2: 0.519874
[1000]	training's l2: 0.454635	valid_1's l2: 0.484866
[1500]	training's l2: 0.428716	valid_1's l2: 0.47116
[2000]	training's l2: 0.410711	valid_1's l2: 0.465009
[2500]	training's l2: 0.39625	valid_1's l2: 0.461581
[3000]	training's l2: 0.383981	valid_1's l2: 0.459275
[3500]	training's l2: 0.372984	valid_1's l2: 0.45812
[4000]	training's l2: 0.362999	valid_1's l2: 0.45749
[4500]	training's l2: 0.35375	valid_1's l2: 0.457329
[5000]	training's l2: 0.345103	valid_1's l2: 0.457353
[5500]	training's l2: 0.337024	valid_1's l2: 0.45702
[6000]	training's l2: 0.329495	valid_1's l2: 0.457015
[6500]	training's l2: 0.322073	valid_1's l2: 0.457128
Early stopping, best iteration is:
[5850]	training's l2: 0.331719	valid_1's l2: 0.456903
CV score: 0.45183444
```

使用已经训练完的lightGBM的模型进行特征重要性的判断以及可视化，从结果可以看出，排在重要性第一位的是health/age，就是同龄人中的健康程度，与我们主观的看法基本一致

```python
#---------------特征重要性
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
df = pd.DataFrame(data[use_feature].columns.tolist(), columns=['feature'])
df['importance']=list(lgb_263.feature_importance())
df = df.sort_values(by='importance',ascending=False)
plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="feature", data=df.head(50))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
```

图略

```python
## xgboost - 263维
xgb_263_params = {'eta': 0.02,  #lr
              'max_depth': 6,  
              'min_child_weight':3,#最小叶子节点样本权重和
              'gamma':0, #指定节点分裂所需的最小损失函数下降值。
              'subsample': 0.7,  #控制对于每棵树，随机采样的比例
              'colsample_bytree': 0.3,  #用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
              'lambda':2,
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': -1}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof_xgb_263 = np.zeros(len(X_train_263))
predictions_xgb_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train_263[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train_263[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb_263 = xgb.train(dtrain=trn_data, num_boost_round=3000, evals=watchlist, early_stopping_rounds=600, verbose_eval=500, params=xgb_263_params)
    oof_xgb_263[val_idx] = xgb_263.predict(xgb.DMatrix(X_train_263[val_idx]), ntree_limit=xgb_263.best_ntree_limit)
    predictions_xgb_263 += xgb_263.predict(xgb.DMatrix(X_test_263), ntree_limit=xgb_263.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb_263, target)))
```

```python
## xgboost - 263维
xgb_263_params = {'eta': 0.02,  #lr
              'max_depth': 6,  
              'min_child_weight':3,#最小叶子节点样本权重和
              'gamma':0, #指定节点分裂所需的最小损失函数下降值。
              'subsample': 0.7,  #控制对于每棵树，随机采样的比例
              'colsample_bytree': 0.3,  #用来控制每棵随机采样的列数的占比 (每一列是一个特征)。
              'lambda':2,
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': -1}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof_xgb_263 = np.zeros(len(X_train_263))
predictions_xgb_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train_263[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train_263[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb_263 = xgb.train(dtrain=trn_data, num_boost_round=3000, evals=watchlist, early_stopping_rounds=600, verbose_eval=500, params=xgb_263_params)
    oof_xgb_263[val_idx] = xgb_263.predict(xgb.DMatrix(X_train_263[val_idx]), ntree_limit=xgb_263.best_ntree_limit)
    predictions_xgb_263 += xgb_263.predict(xgb.DMatrix(X_test_263), ntree_limit=xgb_263.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb_263, target)))


#RandomForestRegressor随机森林
folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_rfr_263 = np.zeros(len(X_train_263))
predictions_rfr_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_263[trn_idx]
    tr_y = y_train[trn_idx]
    rfr_263 = rfr(n_estimators=1600,max_depth=9, min_samples_leaf=9, min_weight_fraction_leaf=0.0,
            max_features=0.25,verbose=1,n_jobs=-1) #并行化
    #verbose = 0 为不在标准输出流输出日志信息
    #verbose = 1 为输出进度条记录
    #verbose = 2 为每个epoch输出一行记录
    rfr_263.fit(tr_x,tr_y)
    oof_rfr_263[val_idx] = rfr_263.predict(X_train_263[val_idx])
    
    predictions_rfr_263 += rfr_263.predict(X_test_263) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_rfr_263, target)))


#GradientBoostingRegressor梯度提升决策树
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
oof_gbr_263 = np.zeros(train_shape)
predictions_gbr_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_263[trn_idx]
    tr_y = y_train[trn_idx]
    gbr_263 = gbr(n_estimators=400, learning_rate=0.01,subsample=0.65,max_depth=7, min_samples_leaf=20,
            max_features=0.22,verbose=1)
    gbr_263.fit(tr_x,tr_y)
    oof_gbr_263[val_idx] = gbr_263.predict(X_train_263[val_idx])
    
    predictions_gbr_263 += gbr_263.predict(X_test_263) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_gbr_263, target)))

#ExtraTreesRegressor 极端随机森林回归
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_etr_263 = np.zeros(train_shape)
predictions_etr_263 = np.zeros(len(X_test_263))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_263, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_263[trn_idx]
    tr_y = y_train[trn_idx]
    etr_263 = etr(n_estimators=1000,max_depth=8, min_samples_leaf=12, min_weight_fraction_leaf=0.0,
            max_features=0.4,verbose=1,n_jobs=-1)# max_feature：划分时考虑的最大特征数
    etr_263.fit(tr_x,tr_y)
    oof_etr_263[val_idx] = etr_263.predict(X_train_263[val_idx])
    
    predictions_etr_263 += etr_263.predict(X_test_263) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_etr_263, target)))

train_stack2 = np.vstack([oof_lgb_263,oof_xgb_263,oof_gbr_263,oof_rfr_263,oof_etr_263]).transpose()
# transpose()函数的作用就是调换x,y,z的位置,也就是数组的索引值
test_stack2 = np.vstack([predictions_lgb_263, predictions_xgb_263,predictions_gbr_263,predictions_rfr_263,predictions_etr_263]).transpose()

#交叉验证:5折，重复2次
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack2 = np.zeros(train_stack2.shape[0])
predictions_lr2 = np.zeros(test_stack2.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack2,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack2[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack2[val_idx], target.iloc[val_idx].values
    #Kernel Ridge Regression
    lr2 = kr()
    lr2.fit(trn_data, trn_y)
    
    oof_stack2[val_idx] = lr2.predict(val_data)
    predictions_lr2 += lr2.predict(test_stack2) / 10
    
mean_squared_error(target.values, oof_stack2) 
```

```
fold 0
fold 1
fold 2
fold 3
fold 4
fold 5
fold 6
fold 7
fold 8
fold 9
```

Out[27]:

```
0.44737278925955715
```

对49维特征进行建模

```python
##### lgb_49
lgb_49_param = {
'num_leaves': 9,
'min_data_in_leaf': 23,
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.002,
"boosting": "gbdt",
"feature_fraction": 0.45, 
"bagging_freq": 1,
"bagging_fraction": 0.65, 
"bagging_seed": 15,
"metric": 'mse',
"lambda_l2": 0.2, 
"verbosity": -1} # 一个叶子上数据的最小数量 \ feature_fraction将会在每棵树训练之前选择 45% 的特征。可以用来加速训练，可以用来处理过拟合。 #bagging_fraction不进行重采样的情况下随机选择部分数据。可以用来加速训练，可以用来处理过拟合。
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)   
oof_lgb_49 = np.zeros(len(X_train_49))
predictions_lgb_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train_49[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train_49[val_idx], y_train[val_idx])

    num_round = 12000
    lgb_49 = lgb.train(lgb_49_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
    oof_lgb_49[val_idx] = lgb_49.predict(X_train_49[val_idx], num_iteration=lgb_49.best_iteration)
    predictions_lgb_49 += lgb_49.predict(X_test_49, num_iteration=lgb_49.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb_49, target)))

##### xgb_49
xgb_49_params = {'eta': 0.02, 
              'max_depth': 5, 
              'min_child_weight':3,
              'gamma':0,
              'subsample': 0.7, 
              'colsample_bytree': 0.35, 
              'lambda':2,
              'objective': 'reg:linear', 
              'eval_metric': 'rmse', 
              'silent': True, 
              'nthread': -1}


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_xgb_49 = np.zeros(len(X_train_49))
predictions_xgb_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train_49[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train_49[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb_49 = xgb.train(dtrain=trn_data, num_boost_round=3000, evals=watchlist, early_stopping_rounds=600, verbose_eval=500, params=xgb_49_params)
    oof_xgb_49[val_idx] = xgb_49.predict(xgb.DMatrix(X_train_49[val_idx]), ntree_limit=xgb_49.best_ntree_limit)
    predictions_xgb_49 += xgb_49.predict(xgb.DMatrix(X_test_49), ntree_limit=xgb_49.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb_49, target)))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
oof_gbr_49 = np.zeros(train_shape)
predictions_gbr_49 = np.zeros(len(X_test_49))
#GradientBoostingRegressor梯度提升决策树
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    gbr_49 = gbr(n_estimators=600, learning_rate=0.01,subsample=0.65,max_depth=6, min_samples_leaf=20,
            max_features=0.35,verbose=1)
    gbr_49.fit(tr_x,tr_y)
    oof_gbr_49[val_idx] = gbr_49.predict(X_train_49[val_idx])
    
    predictions_gbr_49 += gbr_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_gbr_49, target)))


## stacking
train_stack3 = np.vstack([oof_lgb_49,oof_xgb_49,oof_gbr_49]).transpose()
test_stack3 = np.vstack([predictions_lgb_49, predictions_xgb_49,predictions_gbr_49]).transpose()
#
folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack3 = np.zeros(train_stack3.shape[0])
predictions_lr3 = np.zeros(test_stack3.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack3,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack3[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack3[val_idx], target.iloc[val_idx].values
        #Kernel Ridge Regression
    lr3 = kr()
    lr3.fit(trn_data, trn_y)
    
    oof_stack3[val_idx] = lr3.predict(val_data)
    predictions_lr3 += lr3.predict(test_stack3) / 10
    
mean_squared_error(target.values, oof_stack3) 
```

对383维特征进行建模

```python
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_kr_383 = np.zeros(train_shape)
predictions_kr_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #Kernel Ridge Regression 岭回归
    kr_383 = kr()
    kr_383.fit(tr_x,tr_y)
    oof_kr_383[val_idx] = kr_383.predict(X_train_383[val_idx])
    
    predictions_kr_383 += kr_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_kr_383, target)))

folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_ridge_383 = np.zeros(train_shape)
predictions_ridge_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #使用岭回归
    ridge_383 = Ridge(alpha=1200)
    ridge_383.fit(tr_x,tr_y)
    oof_ridge_383[val_idx] = ridge_383.predict(X_train_383[val_idx])
    
    predictions_ridge_383 += ridge_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_ridge_383, target)))

folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_en_383 = np.zeros(train_shape)
predictions_en_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #ElasticNet 弹性网络
    en_383 = en(alpha=1.0,l1_ratio=0.06)
    en_383.fit(tr_x,tr_y)
    oof_en_383[val_idx] = en_383.predict(X_train_383[val_idx])
    
    predictions_en_383 += en_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_en_383, target)))


folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_br_383 = np.zeros(train_shape)
predictions_br_383 = np.zeros(len(X_test_383))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_383, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_383[trn_idx]
    tr_y = y_train[trn_idx]
    #BayesianRidge 贝叶斯回归
    br_383 = br()
    br_383.fit(tr_x,tr_y)
    oof_br_383[val_idx] = br_383.predict(X_train_383[val_idx])
    
    predictions_br_383 += br_383.predict(X_test_383) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_br_383, target)))

## stacking
train_stack1 = np.vstack([oof_br_383,oof_kr_383,oof_en_383,oof_ridge_383]).transpose()
test_stack1 = np.vstack([predictions_br_383, predictions_kr_383,predictions_en_383,predictions_ridge_383]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack1 = np.zeros(train_stack1.shape[0])
predictions_lr1 = np.zeros(test_stack1.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack1,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack1[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack1[val_idx], target.iloc[val_idx].values
    # LinearRegression简单的线性回归
    lr1 = lr()
    lr1.fit(trn_data, trn_y)
    
    oof_stack1[val_idx] = lr1.predict(val_data)
    predictions_lr1 += lr1.predict(test_stack1) / 10
    
mean_squared_error(target.values, oof_stack1) 
```

```
fold 0
fold 1
fold 2
fold 3
fold 4
fold 5
fold 6
fold 7
fold 8
fold 9
```

Out[36]:

```
0.4872459309976773
```

基于该49维特征的重要性，再使用其他模型对其进行建模

```python
# KR
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_kr_49 = np.zeros(train_shape)
predictions_kr_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    kr_49 = kr()
    kr_49.fit(tr_x,tr_y)
    oof_kr_49[val_idx] = kr_49.predict(X_train_49[val_idx])
    
    predictions_kr_49 += kr_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_kr_49, target)))

#ridge
folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_ridge_49 = np.zeros(train_shape)
predictions_ridge_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    ridge_49 = Ridge(alpha=6)
    ridge_49.fit(tr_x,tr_y)
    oof_ridge_49[val_idx] = ridge_49.predict(X_train_49[val_idx])
    
    predictions_ridge_49 += ridge_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_ridge_49, target)))

folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_br_49 = np.zeros(train_shape)
predictions_br_49 = np.zeros(len(X_test_49))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    br_49 = br()
    br_49.fit(tr_x,tr_y)
    oof_br_49[val_idx] = br_49.predict(X_train_49[val_idx])
    
    predictions_br_49 += br_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_br_49, target)))

folds = KFold(n_splits=5, shuffle=True, random_state=13)
oof_en_49 = np.zeros(train_shape)
predictions_en_49 = np.zeros(len(X_test_49))
#
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_49, y_train)):
    print("fold n°{}".format(fold_+1))
    tr_x = X_train_49[trn_idx]
    tr_y = y_train[trn_idx]
    en_49 = en(alpha=1.0,l1_ratio=0.05)
    en_49.fit(tr_x,tr_y)
    oof_en_49[val_idx] = en_49.predict(X_train_49[val_idx])
    
    predictions_en_49 += en_49.predict(X_test_49) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_en_49, target)))

## stacking
train_stack4 = np.vstack([oof_br_49,oof_kr_49,oof_en_49,oof_ridge_49]).transpose()
test_stack4 = np.vstack([predictions_br_49, predictions_kr_49,predictions_en_49,predictions_ridge_49]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack4 = np.zeros(train_stack4.shape[0])
predictions_lr4 = np.zeros(test_stack4.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack4,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack4[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack4[val_idx], target.iloc[val_idx].values
    #LinearRegression
    lr4 = lr()
    lr4.fit(trn_data, trn_y)
    
    oof_stack4[val_idx] = lr4.predict(val_data)
    # predictions_lr4 += lr4.predict(test_stack1) / 10
    predictions_lr4 += lr4.predict(test_stack4) / 10
    
mean_squared_error(target.values, oof_stack4) 
```

```
fold 0
fold 1
fold 2
fold 3
fold 4
fold 5
fold 6
fold 7
fold 8
fold 9
```

Out[41]:

```
0.4949155250832276
```

## 3.6 模型融合

对于上述四种集成学习的模型的预测结果进行加权的求和,直接使用LinearRegression简单线性回归的进行集成

```python
train_stack5 = np.vstack([oof_stack1,oof_stack2,oof_stack3,oof_stack4]).transpose()
test_stack5 = np.vstack([predictions_lr1, predictions_lr2,predictions_lr3,predictions_lr4]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)
oof_stack5 = np.zeros(train_stack5.shape[0])
predictions_lr5= np.zeros(test_stack5.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack5,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack5[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack5[val_idx], target.iloc[val_idx].values
    #LinearRegression
    lr5 = lr()
    lr5.fit(trn_data, trn_y)
    
    oof_stack5[val_idx] = lr5.predict(val_data)
    predictions_lr5 += lr5.predict(test_stack5) / 10
    
mean_squared_error(target.values, oof_stack5) 

```

```
fold 0
fold 1
fold 2
fold 3
fold 4
fold 5
fold 6
fold 7
fold 8
fold 9
```

Out[43]:

```
0.4472423816608294
```

## 3.7 结果处理

```python
submit_example.loc[submit_example['happiness']>4.96,'happiness']= 5
submit_example.loc[submit_example['happiness']<=1.04,'happiness']= 1
submit_example.loc[(submit_example['happiness']>1.96)&(submit_example['happiness']<2.04),'happiness']= 2

submit_example.to_csv("submision.csv",index=False)
submit_example.happiness.describe()
```

```
count    2968.000000
mean        3.879920
std         0.463834
min         1.694826
25%         3.665735
50%         3.954880
75%         4.186808
max         5.000000
Name: happiness, dtype: float64
```

模型预测的值与目标值的统计特性近似！