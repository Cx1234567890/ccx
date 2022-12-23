

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('D:\\Cunchu\\WorkSpace\pythonProject4\\used_car_train_20200313\\used_car_train_20200313.csv', sep=' ')
# 缺失值可视化
missing = df.isnull().sum()/len(df)
missing = missing[missing > 0]
missing.sort_values(inplace=True) #排个序
missing.plot.bar()
df.describe().T
# 分离数值变量与分类变量
Nu_feature = list(df.select_dtypes(exclude=['object']).columns)  # 数值变量
Ca_feature = list(df.select_dtypes(include=['object']).columns)
plt.figure(figsize=(30,25))
i=1
for col in Nu_feature:
    ax=plt.subplot(6,5,i)
    ax=sns.kdeplot(df[col],color='red')
    ax=sns.kdeplot(test[col],color='cyan')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax=ax.legend(['train','test'])
    i+=1
plt.show()
correlation_matrix=df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix,vmax=0.9,linewidths=0.05,cmap="RdGy")
# 众数填充缺失值
df['notRepairedDamage']=df['notRepairedDamage'].replace('-',0.0)
df['fuelType'] = df['fuelType'].fillna(0)
df['gearbox'] = df['gearbox'].fillna(0)
df['bodyType'] = df['bodyType'].fillna(0)
df['model'] = df['model'].fillna(0)
# 截断异常值
df['power'][df['power']>600] = 600
df['power'][df['power']<1] = 1
df['v_13'][df['v_13']>6] = 6
df['v_14'][df['v_14']>4] = 4
# 目标变量进行对数变换服从正态分布
df['price'] = np.log1p(df['price'])
from datetime import datetime
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])
    if month < 1:
        month = 1
    date = datetime(year, month, day)
    return date
df['regDate'] = df['regDate'].apply(date_process)
df['creatDate'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days#二手车使用天数
df['car_age_year'] = round(df['car_age_day'] / 365, 1)#二手车使用年数
num_cols = [0, 2, 3, 6, 8, 10, 12, 14]
for index, value in enumerate(num_cols):
    for j in num_cols[index + 1:]:
        df['new' + str(value) + '*' + str(j)] = df['v_' + str(value)] * df['v_' + str(j)]
        df['new' + str(value) + '+' + str(j)] = df['v_' + str(value)] + df['v_' + str(j)]
        df['new' + str(value) + '-' + str(j)] = df['v_' + str(value)] - df['v_' + str(j)]

num_cols1 = [3, 5, 1, 11]
for index, value in enumerate(num_cols1):
    for j in num_cols1[index + 1:]:
        df['new' + str(value) + '-' + str(j)] = df['v_' + str(value)] - df['v_' + str(j)]

for i in range(15):
    df['new' + str(i) + '*year'] = df['v_' + str(i)] * df['car_age_year']
X = df.drop(columns=['price', 'SaleID', 'seller', 'offerType', 'name', 'creatDate', 'regionCode'])
Y = df['price']

import Meancoder  # 平均数编码

class_list = ['model', 'brand', 'power', 'v_0', 'v_3', 'v_8', 'v_12']
MeanEnocodeFeature = class_list  # 声明需要平均数编码的特征
ME = Meancoder.MeanEncoder(MeanEnocodeFeature, target_type='regression')  # 声明平均数编码的类
X = ME.fit_transform(X, Y)  # 对训练数据集的X和y进行拟合
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 划分训练及测试集
x_train,x_test,y_train,y_test = train_test_split( X, Y,test_size=0.3,random_state=1)
# 模型训练
clf=CatBoostRegressor(
            loss_function="MAE",
            eval_metric= 'MAE',
            task_type="CPU",
            od_type="Iter",   #过拟合检查类型
            random_seed=2022)  # learning_rate、iterations、depth可以自己尝试
# 5折交叉  test是测试集B，已经经过清洗及特征工程，方法与训练集一致
result = []
mean_score = 0
n_folds=5
kf = KFold(n_splits=n_folds ,shuffle=True,random_state=2022)
for train_index, test_index in kf.split(X):
    x_train = X.iloc[train_index]
    y_train = Y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print('验证集MAE:{}'.format(mean_absolute_error(np.expm1(y_test),np.expm1(y_pred))))
    mean_score += mean_absolute_error(np.expm1(y_test),np.expm1(y_pred))/ n_folds
    y_pred_final = clf.predict(test)
    y_pred_test=np.expm1(y_pred_final)
    result.append(y_pred_test)
# 模型评估
print('mean 验证集MAE:{}'.format(mean_score))
cat_pre=sum(result)/n_folds
ret=pd.DataFrame(cat_pre,columns=['price'])
ret.to_csv('/预测.csv')
from lightgbm.sklearn import LGBMRegressor

gbm = LGBMRegressor()  # 参数可以去论坛参考
# 由于模型不支持object类型的处理，所以需要转化
X['notRepairedDamage'] = X['notRepairedDamage'].astype('float64')
test['notRepairedDamage'] = test['notRepairedDamage'].astype('float64')
result1 = []
mean_score1 = 0
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)
for train_index, test_index in kf.split(X):
    x_train = X.iloc[train_index]
    y_train = Y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]
    gbm.fit(x_train, y_train)
    y_pred1 = gbm.predict(x_test)
    print('验证集MAE:{}'.format(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred1))))
    mean_score1 += mean_absolute_error(np.expm1(y_test), np.expm1(y_pred1)) / n_folds
    y_pred_final1 = gbm.predict((test), num_iteration=gbm.best_iteration_)
    y_pred_test1 = np.expm1(y_pred_final1)
    result1.append(y_pred_test1)
# 模型评估
print('mean 验证集MAE:{}'.format(mean_score1))
cat_pre1 = sum(result1) / n_folds

# 加权融合
sub_Weighted = (1 - mean_score1 / (mean_score1 + mean_score)) * cat_pre1 + (
            1 - mean_score / (mean_score1 + mean_score)) * cat_pre