import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBRegressor
from scipy.stats import skew

trainDf = pd.read_csv('train_hw.csv')
testDf = pd.read_csv('test_hw.csv')

trainID = trainDf['Id']
testID = testDf['Id']

trainDf.drop('Id', axis = 1, inplace = True)
testDf.drop('Id', axis = 1, inplace = True)

trainDf = trainDf.drop(trainDf[(trainDf['GrLivArea']>4000)].index)
trainDf = trainDf.drop(trainDf[(trainDf['TotalBsmtSF']>2750)].index)
trainDf = trainDf.drop(trainDf[(trainDf['1stFlrSF']>2750)].index)

trainDf['SalePrice'] = np.log1p(trainDf['SalePrice'])

ntrain = trainDf.shape[0]
ntest = testDf.shape[0]
Ytrain = trainDf['SalePrice'].values
allData = pd.concat((trainDf, testDf)).reset_index(drop=True)
allData.drop(['SalePrice'], axis=1, inplace=True)

allData['PoolQC'] = allData['PoolQC'].fillna('None')
allData['MiscFeature'] = allData['MiscFeature'].fillna('None')
allData['Alley'] = allData['Alley'].fillna('None')
allData['Fence'] = allData['Fence'].fillna('None')
allData['FireplaceQu'] = allData['FireplaceQu'].fillna('None')

allData['LotFrontage'] = allData.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    allData[col] = allData[col].fillna('None')
allData['GarageYrBlt'] = allData['GarageYrBlt'].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2'):
    allData[col] = allData[col].fillna('None')

allData['MasVnrType'] = allData['MasVnrType'].fillna('None')
allData['MasVnrArea'] = allData['MasVnrArea'].fillna(0)

allData['Electrical'] = allData['Electrical'].fillna(allData['Electrical'].mode()[0])


allData['MSSubClass'] = allData['MSSubClass'].apply(str)
allData['OverallCond'] = allData['OverallCond'].astype(str)
allData['YrSold'] = allData['YrSold'].astype(str)
allData['MoSold'] = allData['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for col in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(allData[col].values)) 
    allData[col] = lbl.transform(list(allData[col].values))

allData['TotalSF'] = allData['TotalBsmtSF'] + allData['1stFlrSF'] + allData['2ndFlrSF']

numCols = allData.select_dtypes(exclude=['object']).columns
skewedCols = allData[numCols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewedCols})
skewness = skewness[abs(skewness) > 0.75].dropna()

from scipy.special import boxcox1p
skewedFeatures = skewness.index

for feat in skewedFeatures:
    skewValue = skewness.loc[feat, 'Skew']

    if abs(skewValue) > 2: 
        lam = 0.05  
    elif abs(skewValue) > 0.75:
        lam = 0.15

    allData[feat] = boxcox1p(allData[feat], lam)

allData = pd.get_dummies(allData)
print(allData.shape)

trainDf = allData[:ntrain]
testDf = allData[ntrain:]

def rmsle_cv(model):
    kf = KFold(n_splits = 5, shuffle=True, random_state=42)
    rmse= np.sqrt(-cross_val_score(model, trainDf.values, Ytrain, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

models = {
    #"Linear Regression": make_pipeline(RobustScaler(), LinearRegression()),
    "Lasso Regression": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42)),
    "Elastic Net": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)),
    #"Ridge Regression": make_pipeline(RobustScaler(), Ridge(alpha=0.1, random_state=42)),
    #"XGBoost Regressor": XGBRegressor()
}
results = {}

for name, model in models.items():
    score = rmsle_cv(model)
    print(f"{name} score: {score.mean():.4f} ({score.std():.4f})")
print()

fig, axes = plt.subplots(3, len(models), figsize=(len(models)*4, 10))
fig.suptitle('Model Performance')

for i, (name, model) in enumerate(models.items()):
    model.fit(trainDf.values, Ytrain)
    YtrainPred = model.predict(trainDf.values)
    YtestPred = model.predict(testDf.values)

    YtrainPredExp = np.expm1(YtrainPred)
    YtrainExp = np.expm1(Ytrain)
    YtestPredExp = np.expm1(YtestPred)

    trainRMSE = np.sqrt(mean_squared_error(YtrainExp, YtrainPredExp))
    logRMSE = np.log1p(trainRMSE)
    results[name] = trainRMSE
    print(f'RMSE ({name}) на тренировочном наборе: {trainRMSE:.2f} (log: {logRMSE:.2f})')

    sns.kdeplot(YtrainExp, label='Train True', fill=True, ax=axes[1, i])
    sns.kdeplot(YtrainPredExp, label='Train Predictions', fill=True, ax=axes[1, i])
    sns.kdeplot(YtestPredExp, label='Test Predictions', fill=True, ax=axes[1, i])
    axes[1, i].set_title(f'Density of Predictions ({name})')
    axes[1, i].legend()

    axes[2, i].scatter(Ytrain, YtrainPred, alpha=0.5)
    axes[2, i].set_xlabel('True Values')
    axes[2, i].set_ylabel('Predictions')
    axes[2, i].set_title(f'True vs Predicted ({name})')

for i, (name, score) in enumerate(results.items()):
    axes[0, i].bar(name, score, color='blue')
    axes[0, i].set_title(f'RMSE: {score:.2f}')
    axes[0, i].set_ylim(0, max(results.values()) * 1.2)

plt.tight_layout()
plt.show()

bestModelName = min(results, key=results.get)
finalModel = models[bestModelName].fit(trainDf.values, Ytrain)
YtestPredFinal = np.expm1(finalModel.predict(testDf.values))

submission = pd.DataFrame({
    'Id': testID,
    'SalePrice': YtestPredFinal
})
submission.to_csv('submission.csv', index=False)

print(f"Лучший алгоритм: {bestModelName} с RMSE {results[bestModelName]:.2f}")
print("Файл submission.csv сохранён!")
