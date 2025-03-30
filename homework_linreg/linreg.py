import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor
from scipy.stats import skew

train = pd.read_csv('train_hw.csv')
test = pd.read_csv('test_hw.csv')

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)



train["SalePrice"] = np.log1p(train["SalePrice"])

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]

def rmsle_cv(model):
    kf = KFold(n_splits = 5, shuffle=True, random_state=42)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

models = {
    "Linear Regression": make_pipeline(RobustScaler(), LinearRegression()),
    "Lasso Regression": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42)),
    "Elastic Net": make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)),
    "Ridge Regression": Ridge(alpha=3, random_state=42),
    "XGBoost Regressor": XGBRegressor()
}
results = {}

for name, model in models.items():
    score = rmsle_cv(model)
    print(f"\n{name} score: {score.mean():.4f} ({score.std():.4f})\n")

fig, axes = plt.subplots(3, len(models), figsize=(20, 10))
fig.suptitle('Model Performance')

for i, (name, model) in enumerate(models.items()):
    model.fit(train.values, y_train)
    y_train_pred = model.predict(train.values)
    y_test_pred = model.predict(test.values)

    y_train_pred_exp = np.expm1(y_train_pred)
    y_train_exp = np.expm1(y_train)
    y_test_pred_exp = np.expm1(y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train_exp, y_train_pred_exp))
    log_rmse = np.log1p(train_rmse)
    results[name] = train_rmse
    print(f'RMSE ({name}) на тренировочном наборе: {train_rmse:.2f} (log: {log_rmse:.2f})')

    sns.kdeplot(y_train_exp, label='Train True', fill=True, ax=axes[1, i])
    sns.kdeplot(y_train_pred_exp, label='Train Predictions', fill=True, ax=axes[1, i])
    sns.kdeplot(y_test_pred_exp, label='Test Predictions', fill=True, ax=axes[1, i])
    axes[1, i].set_title(f'Density of Predictions ({name})')
    axes[1, i].legend()

    axes[2, i].scatter(y_train, y_train_pred, alpha=0.5)
    axes[2, i].set_xlabel('True Values')
    axes[2, i].set_ylabel('Predictions')
    axes[2, i].set_title(f'True vs Predicted ({name})')

for i, (name, score) in enumerate(results.items()):
    axes[0, i].bar(name, score, color='blue')
    axes[0, i].set_title(f'RMSE: {score:.2f}')
    axes[0, i].set_ylim(0, max(results.values()) * 1.2)

plt.tight_layout()
plt.show()

best_model_name = min(results, key=results.get)
final_model = models[best_model_name].fit(train.values, y_train)
y_test_pred_final = np.expm1(final_model.predict(test.values))

submission = pd.DataFrame({
    'Id': test_ID,
    'SalePrice': y_test_pred_final
})
submission.to_csv('submission.csv', index=False)

print(f"Лучший алгоритм: {best_model_name} с RMSE {results[best_model_name]:.2f}")
print("Файл submission.csv сохранён!")