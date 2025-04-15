import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)

trainDf = pd.read_csv('train.csv')
testDf = pd.read_csv('test.csv')

testIds = testDf['id']
trainDf.drop('id', axis=1, inplace=True)
testDf.drop('id', axis=1, inplace=True)

yTrain = trainDf['Depression']
xTrain = trainDf.drop('Depression', axis=1)
xTest = testDf.copy()

xTrain = xTrain.drop(['Name', 'Gender', 'CGPA', 'City', 'Profession'], axis=1)
xTest = xTest.drop(['Name', 'Gender', 'CGPA', 'City', 'Profession'], axis=1)

catCols = xTrain.select_dtypes(include='object').columns.tolist()
numCols = xTrain.select_dtypes(exclude='object').columns.tolist()

numPipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

catPipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numPipeline, numCols),
    ("cat", catPipeline, catCols)
])

xTrainProcessed = preprocessor.fit_transform(xTrain)
xTestProcessed = preprocessor.transform(xTest)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=200, solver='liblinear', penalty='l1', random_state=42),
    'KNN': KNeighborsClassifier(metric='manhattan'),
    'SVM': SVC(random_state=42, C=20000.1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=300)
}


bestAcc = 0
bestModel = None
bestModelName = ''
results = {}

print("Model Accuracy Scores (train set):\n" + "-"*30)
for name, model in models.items():
    model.fit(xTrainProcessed, yTrain)

    yTrainPred = model.predict(xTrainProcessed)
    yTestPred = model.predict(xTestProcessed)

    acc = accuracy_score(yTrain, yTrainPred)
    print(f"{name:20s}: {acc:.4f}")

    if acc > bestAcc:
        bestAcc = acc
        bestModel = model
        bestModelName = name

    results[name] = {
        "model": model,
        "yTrainPred": yTrainPred,
        "yTestPred": yTestPred,
        "yTrainProba": model.predict_proba(xTrainProcessed)[:, 1] if hasattr(model, "predict_proba") else None,
        "yTestProba": model.predict_proba(xTestProcessed)[:, 1] if hasattr(model, "predict_proba") else None
    }

if 'Logistic Regression' in results:
    print("\nClassification Report (Logistic Regression):\n" + "-"*50)
    print(classification_report(yTrain, results['Logistic Regression']['yTrainPred']))

print(f"Best model: {bestModelName}\n")
testPreds = bestModel.predict(xTestProcessed)
submission = pd.DataFrame({
    'id': testIds,
    'Depression': testPreds
})
submission.to_csv('submission.csv', index=False)


figTrain, axesTrain = plt.subplots(3, 5, figsize=(20, 10))
figTrain.suptitle("Train Metrics for All Models", fontsize=16, y=1.05)

for i, (name, result) in enumerate(results.items()):
    model = result["model"]
    yPred = result["yTrainPred"]
    yProba = result["yTrainProba"]

    cm = confusion_matrix(yTrain, yPred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axesTrain[0, i], cmap='Blues', colorbar=False)
    axesTrain[0, i].set_title(f'{name}\nConfusion')

    if yProba is not None:
        fpr, tpr, _ = roc_curve(yTrain, yProba)
        auc_score = auc(fpr, tpr)
        axesTrain[1, i].plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
        axesTrain[1, i].plot([0, 1], [0, 1], 'k--')
        axesTrain[1, i].legend()
    axesTrain[1, i].set_title(f'{name}\nROC Curve')
    axesTrain[1, i].grid(True)

    sns.countplot(x=yPred, hue=yPred, ax=axesTrain[2, i], palette='Blues', legend=False)
    axesTrain[2, i].set_title(f'{name}\nClass Dist')

plt.tight_layout()
plt.show()
