import os
import time
import warnings
import pathlib

import lightgbm as lgb
from lleaves import lleaves

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns

NB_ITERATIONS = 10000

DATA_FOLDER = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "data")
MODEL_FOLDER = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "model")
DATA_FILE = os.path.join(DATA_FOLDER, 'Breast_cancer_data.csv')

warnings.filterwarnings("ignore")

df = pd.read_csv(DATA_FILE)

print(df.info())

print(df['diagnosis'].value_counts())

X = df[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

y_pred_train = clf.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))

# Cnfusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

cm_matrix = pd.DataFrame(data=cm,
                         columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

clf.booster_.save_model(os.path.join(MODEL_FOLDER, "lgbm.txt"))

llvm_model = lleaves.Model(os.path.join(MODEL_FOLDER, "lgbm.txt"))

llvm_model.compile(cache=os.path.join(MODEL_FOLDER, "lleaves.o"))

start = time.time()
for i in range(0, NB_ITERATIONS):
    y_test_again = clf.predict(X_train)
end = time.time()
print(end - start)

start = time.time()
for i in range(0, NB_ITERATIONS):
    llvm_y_test = llvm_model.predict(X_train)
end = time.time()
print(end - start)

print(llvm_model.model_file)
