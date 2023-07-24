""" 
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
SUPERVISIONED REGRESSION MACHINE LEARNING 
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

data_all = pd.read_csv('data/train.csv')

data_all.columns
data_all.head()

variables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X = data_all[variables]

y = data_all['Survived']

### dealing with nas ###

nas = X.isna().sum()

X = X.drop(nas[nas > 0].index[0], axis = 1)

X.corr()

### transforming data ###
label_enconder = LabelEncoder()
X['Sex'] = label_enconder.fit_transform(X['Sex'])

### separating data and machine ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


### model 1 ###
model1 = LogisticRegression()

model1.fit(X_train, y_train)

predict1 = model1.predict(X_test)

acc_model1 = accuracy_score(y_test, predict1)
conf_model1 = confusion_matrix(y_test, predict1)
class_model1 = classification_report(y_test, predict1)
prec_model1 = precision_score(y_test, predict1)
reca_model1 = recall_score(y_test, predict1)
f1_model1 = f1_score(y_test, predict1)

print('accuracy:', acc_model1, 'precision:', prec_model1, 'recall:', reca_model1, 'f1_score:', f1_model1)

### model 2 ###
model2 = RandomForestRegressor()

model2.fit(X_train, y_train)

predict2 = model2.predict(X_test)

def proximate(value):
    if value >= 0.5:
        value = 1
        return value
    else: 
        value = 0
        return value

predict2 = pd.Series(predict2)
predict2 = predict2.apply(proximate)

acc_model2 = accuracy_score(y_test, predict2)
conf_model2 = confusion_matrix(y_test, predict2)
class_model2 = classification_report(y_test, predict2)
prec_model2 = precision_score(y_test, predict2)
reca_model2 = recall_score(y_test, predict2)
f1_model2 = f1_score(y_test, predict2)

print('accuracy:', acc_model2, 'precision:', prec_model2, 'recall:', reca_model2, 'f1_score:', f1_model2)

### best model ###
model2.fit(X, y)
