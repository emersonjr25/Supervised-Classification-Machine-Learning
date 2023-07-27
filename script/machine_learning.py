""" 
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
SUPERVISIONED REGRESSION MACHINE LEARNING 
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

data_all = pd.read_csv('data/train.csv')

data_all.columns
data_all.head()

variables = ['Pclass', 'Sex', 'Age', 'Fare']

data_all.isna().sum()

X = data_all[variables]

y = data_all['Survived']

#name and ticket maybe
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
model2 = RandomForestClassifier()

model2.fit(X_train, y_train)

predict2 = model2.predict(X_test)

acc_model2 = accuracy_score(y_test, predict2)
conf_model2 = confusion_matrix(y_test, predict2)
class_model2 = classification_report(y_test, predict2)
prec_model2 = precision_score(y_test, predict2)
reca_model2 = recall_score(y_test, predict2)
f1_model2 = f1_score(y_test, predict2)

print('accuracy:', acc_model2, 'precision:', prec_model2, 'recall:', reca_model2, 'f1_score:', f1_model2)

### model 3 ###
model3 = DecisionTreeClassifier(max_depth=5)
model3.fit(X_train, y_train)
predict3 = model3.predict(X_test)

acc_model3 = accuracy_score(y_test, predict3)
conf_model3 = confusion_matrix(y_test, predict3)
class_model3 = classification_report(y_test, predict3)
prec_model3 = precision_score(y_test, predict3)
reca_model3 = recall_score(y_test, predict3)
f1_model3 = f1_score(y_test, predict3)

print('accuracy:', acc_model3, 'precision:', prec_model3, 'recall:', reca_model3, 'f1_score:', f1_model3)

### model 4 ###
model4 = SVC(kernel='linear', C=1.0)
model4.fit(X_train, y_train)
predict4 = model4.predict(X_test)

acc_model4 = accuracy_score(y_test, predict4)
conf_model4 = confusion_matrix(y_test, predict4)
class_model4 = classification_report(y_test, predict4)
prec_model4 = precision_score(y_test, predict4)
reca_model4 = recall_score(y_test, predict4)
f1_model4 = f1_score(y_test, predict4)

print('accuracy:', acc_model4, 'precision:', prec_model4, 'recall:', reca_model4, 'f1_score:', f1_model4)

### model 5 ###
model5 = KNeighborsClassifier(n_neighbors=3)
model5.fit(X_train, y_train)
predict5 = model5.predict(X_test)

acc_model5 = accuracy_score(y_test, predict5)
conf_model5 = confusion_matrix(y_test, predict5)
class_model5 = classification_report(y_test, predict5)
prec_model5 = precision_score(y_test, predict5)
reca_model5 = recall_score(y_test, predict5)
f1_model5 = f1_score(y_test, predict5)

print('accuracy:', acc_model5, 'precision:', prec_model5, 'recall:', reca_model5, 'f1_score:', f1_model5)

### model 6 ###
estimators = np.array([10, 50, 100, 200, 500])
depth = np.array([3, 5, 7, 10, 15])
accuracy = []

for i in range(0, len(estimators)):
    for j in range(0, len(depth)):
        model = GradientBoostingClassifier(n_estimators=estimators[i], max_depth=depth[j])
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        acc_model6 = accuracy_score(y_test, predict)
        accuracy.append(acc_model6)
        print('accuracy:', acc_model6, 'estimators:', estimators[i], 'depth:', depth[j])
        
max(accuracy)
model6 = GradientBoostingClassifier(n_estimators=50, max_depth=10)
model6.fit(X_train, y_train)
predict6 = model6.predict(X_test)

acc_model6 = accuracy_score(y_test, predict6)
conf_model6 = confusion_matrix(y_test, predict6)
class_model6 = classification_report(y_test, predict6)
prec_model6 = precision_score(y_test, predict6)
reca_model6 = recall_score(y_test, predict6)
f1_model6 = f1_score(y_test, predict6)

print('accuracy:', acc_model6, 'precision:', prec_model6, 'recall:', reca_model6, 'f1_score:', f1_model6)

### best model ###
final_model = model6.fit(X, y)
