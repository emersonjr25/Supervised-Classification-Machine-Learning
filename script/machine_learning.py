""" 
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR

"""

import pandas as pd
import sklearn


data_all = pd.read_csv('data/train.csv')

data_all.columns
data_all.head()

variables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']



y = data_all['Survived']


