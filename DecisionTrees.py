#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")
df.head()

duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
df.head()

y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)
X.head()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

print (clf.predict([[5, 1, 3, 0, 0, 0]]))

print (clf.predict([[2, 0, 7, 0, 1, 0]]))

