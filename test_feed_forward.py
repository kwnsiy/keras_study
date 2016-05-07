# -*- coding:utf-8 -*-

"""
  iris-dataset分類プログラム
  require: python 2.7, scipy, numpy, sklearn, pandas, keras
"""

import pandas as pd
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

# 次元：入力層4, 中間層10, 出力層3
model = Sequential()
model.add(Dense(output_dim=10, input_dim=4))
model.add(Activation("relu"))
model.add(Dense(output_dim=3))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

iris = load_iris() # iris-flower-dataset
X = iris.data # 素性
X = sp.stats.zscore(X, axis=0) # 正規標準化
Y = iris.target # ラベル
Y = np.array(pd.get_dummies(Y)) # ダミー化
X_test, X_train, Y_test, Y_train = train_test_split(X,Y,test_size=0.5)
print "model lerning(/・ω・)/・・・"
model.fit(X_train, Y_train, nb_epoch=500) 
print "test inputing(/・ω・)/・・・"
res = model.predict(X_test)

# 精度確認
print "out_i",":","t_i"
out_pred = []
out_true = [] 
for out_i, y_i in zip(res, Y_test):
  out_pred.append(np.argmax(out_i))
  out_true.append(np.argmax(y_i))
  if np.argmax(out_i) == np.argmax(y_i): 
    print np.argmax(out_i),":",np.argmax(y_i),"-->","o"
  else:   
    print np.argmax(out_i),":",np.argmax(y_i),"-->","x"

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(out_true, out_pred, target_names = target_names))
