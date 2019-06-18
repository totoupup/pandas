#encoding: utf-8

import pandas as pd
import xgboost as xgb
# 读取数据
train = pd.read_csv(r"D:\03code\python\pandas\train.csv")
print(train)
test = pd.read_csv(r"D:\03code\python\pandas\test.csv")
feature_columns_to_use = ['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I']
train_for_matrix = train[feature_columns_to_use]
test_for_matrix = test[feature_columns_to_use]
train_x = train_for_matrix.values
test_x = test_for_matrix.values
train_y = train['YYYY']
gbm = xgb.XGBRegressor(max_depth=10, n_estimators=3000, learning_rate=0.01)
gbm.fit(train_x, train_y)
predictions = gbm.predict(test_x)
submission = pd.DataFrame({'row_id': test['row_id'],
                            'BH': predictions})
print(submission)
submission.to_csv("submission.csv", index=False)


##用训练集训练得到的模型，
##应用到测试集上后，所有样本都获得了同样的拟合值
##（且该拟合值就是训练集最后一个数据的y），为什么？
