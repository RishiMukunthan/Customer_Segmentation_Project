from fancyimpute import KNN
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv('credit-card-data.csv')
data_original = data.copy()
data.drop('CUST_ID',axis=1,inplace=True)

#Apply KNN imputation algorithm
data = pd.DataFrame(KNN(k = 3).fit_transform(data), columns = data.columns)

X = data.iloc[:,:].values

loaded_scaler = joblib.load("customer_segment_pred_scaler.pkl")
loaded_model = joblib.load("customer_segment_pred_model.pkl")

X = loaded_scaler.transform(X)

y = loaded_model.predict(X)

# Conactenating labels
data_label=pd.concat([data_original,pd.Series(y, name = 'GROUP')],axis=1)

data_label.to_csv('data_predicted.csv', index = False)