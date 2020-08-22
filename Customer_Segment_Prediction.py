import pandas as pd
import numpy as np

print("Reading the labelled data from clustering/unsupervised learning")
data = pd.read_csv('Clustering_Output_Python.csv')
data.drop('CUST_ID',axis=1,inplace=True)

print("Storing features and targets in X and y ndarrays respectively")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

print("Performing Train test split")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Scaling X_train and X_test")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Logistic Regression model on train data")
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42, multi_class="multinomial", max_iter=2000)
classifier.fit(X_train,y_train)

print("Predicting for X_test")
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix and Accuracy score")
print(cm)
print(accuracy_score(y_test, y_pred))

print("Retraining on whole data set")
scaler_final = StandardScaler()
X = scaler_final.fit_transform(X)

classifier_final = LogisticRegression(random_state=42, multi_class="multinomial", max_iter=2000)
classifier_final.fit(X,y)

print("Saving final scaler and model as pickle file")
import joblib
joblib.dump(scaler_final,'customer_segment_pred_scaler.pkl')
joblib.dump(classifier_final,'customer_segment_pred_model.pkl')