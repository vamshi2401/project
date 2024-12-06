import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


df=pd.read_csv("C:/Users/vamshi raj/WSN-DS.csv")
print(df.head())
X=df.iloc[:,:-1]
X
y=df.iloc[:,-1]
y


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_encoded, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 10,  
    'eta': 0.3,  
    'objective': 'multi:softmax',  
    'num_class': 5  
}
epochs = 10  

bst = xgb.train(param, dtrain, epochs)

preds = bst.predict(dtest)

accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")

class_accuracy = {}
for cls in set(y_test):
    correct_predictions = sum(1 for pred, actual in zip(preds, y_test) if pred == actual == cls)
    total_instances = sum(1 for label in y_test if label == cls)
    class_accuracy[cls] = (correct_predictions / total_instances)*100

print("Accuracy for each class:")
for cls, accuracy in class_accuracy.items():
    print(f"Class {cls}: {accuracy:.2f}")
