import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv("C:/Users/vamshi raj/WSN-DS.csv")
df.head()
df.describe()

x1=df.iloc[:,:-1]
x1
y=df.iloc[:,-1]
y
print("variables done")


smote = SMOTE(random_state=42)
X_resampled1, y_resampled1 = smote.fit_resample(x1, y)

X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled1, y_resampled1, test_size=0.4, random_state=42)


print("training testing done")
randomclassifier_over = RandomForestClassifier(random_state=42)
randomclassifier_over.fit(X_train_resampled, y_train_resampled)

print("model fitting done")
y_pred_test = randomclassifier_over.predict(X_test_resampled)


accuracy = accuracy_score(y_test_resampled, y_pred_test)*100
print("Accuracy on the test set ", accuracy)

classification_rep=classification_report(y_test_resampled, y_pred_test)
print('Classification Report:\n', classification_rep)

class_accuracy = {}
for cls in set(y_test_resampled):
    correct_predictions = sum(1 for pred, actual in zip(y_pred_test, y_test_resampled) if pred == actual == cls)
    total_instances = sum(1 for label in y_test_resampled if label == cls)
    class_accuracy[cls] = (correct_predictions / total_instances)*100

print("Accuracy for each class:")
for cls, accuracy in class_accuracy.items():
    print(f"Class {cls}: {accuracy:.2f}")
