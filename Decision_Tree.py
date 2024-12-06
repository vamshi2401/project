import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle


df=pd.read_csv("C:/Users/vamshi raj/WSN-DS.csv")
h=df.head()
d=df.describe()
l=df.info()

x1=df.iloc[:,:-1]
x1
y=df.iloc[:,-1]
y


smote = SMOTE(random_state=42)
X_resampled1, y_resampled1 = smote.fit_resample(x1, y)
print("Class distribution before resampling:")
print(y.value_counts())
print("Class distribution after resampling:")
print(pd.Series(y_resampled1).value_counts())


X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled1, y_resampled1, test_size=0.4, random_state=42)

classifier_over = DecisionTreeClassifier(criterion='gini')
classifier_over.fit(X_train_resampled, y_train_resampled)
y_pred_test = classifier_over.predict(X_test_resampled)

accuracy = accuracy_score(y_test_resampled, y_pred_test)*100
print("Accuracy after oversampling using Gini index:", accuracy)


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
    





y_test_binarized = label_binarize(y_test_resampled, classes=np.unique(y_test_resampled))
n_classes = y_test_binarized.shape[1]
y_test_binarized

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], classifier_over.predict_proba(X_test_resampled)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), classifier_over.predict_proba(X_test_resampled).ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green']) # You may need to add more colors if you have more than 5 classes
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

conf_matrix = confusion_matrix(y_test_resampled, y_pred_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
plt.show()
