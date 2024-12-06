from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
import pickle


df=pd.read_csv("C:/Users/vamshi raj/WSN-DS.csv")
df.head()
df.describe()

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

pickle.dump(classifier_over, open('model_1.pickle', 'wb'))

