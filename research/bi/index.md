```python
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/balabhadra/datasets/main/Iris.csv')
x = df.iloc[:,1:5]
y = df[['Species']]

x_train,x_test, y_train, y_test, = train_test_split(x,y, test_size = 0.5, random_state=42)

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print("Test:", knn.score(x_test, y_test))
print("Train:", knn.score(x_train, y_train))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

report = metrics.classification_report(y_test, y_pred, target_names=knn.classes_,zero_division=0)
print(report)
#print(cm)
```
