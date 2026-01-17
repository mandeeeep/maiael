# Classification

## KNN
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

## Logistic Regression
```python
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression # Import LogisticRegression

df = pd.read_csv('https://raw.githubusercontent.com/balabhadra/datasets/main/Iris.csv')
x = df.iloc[:,1:5]
y = df[['Species']]

x_train,x_test, y_train, y_test, = train_test_split(x,y, test_size = 0.5, random_state=42)

# Change from KNeighborsClassifier to LogisticRegression
logistic_model = LogisticRegression(random_state=42, solver='liblinear') # Initialize Logistic Regression model
logistic_model.fit(x_train, y_train.values.ravel()) # Fit the model, ravel y_train for LogisticRegression

y_pred = logistic_model.predict(x_test)

print("Test:", logistic_model.score(x_test, y_test))
print("Train:", logistic_model.score(x_train, y_train))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic_model.classes_)
disp.plot()

report = metrics.classification_report(y_test, y_pred, target_names=logistic_model.classes_,zero_division=0)
print(report)
#print(cm)
```

## Decision Tree
```python
#import necessary modules for Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import metrics

# Initialize and fit the Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42) # Initialize Decision Tree model
dtc.fit(x_train, y_train)

y_pred_dtc = dtc.predict(x_test)

print("Test Accuracy (Decision Tree):", dtc.score(x_test, y_test))
print("Train Accuracy (Decision Tree):", dtc.score(x_train, y_train))

cm_dtc = confusion_matrix(y_test, y_pred_dtc)
disp_dtc = ConfusionMatrixDisplay(confusion_matrix=cm_dtc, display_labels=dtc.classes_)
disp_dtc.plot()

report_dtc = metrics.classification_report(y_test, y_pred_dtc, target_names=dtc.classes_,zero_division=0)
print(report_dtc)
```

## Random Forest
```python
#import necessary modules for Random Forest
from sklearn.ensemble import RandomForestClassifier # Import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import metrics

# Initialize and fit the Random Forest Classifier
rfc = RandomForestClassifier(random_state=42) # Initialize Random Forest model
rfc.fit(x_train, y_train.values.ravel()) # Fit the model, ravel y_train

y_pred_rfc = rfc.predict(x_test)

print("Test Accuracy (Random Forest):", rfc.score(x_test, y_test))
print("Train Accuracy (Random Forest):", rfc.score(x_train, y_train))

cm_rfc = confusion_matrix(y_test, y_pred_rfc)
disp_rfc = ConfusionMatrixDisplay(confusion_matrix=cm_rfc, display_labels=rfc.classes_)
disp_rfc.plot()

report_rfc = metrics.classification_report(y_test, y_pred_rfc, target_names=rfc.classes_,zero_division=0)
print(report_rfc)
```

### Models Review
```python
import pandas as pd
from sklearn.metrics import classification_report

# Assuming all models (knn, logistic_model, dtc, rfc) and data (x_train, x_test, y_train, y_test) are available from previous cells.

# --- Overall Model Accuracies ---
print("\n--- Overall Model Accuracies ---")
display(accuracy_df)

#Assuming all models and y_test, y_pred, y_pred_dtc, y_pred_rfc are available.

print("--- Confusion Matrix for K-Nearest Neighbors ---")
cm_knn = confusion_matrix(y_test, knn.predict(x_test))
cm_knn_df = pd.DataFrame(cm_knn, index=knn.classes_, columns=knn.classes_)
display(cm_knn_df)

print("\n--- Confusion Matrix for Logistic Regression ---")
cm_logistic = confusion_matrix(y_test, logistic_model.predict(x_test))
cm_logistic_df = pd.DataFrame(cm_logistic, index=logistic_model.classes_, columns=logistic_model.classes_)
display(cm_logistic_df)

print("\n--- Confusion Matrix for Decision Tree ---")
cm_dtc = confusion_matrix(y_test, dtc.predict(x_test))
cm_dtc_df = pd.DataFrame(cm_dtc, index=dtc.classes_, columns=dtc.classes_)
display(cm_dtc_df)

print("\n--- Confusion Matrix for Random Forest ---")
cm_rfc = confusion_matrix(y_test, rfc.predict(x_test))
cm_rfc_df = pd.DataFrame(cm_rfc, index=rfc.classes_, columns=rfc.classes_)
display(cm_rfc_df)
```
