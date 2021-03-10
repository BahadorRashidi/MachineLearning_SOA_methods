#%%
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%% Load the data which is abput blood transfusion
X, y = fetch_openml(data_id=1464, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

#%%
'''
Here we first make a pipeline that combine a normalization and then we apply
a simple logestic regression to train our classifier
'''
pipeline1 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l2', max_iter=200))

pipeline1.fit(X_train, y_train)

#%%
'''
Here we firstly classify the test data set and then see how we can evaluate the 
performanc eof the classifier using the condusion matrix
'''
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

y_pred = pipeline1.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, pipeline1.classes_[1]).plot()

'''
In this tep we will use ROC curve to show the result
'''
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

y_score = pipeline1.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pipeline1.classes_[1])
AUC = roc_auc_score(y_test, y_score)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc= AUC, estimator_name ='logestic regression').plot()

'''
Here we apply percesion and recall curve to see how much both variables changes when we change the threshold
'''
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, y_score, pos_label='1')

prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=pipeline1.classes_[1])
pr_display = PrecisionRecallDisplay(
    precision=prec, recall=recall,
    average_precision=average_precision,
    estimator_name ='logestic regression').plot()

#%%
'''
Here we would like to take advantage of the classification_report module in sklearn
'''

from sklearn.metrics import classification_report

class_matrix = classification_report(y_test, y_pred)
#%%


