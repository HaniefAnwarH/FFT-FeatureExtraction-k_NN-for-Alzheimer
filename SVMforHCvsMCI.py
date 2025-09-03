from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import scikitplot.metrics

# Define the parameter grid
grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['linear', 'rbf']  # Kernel type
}

# Create SVM model
svm = SVC(probability=True, random_state=100)

# GridSearchCV for parameter tuning
paramGrid = GridSearchCV(estimator=svm, param_grid=grid, scoring='roc_auc', n_jobs=2, cv=5)
paramGrid.fit(XTrainHvM, YTrainHvM)

# Best parameters and model
bestModel = paramGrid.best_estimator_
print("Best parameters:", paramGrid.best_params_)
print("Best score:", paramGrid.best_score_)

# Predictions on Test Set
SVMTestPred = bestModel.predict_proba(XTestHvM)
SVMTestPred2 = bestModel.predict(XTestHvM)

# Plot ROC Curve and Confusion Matrix
plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_roc(YTestHvM, SVMTestPred, title='Support Vector Machine (SVM)')
plt.show()

plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_confusion_matrix(YTestHvM, SVMTestPred2)
plt.show()

# Metrics on test set
test_auc = roc_auc_score(YTestHvM, SVMTestPred[:, 1])
test_acc = accuracy_score(YTestHvM, SVMTestPred2)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
