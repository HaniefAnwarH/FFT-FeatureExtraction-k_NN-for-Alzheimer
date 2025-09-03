from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import scikitplot.metrics

# Define train, validation, and test sets
[train_inds, test_inds] = next(model_selection.ShuffleSplit(test_size=0.2, random_state=100).split(MCIvAD, y=YMvA))
XTempTrain = MCIvAD[train_inds, :]

[train2_inds, val_inds] = next(model_selection.ShuffleSplit(test_size=0.4, random_state=100).split(XTempTrain, y=YMvA[train_inds]))
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds

# Create sets of X and Y data using indices for MCIvAD
XTrainMvA = MCIvAD[TrainInds, :]
YTrainMvA = YMvA[TrainInds]
XValMvA = MCIvAD[ValInds, :]
YValMvA = YMvA[ValInds]
XTestMvA = MCIvAD[TestInds, :]
YTestMvA = YMvA[TestInds]

# Define the parameter grid for SVM
grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient for 'rbf'
    'kernel': ['rbf']  # Kernel type
}

# Create SVM model
svm = SVC(probability=True, random_state=100)

# GridSearchCV for parameter tuning
paramGrid = GridSearchCV(estimator=svm, param_grid=grid, scoring='roc_auc', n_jobs=2, cv=5)
paramGrid.fit(XTrainMvA, YTrainMvA)

# Best parameters and model
bestModel = paramGrid.best_estimator_
print("Best parameters:", paramGrid.best_params_)
print("Best score:", paramGrid.best_score_)

# Predictions on Test Set
SVMTestPred = bestModel.predict_proba(XTestMvA)
SVMTestPred2 = bestModel.predict(XTestMvA)

# Plot ROC Curve and Confusion Matrix
plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_roc(YTestMvA, SVMTestPred, title='Support Vector Machine (SVM)')
plt.show()

plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_confusion_matrix(YTestMvA, SVMTestPred2)
plt.show()

# Metrics on test set
test_auc = roc_auc_score(YTestMvA, SVMTestPred[:, 1])
test_acc = accuracy_score(YTestMvA, SVMTestPred2)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
