from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
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

# Define the parameter grid for K-NN
grid = {
    'n_neighbors': range(1, 21),  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan']  # Distance metric used
}

# Create KNN model
knn = KNeighborsClassifier()

# GridSearchCV for parameter tuning
paramGrid = GridSearchCV(estimator=knn, param_grid=grid, scoring='roc_auc', n_jobs=2, cv=5)
paramGrid.fit(XTrainMvA, YTrainMvA)

# Best parameters and model
bestModel = paramGrid.best_estimator_
print("Best parameters:", paramGrid.best_params_)
print("Best score:", paramGrid.best_score_)

# Predictions on Test Set
KNNTestPred = bestModel.predict_proba(XTestMvA)
KNNTestPred2 = bestModel.predict(XTestMvA)

# Plot ROC Curve and Confusion Matrix
plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_roc(YTestMvA, KNNTestPred, title='K-Nearest Neighbors (KNN)')
plt.show()

plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_confusion_matrix(YTestMvA, KNNTestPred2)
plt.show()

# Metrics on test set
test_auc = roc_auc_score(YTestMvA, KNNTestPred[:, 1])
test_acc = accuracy_score(YTestMvA, KNNTestPred2)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
