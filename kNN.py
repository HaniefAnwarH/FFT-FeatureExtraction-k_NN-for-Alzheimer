from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import scikitplot.metrics

# Define the parameter grid
grid = {
    'n_neighbors': range(1, 21),  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan']  # Distance metric used
}

# Create KNN model
knn = KNeighborsClassifier()

# GridSearchCV for parameter tuning
paramGrid = GridSearchCV(estimator=knn, param_grid=grid, scoring='roc_auc', n_jobs=2, cv=5)
paramGrid.fit(XTrainHvM, YTrainHvM)

# Best parameters and model
bestModel = paramGrid.best_estimator_
print("Best parameters:", paramGrid.best_params_)
print("Best score:", paramGrid.best_score_)

# Predictions on Test Set
KNNTestPred = bestModel.predict_proba(XTestHvM)
KNNTestPred2 = bestModel.predict(XTestHvM)

# Plot ROC Curve and Confusion Matrix
plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_roc(YTestHvM, KNNTestPred, title='K-Nearest Neighbors (KNN)')
plt.show()

plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_confusion_matrix(YTestHvM, KNNTestPred2)
plt.show()
