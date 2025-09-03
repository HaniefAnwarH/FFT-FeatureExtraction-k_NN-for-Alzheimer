from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import scikitplot

# Assuming MCIvAD, XTrainMvA, YTrainMvA, XValMvA, YValMvA, XTestMvA, YTestMvA are defined

# Testing for multicollinearity
coef2 = np.corrcoef(MCIvAD, rowvar=False)
plt.hist(coef2)

# Define the parameter grid
ncores = 2
grid = {
    'C': np.linspace(1e-10, 1e5, num=100),  # Inverse lambda
    'penalty': ['l1'], # L1 penalty for Lasso
    'solver': ['liblinear', 'saga'] # Solvers that support L1 penalty
}

# Create Logistic Regression model
RLRMod = LogisticRegression(tol=1e-10, random_state=100, n_jobs=ncores, verbose=1)

# Perform Grid Search using GridSearchCV
grid_search = GridSearchCV(estimator=RLRMod, param_grid=grid,
                           scoring='roc_auc', n_jobs=ncores, verbose=1, cv=5)

grid_search.fit(XTrainMvA, YTrainMvA)

# Get the best model and best score
bestModel = grid_search.best_estimator_
bestScore = grid_search.best_score_

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestMvA)
RLRTestPred2 = bestModel.predict(XTestMvA)

# Convert string labels to binary (assuming ' AD' is the positive class)
YTestMvA_binary = np.where(YTestMvA == ' AD', 1, 0)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(YTestMvA_binary, RLRTestPred[:, 1])
roc_auc = auc(fpr, tpr) # Calculate AUC

# Plot Receiver Operating Characteristic (ROC) Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LR with LASSO')
plt.legend(loc="lower right")
plt.show()
