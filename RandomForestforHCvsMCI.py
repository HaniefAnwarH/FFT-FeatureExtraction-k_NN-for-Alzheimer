from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import scikitplot.metrics

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

# Random Forest Classifier
RFC = RandomForestClassifier(criterion='gini', random_state=100)

# RandomizedSearchCV for parameter tuning
RFC_RandomSearch = RandomizedSearchCV(estimator=RFC, param_distributions=random_grid,
                                      n_iter=100, cv=3, verbose=2, random_state=10, n_jobs=2)
RFC_RandomSearch.fit(XTrainHvM, YTrainHvM)

# Best parameters and model
print("Best parameters:", RFC_RandomSearch.best_params_)
print("Best score:", RFC_RandomSearch.best_score_)

# Feature importance
feat_imp = RFC_RandomSearch.best_estimator_.feature_importances_

# Test predictions
Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestHvM)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestHvM)

# Plot ROC Curve and Confusion Matrix
plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_roc(YTestHvM, Pred2_S2, title='HCvMC RF')
plt.show()

plt.figure(figsize=(8, 6))
scikitplot.metrics.plot_confusion_matrix(YTestHvM, Pred1_S2)
plt.show()

# Metrics on test set
test_auc = roc_auc_score(YTestHvM, RLRTestPred[:, 1])
test_acc = accuracy_score(YTestHvM, RLRTestPred2)
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
