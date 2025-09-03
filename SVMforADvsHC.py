# SVM - AD vs HC without imblearn

# Import necessary libraries
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score # Removed plot_roc_curve, plot_confusion_matrix
import scikitplot # scikitplot will be used instead for plotting
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Split data into train and test sets (stratified sampling for class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

# Define the Hyperparameters
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}

# Initialize the SVM Classifier
svm = SVC(probability=True, random_state=100)

# Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the Best Parameters found by GridSearchCV
print("Best Parameters found by GridSearchCV:")
print(grid_search.best_params_)
print()

# Best estimator from GridSearchCV
best_svm = grid_search.best_estimator_

# Predictions on the test set
Pred1_S2 = best_svm.predict(X_test)
Pred2_S2 = best_svm.predict_proba(X_test)

# Plot ROC Curve and Confusion Matrix using scikitplot
scikitplot.metrics.plot_roc(y_test, Pred2_S2) # Use predict_proba output for ROC curve
plt.title('AD vs HC SVM - ROC Curve')
plt.show()

scikitplot.metrics.plot_confusion_matrix(y_test, Pred1_S2) # Use predict output for confusion matrix
plt.title('AD vs HC SVM - Confusion Matrix')
plt.show()

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, Pred1_S2)
print(f"SVM Accuracy on Test Set: {accuracy:.4f}")
