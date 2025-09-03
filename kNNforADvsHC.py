# K-NN - AD vs HC without imblearn

# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # Removed plot_roc_curve, plot_confusion_matrix
import scikitplot # scikitplot will be used instead for plotting
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Assuming X and y are your feature and target data respectively.
# If you haven't defined them, replace them with the appropriate variables.
# For example, if your data is in XTrainAvH and YTrainAvH, use:
X = XTrainAvH
y = YTrainAvH

# Split data into train and test sets (stratified sampling for class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

# Define the Hyperparameters
n_neighbors = range(1, 21)  # Number of neighbors to consider
weights = ['uniform', 'distance']  # Weight function used in prediction

# Initialize the K-NN Classifier
knn = KNeighborsClassifier()

# Grid Search for Hyperparameter Tuning
param_grid = {'n_neighbors': n_neighbors, 'weights': weights}
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the Best Parameters found by GridSearchCV
print("Best Parameters found by GridSearchCV:")
print(grid_search.best_params_)
print()

# Best estimator from GridSearchCV
best_knn = grid_search.best_estimator_

# Predictions on the test set
Pred1_S2 = best_knn.predict(X_test)
Pred2_S2 = best_knn.predict_proba(X_test)

# Plot ROC Curve and Confusion Matrix using scikitplot
scikitplot.metrics.plot_roc(y_test, Pred2_S2) # Use predict_proba output for ROC curve
plt.title('AD vs HC K-NN - ROC Curve')
plt.show()

scikitplot.metrics.plot_confusion_matrix(y_test, Pred1_S2) # Use predict output for confusion matrix
plt.title('AD vs HC K-NN - Confusion Matrix')
plt.show()
# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, Pred1_S2)
print(f"K-NN Accuracy on Test Set: {accuracy:.4f}")

# Note: Feature Importance is not applicable for K-NN, as it does not inherently provide feature importance scores.
