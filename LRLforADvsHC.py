import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import scikitplot

# Assuming XTrainAvH, YTrainAvH, XValAvH, YValAvH, XTestAvH, YTestAvH are defined

# Testing for multicollinearity
coef3 = np.corrcoef(ADvHC, rowvar=False)
plt.hist(coef3)

# Split data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(XTrainAvH, YTrainAvH, test_size=0.4, random_state=100)
X_test, y_test = XTestAvH, YTestAvH

# Parameter grid for Logistic Regression
grid = {
    'C': np.linspace(1e-10, 1e5, num=100),  # Inverse lambda
    'penalty': ['l2']  # Ridge regularization
}
# paramGrid = ParameterGrid(grid)  # No need to create a ParameterGrid object

# Logistic Regression model
RLRMod = LogisticRegression(tol=1e-10, random_state=100, n_jobs=2, verbose=1)

# Perform GridSearchCV, pass the 'grid' dictionary directly
grid_search = GridSearchCV(estimator=RLRMod, param_grid=grid,  # Use 'grid' instead of 'paramGrid'
                           scoring='roc_auc', n_jobs=2, verbose=1, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
bestModel = grid_search.best_estimator_
bestScore = grid_search.best_score_
allModels = grid_search.cv_results_['params']
allScores = grid_search.cv_results_['mean_test_score']

print(f"Best model parameters: {grid_search.best_params_}")
print(f"Best model ROC AUC score: {bestScore:.4f}")

# Predict probabilities and labels on the test set
RLRTestPred = bestModel.predict_proba(X_test)
RLRTestPred2 = bestModel.predict(X_test)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(y_test, RLRTestPred, title='AD vs HC with Ridge')
plt.show()

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(y_test, RLRTestPred2)
plt.show()

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, RLRTestPred2) * 100
print(f"Logistic Regression Accuracy on Test Set: {accuracy:.4f}%")
