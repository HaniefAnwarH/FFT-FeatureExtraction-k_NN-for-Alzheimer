# RF - AD vs HC without imblearn

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score # Removed plot_roc_curve, plot_confusion_matrix
import scikitplot # scikitplot will be used instead for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

# Split data into train and test sets (stratified sampling for class balance)
X_train, X_test, y_train, y_test = train_test_split(XTrainAvH, YTrainAvH, test_size=0.2, random_state=100, stratify=YTrainAvH)
# Use XTrainAvH and YTrainAvH since it appears you are building a model for AD vs HC

# Declare the Classifier
RFC = RandomForestClassifier(criterion='gini')  # can use 'entropy' instead

# RandomizedSearchCV for hyperparameter tuning
RFC_RandomSearch = RandomizedSearchCV(estimator=RFC,
                                      param_distributions=random_grid,
                                      n_iter=100, cv=3, verbose=2,
                                      random_state=10, n_jobs=2)
RFC_RandomSearch.fit(X_train, y_train)

# Look at the Tuned "Best" Parameters
print("Best Parameters found by RandomizedSearchCV:")
print(RFC_RandomSearch.best_params_)
print()

# Best estimator from RandomizedSearchCV
best_RFC = RFC_RandomSearch.best_estimator_

# Predictions on the test set
Pred1_S2 = best_RFC.predict(X_test)
Pred2_S2 = best_RFC.predict_proba(X_test)

# Plot ROC Curve and Confusion Matrix using scikitplot
scikitplot.metrics.plot_roc(y_test, Pred2_S2) # Use predict_proba output for ROC curve
plt.title('AD vs HC RF - ROC Curve')
plt.show()

scikitplot.metrics.plot_confusion_matrix(y_test, Pred1_S2) # Use predict output for confusion matrix
plt.title('AD vs HC RF - Confusion Matrix')
plt.show()

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, Pred1_S2)
print(f"Random Forest Accuracy on Test Set: {accuracy:.4f}")

# Feature Importance
FeatImp = best_RFC.feature_importances_

# Visualization of Feature Importance (if applicable to your dataset)
# Example of plotting feature importances by channel frequency
# Adjust this visualization according to your specific dataset
FeatImp_RF_MvA_reshape = np.reshape(FeatImp, [19, 16])
FeatImp_RF_mean = np.mean(FeatImp_RF_MvA_reshape, axis=0)
FeatImp_RF_std = np.std(FeatImp_RF_MvA_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0, 30, 16)
plt.errorbar(Freq_values, FeatImp_RF_mean, yerr=FeatImp_RF_std, fmt='o')
plt.title("RF Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.show()
