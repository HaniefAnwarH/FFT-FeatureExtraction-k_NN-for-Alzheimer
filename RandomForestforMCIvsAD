# RF - MCIvAD

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainMvA,YTrainMvA)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = np.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if
               (abs(x) >= np.mean(FeatImp[NZInds[0]])
               + 4*np.std(FeatImp[NZInds[0]]))]
ThreshVal = np.mean(FeatImp[NZInds[0]]) + 2*np.std(FeatImp[NZInds[0]])
Keep_NZInds = np.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestMvA)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestMvA)

scikitplot.metrics.plot_roc(YTestMvA,Pred2_S2, title = 'MCIvAD RF')
scikitplot.metrics.plot_confusion_matrix(YTestMvA,Pred1_S2)

from scipy import stats

FeatImp_RF_MvA_reshape = np.reshape(FeatImp,[19,16])
FeatImp_RF_mean = np.mean(FeatImp_RF_MvA_reshape, axis=0)
FeatImp_RF_std = np.std(FeatImp_RF_MvA_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values,FeatImp_RF_mean, 'o')
plt.title("RF Feature Importance by Channel")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Mean Feature Importance")
plt.show()

# Calculate accuracy on the test set
accuracy = accuracy_score(YTestMvA, Pred1_S2)
print(f"Random Forest Accuracy on Test Set: {accuracy:.4f}")
