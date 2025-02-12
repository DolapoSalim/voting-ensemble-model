import matplotlib as mp
import sklearn as sk
import numpy as np
import pandas as pd

#load the dataset
df = pd.read_csv('c:/Users/dolap/Downloads/diabetes_data.csv')
df.head()

#check for missing values
df.isnull().sum()

#check dataset size
df.shape

#Split the dataset into features and target variable
X = df.drop(columns = ['diabetes'])
y = df['diabetes']

#Split the datsaet into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


#create new a knn model
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)

#fit model to training data
knn_gs.fit(X_train, y_train)

#save best model
knn_best = knn_gs.best_estimator_
#check best n_neigbors value
print(knn_gs.best_params_)

from sklearn.ensemble import RandomForestClassifier

#create a new random forest classifier
rf = RandomForestClassifier()

#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}

#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data
rf_gs.fit(X_train, y_train)