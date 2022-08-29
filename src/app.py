#############################################
# Decision Tree Classifier Project          #
#############################################

###### Importing Libraries ######
import pandas as pd 
import sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler

######################
# Data Preprocessing #
######################

# Loading the dataset
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')
# Create a copy of the original dataset
df = df_raw.copy()
# Remove the outliers
df=df.drop(df[df['Pregnancies'] > 13.5].index)
df=df.drop(df[df['Glucose'] < 37.125].index)
df=df.drop(df[(df['BloodPressure'] > 107.0) | (df['BloodPressure'] < 35.0)].index)
df=df.drop(df[df['SkinThickness'] > 80.0].index)
df=df.drop(df[df['Insulin'] > 318.125].index)
df=df.drop(df[(df['BMI'] > 50.55) | (df['BMI'] < 13.35)].index)
df=df.drop(df[df['DiabetesPedigreeFunction'] > 1.2149999999999999].index)
df=df.drop(df[df['Age'] > 66.5].index)

#####################
# Model and results #
#####################

# Separating the target variable (y) from the predictors(X)
X_inb = df.drop('Outcome',axis=1)
y_inb = df['Outcome']
# Use random Over-Sampling to add more copies to the minority class
ros =  RandomOverSampler()
X,y = ros.fit_resample(X_inb,y_inb)
# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Decision Tree Classifier estimator
pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier()) 
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
# Results
print("Score with in train dataset:", round(pipeline.score(X_train, y_train), 4))
print("Score with in test dataset:", round(pipeline.score(X_test, y_test), 4)) 
print(classification_report(y_test,y_pred))

############ Hypertune ############
#Using Grid Search to get best hyperparameters
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
params = {
    'max_depth': [2, 3, 5, 10, 20, 50],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'splitter': ['best', 'random'],
    'criterion': ["gini", "entropy"]
}
opt_dt = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=cv, scoring='accuracy',error_score=0)
opt_dt.fit(X_train,y_train)
# Summarize results
print("Best: %f using %s" % (opt_dt.best_score_, opt_dt.best_params_))
# Decision Tree Classifier estimator after hypertune
opt_pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(criterion= 'entropy', max_depth= 20, min_samples_leaf= 5)) 
opt_pipeline.fit(X_train, y_train)
y_pred=opt_pipeline.predict(X_test)
# Results after hypertune
print("Score with in train dataset:", round(opt_pipeline.score(X_train, y_train), 4))
print("Score with in test dataset:", round(opt_pipeline.score(X_test, y_test), 4)) 
print(classification_report(y_test,y_pred))

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/processed/dtc.pkl')

joblib.dump(opt_pipeline, filename)