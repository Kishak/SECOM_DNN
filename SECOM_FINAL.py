#This is an attempt to include trustworthy consideration to the UCI-SECOM data problem
#Being a final project for EENGR 750: Trustworthy machine Learning Fall 2022
# Import required libraries
import sys
assert sys.version_info >= (3, 7)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from tensorflow import keras
from sklearn.metrics import classification_report
# Import data from URL: https://archive.ics.uci.edu/ml/machine-learning-databases/secom/
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
# Included a verification mechanism to show that the prescribed data source is utilized
assert url == "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/","Cross check Data Source"
print ("Data Source Verified!:", url)
# Create a Features and Labels dataframe
X_path = f"{url}secom.data"
Y_path = f"{url}secom_labels.data"
Z_path = f"{url}secom.names" # same as Y_Path
# Assign data and labels to distinct dataframes
X_df = pd.read_csv(X_path, delimiter=" ", header=None)
Y_df = pd.read_csv(Y_path, delimiter=" ", header=None)
Z_df = pd.read_csv(Y_path, delimiter=" ", header=None)
# Obtain general information about the dataframe - a trustworthiness affirming step
print(X_df.describe(include='all'))
print(Y_df.describe(include='all'))
print(Z_df.describe(include='all')) # same as Y_Path
print(X_df.shape, Y_df.shape)
# Take a sample of five rows of each vector group
print(X_df.sample(5))
print(Y_df.sample(5))
# print(Y_df.head())
# print(Y_df.tail())
# print(Z_df.head())
# print(Z_df.tail())
# The data is  but has too many attributes.
# A number of missing values across instances, as indicated by the pandas.describe method
# Strategy 1, eliminate missing values, NaN artifacts, interpolate
# Removing columns where >50 valuse are missing for a sensor feature column  (up to >150 is 422 columns)
print('Dropping columns with more missing values: ', X_df.columns.where(X_df.isna().sum()>50).dropna().values)
X_df = X_df.drop(columns = X_df.columns.where(X_df.isna().sum()>50).dropna())
print('New dataframe shape: ', X_df.shape)

# Filling missing values via interpolation
X_df = X_df.fillna(X_df.mean())
aX_df = X_df
# aX_df = X_df.interpolate()
# aX_df = aX_df.fillna(method='backfill')
assert aX_df.isna().any().any()==False, "Still NaN values in df!!"
# Columns with Zero (0.0) Standard Deviation are eliminated
print('Columns with constant values', (aX_df.loc[:, aX_df.std()==0.0]).columns.values)
dX_df = aX_df.loc[:, aX_df.std()>0.0]
print('\nNew DF shape: ', dX_df.shape)
# Transform data to numerically manipulate
X = X_df.values
y = Y_df.iloc[:,0].values  # To remove unneeded data columns that has date and time associated with labels
# Just to Check the labels
print (y[:15])
# Split train and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Feature Scaling
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.fit_transform(X_test)
# FURTHER DIMENSIONALITY REDUCTION
# Reduce features to 50 only
pca = PCA(n_components = 14)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
# Check reduced inputs shape
print(X_train.shape, X_test.shape)
# Perform Logistic regression
pca_clf = LogisticRegression(random_state = 42)
pca_clf.fit(X_train, y_train)

pca_pred_tr = pca_clf.predict(X_train)
cm_pca_tr = confusion_matrix(y_train, pca_pred_tr)
print(cm_pca_tr)

pca_pred_te = pca_clf.predict(X_test)
cm_pca_te = confusion_matrix(y_test, pca_pred_te)
print(cm_pca_te)

print('Training Accuracy :', pca_clf.score(X_train, y_train))
print('Test Accuracy :', pca_clf.score(X_test, y_test))

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer
#model.add(Dense(14, activation ='relu', input_shape =(14, )))
model.add(Dense(14, activation ='tanh', input_shape =(14,)))

# Add one hidden layer
model.add(Dense(5, activation ='tanh'))

# Add an output layer
model.add(Dense(1, activation ='tanh'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors
model.get_weights()
model.compile(loss ='binary_crossentropy', optimizer ='sgd', metrics =['accuracy'])
#model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics =['accuracy'])

# Training Model
model.fit(X_train, y_train, epochs = 5,batch_size = 100, verbose = 1)

# Predicting the Value
y_pred = model.predict(X_test)
print(y_pred)

