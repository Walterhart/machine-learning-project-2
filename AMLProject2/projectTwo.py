import pandas as pd

# read heart disease data
heart = pd.read_csv('heart_cleveland_upload.csv')

heart.head()

# X signifies attributes
# y signifies 
X = heart[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = heart['condition']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=40, max_iter=10000)
lr.fit(X_train, y_train)

# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#scale the dataset using mean and std; do training/testing separately to avoid data leakage problem
standardize = True
if standardize:
    X_train = (X_train - X_train.mean())/X_train.std()
    X_test = (X_test - X.mean())/X.std()

from sklearn import svm
# kernals could be: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’. Try them.
clfsvm = svm.SVC(kernel='rbf')
clfsvm.fit(X_train, y_train)

#### See which data points are critical #####
# get the support vectors
print("clfsvm support vectors: {}".format(clfsvm.support_vectors_))
# get indices of support vectors
print("clfsvm support vector indices: {}".format(clfsvm.support_))
# get number of support vectors for each class
print("clfsvm # of support vectors in each class: {}".format(clfsvm.n_support_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(clfsvm.score(X_train, y_train)))
print("Test set score: {:.2f}".format(clfsvm.score(X_test, y_test)))

heart_prediction = clfsvm.predict(X_test)
compare = pd.DataFrame({'true': y_test, 'predicted': heart_prediction})
print("true vs predicted\n", compare)

# Once you are happy with the model, it can be deployed
# Train the model use all data
clfsvm.fit((X-X.mean())/X.std(), y)
