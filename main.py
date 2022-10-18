#! Project Ai Loan prediction
# TODO libraries # pyline: disable=fixmeP
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # splits data to x_train ,y_train,x_test,y_test
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif


# TODO getting data :-

train_df = pd.read_csv('loan_data.csv')
train_df = train_df.drop(columns=['Loan_ID'])
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education',
                       'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']


# TODO data cleaning:-

train_df_clean=train_df["Gender"].fillna(train_df["Gender"].mode()[0])
train_df_clean=train_df["Married"].fillna(train_df["Married"].mode()[0])
train_df_clean=train_df["Dependents"].fillna(train_df["Dependents"].mode()[0])
train_df_clean=train_df["Self_Employed"].fillna(train_df["Self_Employed"].mode()[0])
train_df_clean=train_df["LoanAmount"].fillna(train_df["LoanAmount"].mean())
train_df_clean=train_df["Loan_Amount_Term"].fillna(train_df["Loan_Amount_Term"].mode()[0])
train_df_clean=train_df["Credit_History"].fillna(train_df["Credit_History"].mode()[0])


# filling null values of the train_df
train_df["Gender"].fillna(train_df["Gender"].mode()[0], inplace=True)
train_df["Married"].fillna(train_df["Married"].mode()[0], inplace=True)
train_df["Dependents"].fillna(train_df["Dependents"].mode()[0], inplace=True)
train_df["Self_Employed"].fillna(train_df["Self_Employed"].mode()[0], inplace=True)
train_df["Loan_Amount_Term"].fillna(train_df["Loan_Amount_Term"].mode()[0], inplace=True)
train_df["Credit_History"].fillna(train_df["Credit_History"].mode()[0], inplace=True)

train_df["LoanAmount"].fillna(train_df["LoanAmount"].mean(), inplace=True)


# TODO encoding categorical data to numbers:-

train_df_encoded = pd.get_dummies(train_df, drop_first=True)  # drop_first:drop 1 attribute to make k-1 attribute ,drop_first=True
pd.set_option('display.max_columns', train_df_encoded.shape[0] + 1)

# TODO dividing data to independant(x) and dependant (y)

x = train_df_encoded.drop(columns='Loan_Status_Y')  # all columns but (Loan_Status_Y)
y = train_df_encoded['Loan_Status_Y']

# TODO feature selection
#feature selection is added here
trans = GenericUnivariateSelect(score_func=chi2, mode="k_best", param=8)
x = trans.fit_transform(x, y)

# TODO dividing data to train and test samples
#changed random satte to 250 because it's the optimal number
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=250)

# TODO scalling:-

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# TODO SVM model

# linear :
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Linear SVM Accuracy=", metrics.accuracy_score(y_test, y_pred))

# poly:

svm_degree2 = svm.SVC(kernel='poly', degree=2)
svm_degree2.fit(X_train, y_train)
y_pred = svm_degree2.predict(X_test)

print("poly SVM Accuracy=", metrics.accuracy_score(y_test, y_pred))

# TODO logistic regression:-

model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)
p_prep = model.predict_proba(x)
y_prep = model.predict(x)
score = model.score(x, y)
conf_a = confusion_matrix(y, y_prep)
report = classification_report(y, y_prep)

print('logistic regression')
print('score_:', score, end='\n\n')

# TODO Decision tree:-

clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Decision tree accuracy ", accuracy_score(y_test,y_pred))

# TODO NB

# initializaing the NB
classifer = BernoulliNB()

# training the model
classifer.fit(X_train, y_train)

# testing the model
y_pred = classifer.predict(X_test)

print('Bernoulli NB', accuracy_score(y_pred, y_test))

# TODO gaussiaan:-
classifer1 = GaussianNB()

# training the model
classifer1.fit(X_train, y_train)

# testing the model
y_pred1 = classifer1.predict(X_test)
print('gaussiaan NB', accuracy_score(y_test, y_pred1))

# TODO  knn

classifier2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)
print('knn ', accuracy_score(y_test, y_pred))


