import models_partc
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from numpy import mean
from sklearn.metrics import *
import numpy as np
import pandas as pd
import utils

RANDOM_STATE = 545510477
def get_acc_auc_kfold(X,Y,k=5):
	# First get the train indices and test indices for each iteration
	# Then train the classifier accordingly
	# Report the mean accuracy and mean auc of all the folds
    kf = KFold(n_splits = k, shuffle = False)
    logModel = LogisticRegression(random_state = RANDOM_STATE)
    
    acc_total_KF = []
    auc_total_KF = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        log_model_KF = logModel.fit(X_train, Y_train)        
        
        Y_pred_KF = log_model_KF.predict(X_test)
        acc_KF = accuracy_score(Y_pred_KF, Y_test)
        new_acc_KF = acc_total_KF.append(acc_KF)
        
        auc_KF = roc_auc_score(Y_pred_KF, Y_test)
        new_auc_KF = auc_total_KF.append(auc_KF)
        
    acc_mean_KF = np.mean(acc_total_KF)
    auc_mean_KF = np.mean(auc_total_KF)
    return acc_mean_KF, auc_mean_KF


def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	# First get the train indices and test indices for each iteration
	# Then train the classifier accordingly
	# Report the mean accuracy and mean auc of all the iterations
    
    RS = ShuffleSplit(n_splits = iterNo, random_state = RANDOM_STATE, test_size = test_percent)
    logModel = LogisticRegression(random_state = RANDOM_STATE)
    
    acc_total_RS = []
    auc_total_RS = []
    for train_index, test_index in RS.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        log_model_RS = logModel.fit(X_train, Y_train)        
        
        Y_pred_RS = log_model_RS.predict(X_test)
        acc_RS = accuracy_score(Y_pred_RS, Y_test)
        new_acc_RS = acc_total_RS.append(acc_RS)
        
        auc_RS = roc_auc_score(Y_pred_RS, Y_test)
        new_auc_RS = auc_total_RS.append(auc_RS)

    acc_mean_RS = np.mean(acc_total_RS)
    auc_mean_RS = np.mean(auc_total_RS)
    return acc_mean_RS, auc_mean_RS


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

