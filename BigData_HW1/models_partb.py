import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import utils

RANDOM_STATE = 545510477
def logistic_regression_pred(X_train, Y_train):
	# train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	#use default params for the classifier
    log_model = LogisticRegression(random_state = RANDOM_STATE).fit(X_train, Y_train)
    Y_pred_log = log_model.predict(X_train)    
    return Y_pred_log

def svm_pred(X_train, Y_train):
	# train a SVM classifier using X_train and Y_train. Use this to predict labels of X_train
	# use default params for the classifier
    SVM_model = LinearSVC(random_state = RANDOM_STATE).fit(X_train, Y_train)
    Y_pred_SVM = SVM_model.predict(X_train)    
    return Y_pred_SVM

def decisionTree_pred(X_train, Y_train):
	# train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_train
	# use max_depth as 5
    Tree_Model = DecisionTreeClassifier(random_state = RANDOM_STATE, max_depth = 5).fit(X_train, Y_train)
    Y_pred_Tree = Tree_Model.predict(X_train)
    return Y_pred_Tree

def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
    acc = accuracy_score(Y_pred, Y_true)
    auc = roc_auc_score(Y_pred, Y_true)
    precision = precision_score(Y_pred, Y_true)
    recall = recall_score(Y_pred, Y_true)
    f1score = f1_score(Y_pred, Y_true)
    return acc, auc, precision, recall, f1score

def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	
	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train),Y_train)
	display_metrics("SVM",svm_pred(X_train,Y_train),Y_train)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train),Y_train)
	

if __name__ == "__main__":
	main()
	
