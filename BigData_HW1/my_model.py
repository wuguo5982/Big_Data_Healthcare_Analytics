import utils
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import *


RANDOM_STATE = 545510477
def my_features(event, feature):
    filter_feature = pd.merge(event, feature, on='event_id')
    filter_feature = filter_feature[['patient_id', 'idx', 'value', 'event_id']]
    filter_feature = filter_feature[filter_feature['value'].notnull()]
    idx_DIAG_DRUG = filter_feature[filter_feature['event_id'].str.contains('DIAG') | filter_feature['event_id'].str.contains('DRUG')]
    idx_DIAG_DRUG = idx_DIAG_DRUG[['patient_id', 'idx', 'value']]
    
    sum_idx1 = idx_DIAG_DRUG.groupby(['patient_id','idx']).agg('sum')
    sum_idx1.reset_index(inplace = True)    
    sum_idx1.columns = ['patient_id', 'idx', 'sum']
    max_idx1 = sum_idx1.groupby(['idx']).agg({'sum':'max'})
    max_idx1.reset_index(inplace = True)
    
    result_idx1 = pd.merge(sum_idx1, max_idx1, on='idx') 
    result_idx1['ratio'] = result_idx1['sum_x'] / result_idx1['sum_y']
    normal_idx1 = result_idx1[['patient_id', 'idx', 'ratio']]
    
    idx_LAB = filter_feature[filter_feature['event_id'].str.contains('LAB')]
    idx_LAB = idx_LAB[['patient_id', 'idx', 'value']]
    
    count_idx2 = idx_LAB.groupby(['patient_id','idx']).agg('count')
    count_idx2.reset_index(inplace = True)
    count_idx2.columns = ['patient_id', 'idx', 'count']
    max_idx2 = count_idx2.groupby(['idx']).agg({'count':'max'})
    max_idx2.reset_index(inplace = True)
    result_idx2 = pd.merge(count_idx2, max_idx2, on='idx') 
    result_idx2['ratio'] = result_idx2['count_x'] / result_idx2['count_y']
    normal_idx2 = result_idx2[['patient_id', 'idx', 'ratio']]    
    
    aggregated_events = pd.concat([normal_idx1, normal_idx2]).reset_index(drop = True)
    aggregated_events.columns = ['patient_id', 'feature_id', 'feature_value']
    
    patient_features = {}
    agg = aggregated_events.copy()
    for m in agg['patient_id'].unique():
        patient_features[m] = [(agg['feature_id'][n], agg['feature_value'][n]) for n in agg[agg['patient_id']==m].index]  
    
    deliverable_test = open('../deliverables/test_features.txt','wb')
    for patient_id in sorted(patient_features, reverse=False):
        pairs = ""
        for feature in sorted(patient_features[patient_id], reverse=False):
            pairs += " " + str(int(feature[0])) + ":" + ("%.6f" % feature[1])
        Test = str(int(patient_id)) + pairs + " \n"
        deliverable_test.write(bytes((Test), 'UTF-8'))  


def get_acc_auc_randomisedCV(X, Y, iterNo=5, test_percent=0.2):
    rand_CV = ShuffleSplit(n_splits = iterNo, random_state = RANDOM_STATE, test_size = test_percent)
    model1 = DecisionTreeClassifier(random_state = RANDOM_STATE, max_depth = 5)
    model2 = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=RANDOM_STATE)    
    model3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4), random_state = RANDOM_STATE, n_estimators= 1000) 
    model4 = GaussianNB()
    model5 = LinearDiscriminantAnalysis()
    
    auc1_total = []
    auc2_total = []
    auc3_total = []
    auc4_total = []
    auc4_total = []
    auc5_total = []
    for train_index, test_index in rand_CV.split(X):
        xTrain, xTest = X[train_index], X[test_index]
        yTrain, yTest = Y[train_index], Y[test_index]
        fit1 = model1.fit(xTrain, yTrain)
        fit2 = model2.fit(xTrain, yTrain)
        fit3 = model3.fit(xTrain, yTrain)
        fit4 = model3.fit(xTrain, yTrain)
        fit5 = model3.fit(xTrain, yTrain)                
        
        pred1 = fit1.predict(xTest)
        pred2 = fit2.predict(xTest)
        pred3 = fit3.predict(xTest)
        pred4 = fit4.predict(xTest)
        pred5 = fit5.predict(xTest)
        
        auc1 = roc_auc_score(pred1, yTest)
        auc2 = roc_auc_score(pred2, yTest)
        auc3 = roc_auc_score(pred3, yTest)
        auc4 = roc_auc_score(pred3, yTest)
        auc5 = roc_auc_score(pred3, yTest)
        
        auc1_total.append(auc1)
        auc2_total.append(auc2)
        auc3_total.append(auc3)
        auc4_total.append(auc4)
        auc5_total.append(auc5)        
       
    auc1_mean = np.mean(auc1_total)
    auc2_mean = np.mean(auc2_total)
    auc3_mean = np.mean(auc3_total)
    auc4_mean = np.mean(auc4_total)
    auc5_mean = np.mean(auc5_total)
    return auc1_mean, auc2_mean, auc3_mean, auc4_mean, auc5_mean


def my_classifier_predictions(X_train, Y_train, X_test):
    CART_model = DecisionTreeClassifier(max_depth = 4)
    ensemble_model = AdaBoostClassifier(CART_model, random_state = RANDOM_STATE, n_estimators= 1000).fit(X_train, Y_train)
    Y_pred = ensemble_model.predict_proba(X_test)[:,1]  
    return Y_pred


def main():
    event = pd.read_csv('../data/test/events.csv')
    feature = pd.read_csv('../data/test/event_feature_map.csv')        
    my_features(event, feature)   
   
    X, Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    X_test, _ = utils.get_data_from_svmlight("../deliverables/test_features.txt")
    
    auc1_mean, auc2_mean, auc3_mean, auc4_mean, auc5_mean = get_acc_auc_randomisedCV(X,Y)
    print("AUC: ", auc1_mean, auc2_mean, auc3_mean, auc4_mean, auc5_mean)
    
    Y_pred = my_classifier_predictions(X_train, Y_train, X_test)
    utils.generate_submission("../deliverables/test_features.txt", Y_pred)
 	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()






