import utils
import pandas as pd
import numpy as np
from datetime import timedelta as subtract_time

def read_csv(filepath):
    # Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    # Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    # Split events into two groups based on whether the patient is alive or deceased
    # Calculate index date for each patient
    dead_date = mortality.copy()
    dead_date['timestamp'] = pd.to_datetime(mortality['timestamp']) + pd.DateOffset(-30)
    dead_date = dead_date[['patient_id', 'timestamp']]
    
    alive = events[~events["patient_id"].isin(mortality["patient_id"])]
    alive_date = alive.groupby('patient_id').max()
    alive_date = alive_date.reset_index()[['patient_id', 'timestamp']]
    alive_date['timestamp'] = pd.to_datetime(alive_date['timestamp'])
    
    indx_date = pd.concat([dead_date, alive_date]).sort_values(['patient_id'])
    indx_date.columns = ["patient_id", "indx_date"]
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    # Join indx_date with events on patient_id
    # Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    window2000 = events.merge(indx_date, how='left', on=['patient_id'])
    filtered_events = window2000[(pd.to_datetime(window2000['timestamp']) >= window2000['indx_date'] - pd.DateOffset(2000)) & (window2000['indx_date'] >= pd.to_datetime(window2000['timestamp']))]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    # Replace event_id's with index available in event_feature_map.csv
    # Remove events with n/a values
    # Aggregate events using sum and count to calculate feature value
    # Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    filter_feature = pd.merge(filtered_events_df, feature_map_df, on='event_id')
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
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events


def create_features(events, mortality, feature_map):
    deliverables_path = '../deliverables/'
    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)
    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    # patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    # mortality : Key - patient_id and value is mortality label
    patient_features = {}
    agg = aggregated_events.copy()
    for m in agg['patient_id'].unique():
        patient_features[m] = [(agg['feature_id'][n], agg['feature_value'][n]) for n in agg[agg['patient_id']==m].index]
    new_events = events.copy()
    new_total = new_events.merge(mortality, how='left', on=['patient_id'])
    new_total['label'] = new_total['label'].fillna(value = 0).astype(int)
    new_label = new_total[['patient_id', 'label']]
    mortality = {}
    mortality = dict(zip(new_label['patient_id'], new_label['label']))
    return patient_features, mortality


def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    #   op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    #   op_deliverable - which saves the features in following format:
    #   patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
    #   patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
    #   Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    for patient_id in sorted(patient_features, reverse=False):
        pairs = ""
        for feature in sorted(patient_features[patient_id], reverse=False):
            pairs += " " + str(int(feature[0])) + ":" + ("%.6f" % feature[1])
        SVM_format = str(mortality[patient_id]) + pairs + ' \n'
    
        deliverable1.write(bytes((SVM_format),'UTF-8')); #Use 'UTF-8'
        deliverable2.write(bytes((str(int(patient_id)) +" " + SVM_format),'UTF-8'));
    
    
def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()