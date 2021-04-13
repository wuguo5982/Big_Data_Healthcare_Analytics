import time
import pandas as pd
import numpy as np

def read_csv(filepath):
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    return events, mortality

def event_count_metrics(events, mortality):
    # Implement this function to return the event count metrics.
    # Event count is defined as the number of events recorded for a given patient.

    # avg_dead_event_count = 0.0
    # max_dead_event_count = 0.0
    # min_dead_event_count = 0.0
    # avg_alive_event_count = 0.0
    # max_alive_event_count = 0.0
    # min_alive_event_count = 0.0
    
    total_dead_event = events[events["patient_id"].isin(mortality["patient_id"])][["patient_id", "event_id"]].groupby('patient_id')['event_id'].count()
    avg_dead_event_count = total_dead_event.mean()
    max_dead_event_count = total_dead_event.max()
    min_dead_event_count = total_dead_event.min()
    
    total_alive_event = events[~events["patient_id"].isin(mortality["patient_id"])][["patient_id", "event_id"]].groupby('patient_id')['event_id'].count()
    avg_alive_event_count = total_alive_event.mean()
    max_alive_event_count = total_alive_event.max()
    min_alive_event_count = total_alive_event.min() 

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    # Implement this function to return the encounter count metrics.
    # Encounter count is defined as the count of unique dates on which a given patient visited the ICU.

    # avg_dead_encounter_count = 0.0
    # max_dead_encounter_count = 0.0
    # min_dead_encounter_count = 0.0 
    # avg_alive_encounter_count = 0.0
    # max_alive_encounter_count = 0.0
    # min_alive_encounter_count = 0.0
    
    total_dead_encounter = events[events["patient_id"].isin(mortality["patient_id"])][["patient_id", "timestamp"]].groupby('patient_id').agg({'timestamp':'nunique'})
    total_dead_encounter = total_dead_encounter.reset_index()    
    avg_dead_encounter_count = total_dead_encounter["timestamp"].mean()
    max_dead_encounter_count = total_dead_encounter["timestamp"].max()
    min_dead_encounter_count = total_dead_encounter["timestamp"].min()
    
    total_alive_encounter = events[~events["patient_id"].isin(mortality["patient_id"])][["patient_id", "timestamp"]].groupby('patient_id').agg({'timestamp':'nunique'})
    total_alive_encounter = total_alive_encounter.reset_index()
    avg_alive_encounter_count = total_alive_encounter["timestamp"].mean()
    max_alive_encounter_count = total_alive_encounter["timestamp"].max()
    min_alive_encounter_count = total_alive_encounter["timestamp"].min() 

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    # Implement this function to return the record length metrics.
    # Record length is the duration between the first event and the last event for a given patient.

    # avg_dead_rec_len = 0.0
    # max_dead_rec_len = 0.0
    # min_dead_rec_len = 0.0
    # avg_alive_rec_len = 0.0
    # max_alive_rec_len = 0.0
    # min_alive_rec_len = 0.0
    
    new_events = events.groupby('patient_id').agg({'timestamp':['min', 'max']})
    new_events.columns = ['date_min', 'date_max']
    total_events = new_events.reset_index()
    total_events['length'] = total_events.apply(lambda x: len(pd.date_range(x['date_min'], x['date_max']))-1, axis=1)
    total_dead_record = total_events[total_events["patient_id"].isin(mortality["patient_id"])]['length'].tolist()
    max_dead_rec_len = np.max(total_dead_record)
    min_dead_rec_len = np.min(total_dead_record)
    avg_dead_rec_len = np.mean(total_dead_record)
    
    total_alive_record = total_events[~total_events["patient_id"].isin(mortality["patient_id"])]['length'].tolist()
    max_alive_rec_len = np.max(total_alive_record)
    min_alive_rec_len = np.min(total_alive_record)
    avg_alive_rec_len = np.mean(total_alive_record)

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    train_path = '../data/train/'
    #train_path = '../tests/data/statistics/'
    events, mortality = read_csv(train_path)

    # Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    # Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    # Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
