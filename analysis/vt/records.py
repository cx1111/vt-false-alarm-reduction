"""
Module for loading alarm data and record names
"""
import os

import pandas as pd

base_dir = os.path.abspath('..')
data_dir = os.path.join(base_dir, 'data')

def get_alarms():
    # Table of record names and the ground truth labels of their alarms
    alarms = pd.read_csv(os.path.join(data_dir, 'alarms.csv'),
                             dtype={'recordname':object, 'result':bool})

    # Set the name as the index for the alarm data table
    alarms.set_index('recordname', inplace=True)
    # List of record names
    record_names = list(alarms.index)
    # Record names with true alarms
    record_names_true = list(alarms.loc[alarms['result']==True].index)
    # Record names with false alarms
    record_names_false = list(alarms.loc[alarms['result']==False].index)

    return alarms, record_names, record_names_true, record_names_false
