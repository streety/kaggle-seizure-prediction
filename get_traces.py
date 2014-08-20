import numpy as np
import os
import re
from scipy.io import loadmat



directory = "/path/to/files/clips/Volumes/Seagate/seizure_detection/competition_data/clips/"

def get_num_traces(patient):
    patient_directory = os.path.join(directory, patient)
    regex = r'^([^_]+_\d+)_(\w+?)_segment_(\d+)\.mat$'
    files = [i for i in os.listdir(patient_directory) if \
            re.match(regex, i).groups()[0] == patient]
    print(len(files))
    data = loadmat(os.path.join(patient_directory, files[0]))['data']
    return data.shape[0]
    



def get_traces(patient, trace_type, channel=None, latency=False):
    patient_directory = os.path.join(directory, patient)
    regex = r'^([^_]+_\d+)_(\w+?)_segment_(\d+)\.mat$'
    files = [i for i in os.listdir(patient_directory) if \
            re.match(regex, i).groups()[1] == trace_type]
    files.sort()
    output = []
    for f in files:
        data = loadmat(os.path.join(patient_directory, f))
        if type(channel) is int:
            output.append(data['data'][channel,:])
        else:
            for i in range(data['data'].shape[0]):
                output.append(data['data'][i,:])
    return np.array(output)

def get_early_traces(patient, channel=None):
    patient_directory = os.path.join(directory, patient)
    regex = r'^([^_]+_\d+)_(\w+?)_segment_(\d+)\.mat$'
    files = [i for i in os.listdir(patient_directory) if \
            re.match(regex, i).groups()[1] in ['ictal', 'interictal']]
    files.sort()
    output = []
    early = []
    for f in files:
        data = loadmat(os.path.join(patient_directory, f))
        output.append(data['data'][channel,:])
        if 'latency' not in data:
            early.append(0)
        elif int(data['latency']) > 15:
            early.append(0)
        else:
            early.append(1)
    return (np.array(output), np.array(early))


def get_training_traces(patient, channel=None):
    """Get training traces"""
    ictal = get_traces(patient, 'ictal', channel)
    interictal = get_traces(patient, 'interictal', channel)
    combined = np.concatenate([ictal, interictal])
    label = np.concatenate([[1]*ictal.shape[0], [0]*interictal.shape[0]])
    return (combined, label)

def get_testing_traces(patient, channel=None):
    """Get testing traces"""
    patient_directory = os.path.join(directory, patient)
    regex = r'^([^_]+_\d+)_(\w+?)_segment_(\d+)\.mat$'
    files = [i for i in os.listdir(patient_directory) if \
            re.match(regex, i).groups()[1] == 'test']
    files.sort()
    output = []
    for f in files:
        data = loadmat(os.path.join(patient_directory, f))['data']
        if type(channel) is int:
            output.append(data[channel,:])
        else:
            for i in range(data.shape[0]):
                output.append(data[i,:])
    return (np.array(output), files)

