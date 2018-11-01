#! /usr/bin/env python
# Edited by Thuong Tran, based on Mr. Thanh Nguyen model
# Date: 23 May 2018

import numpy as np
import os

from load_models import ImportGraph


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = BASE_DIR + '/models/runs_cnn'

tf_models = {
    'key_value': {
        'checkpoint_path': os.path.join(BASE_DIR, "1526021526/checkpoints"),
        'vocab_file': os.path.join(BASE_DIR, "1526021526/vocab"),
        'classes': np.array(['value','key'])
    },
    'values': {
        'checkpoint_path': os.path.join(BASE_DIR, "1526263963/checkpoints"),
        'vocab_file': os.path.join(BASE_DIR, "1526263963/vocab"),
        'classes': np.array(['SHIPPER','CONSIGNEE', 'NOTIFY', 'ALSO_NOTIFY', 'POR', 'POL', 'POD', 'DEL','DESCRIPTION', 'VESSEL'])
    },
    'keys': {
        'checkpoint_path': os.path.join(BASE_DIR, "1526440694/checkpoints"),
        'vocab_file': os.path.join(BASE_DIR, "1526440694/vocab"),
        'classes': np.array(['SHIPPER','CONSIGNEE', 'NOTIFY', 'ALSO_NOTIFY', 'POR', 'POL', 'POD', 'DEL','DESCRIPTION', 'VESSEL','Gross Weight','Measurement'])
    },
    'values_mergedata': {
        'checkpoint_path': os.path.join(BASE_DIR, "1526633610/checkpoints"),
        'vocab_file': os.path.join(BASE_DIR, "1526633610/vocab"),
        'classes': np.array(['SHIPPER','CONSIGNEE', 'NOTIFY', 'ALSO_NOTIFY', 'POR', 'POL', 'POD', 'DEL','DESCRIPTION', 'VESSEL'])
    }
}

key_value_model = ImportGraph(tf_models['key_value']['checkpoint_path'],
                                tf_models['key_value']['vocab_file'],
                                tf_models['key_value']['classes'])

value_model = ImportGraph(tf_models['values']['checkpoint_path'],
                                tf_models['values']['vocab_file'],
                                tf_models['values']['classes'])

key_model = ImportGraph(tf_models['keys']['checkpoint_path'],
                                tf_models['keys']['vocab_file'],
                                tf_models['keys']['classes'])

value_merge_model = ImportGraph(tf_models['values_mergedata']['checkpoint_path'],
                                tf_models['values_mergedata']['vocab_file'],
                                tf_models['values_mergedata']['classes'])

def sent_classifer(sentence):
    # If keyword
    # print("-->" + sentence)
    result = key_value_model.run(sentence)
    if result[0][0] == 'key':
        # Answer which key
        result = key_model.run(sentence)
        print('keyword is', result[0][0])
    else: # Value
        # Answer which keyword of that value
        result = value_model.run(sentence)
        print('sentence classification:', result)
    return result

def keyword_detection(sentence):
    result = key_value_model.run(sentence)
    if result[0][0] == 'key':
        result = key_model.run(sentence)
        # print('keyword is', result[0][0])
        return result[0][0], result[0][1]
    else:
        return None, None

def sentence_classifier(sentence):
    result = value_model.run(sentence)
    return result
