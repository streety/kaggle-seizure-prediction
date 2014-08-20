import numpy as np
import os
import pickle
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier

import get_traces
import transformers as trans


def build_pipeline(X):
    """Helper function to build the pipeline of feature transformations.
    We do the same thing to each channel so rather than manually copying changes
    for all channels this is automatically generated"""
    channels = X.shape[2]
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('select_%d_pipeline' % i, 
                Pipeline([('select_%d' % i, trans.ChannelExtractor(i)),
                ('channel features', FeatureUnion([
                    ('var', trans.VarTransformer()),
                    ('median', trans.MedianTransformer()),
                    ('fft', trans.FFTTransformer()),
                    ])),
                ])
            ) for i in range(channels)])),
        ('classifier', trans.ModelTransformer(RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=1, 
            random_state=0))),
            ])
    return pipeline


def get_transformed_data(patient, func=get_traces.get_training_traces):
    """Load in all the data"""
    X = []
    channels = get_traces.get_num_traces(patient)
    # Reading in 43 Gb of data, again and again and again . . .
    for i in range(channels):
        x, y = func(patient, i)
        X.append(x)
    return (np.dstack(X), y)




all_labels = []
all_predictions = np.array([])
folders = [i for i in os.listdir(get_traces.directory) if i[0] != '.']
folders.sort()
for folder in folders:
    print('Starting %s' % folder)

    print('getting data')
    X, y = get_transformed_data(folder)
    print(X.shape)
    print('stratifiedshufflesplit')
    cv = StratifiedShuffleSplit(y,
        n_iter=5,
        test_size=0.2,
        random_state=0,)
    print('cross_val_score')
    
    pipeline = build_pipeline(X)
    
    # Putting this in a list is unnecessary
    scores = [
        cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        ]
    print('displaying results')
    for score, label in zip(scores, ['pipeline',]):
        print("AUC:  {:.2%} (+/- {:.2%}), {:}".format(score.mean(), 
            score.std(), label))
    
    clf = pipeline
    print('Fitting full model')
    clf.fit(X, y)
    print('Getting test data')
    testing_data, files = get_transformed_data(folder, 
            get_traces.get_testing_traces)
    print('Generating predictions')
    predictions = clf.predict_proba(testing_data)
    print(predictions.shape, len(files))
    with open('%s_randomforest_predictions.pkl' % folder, 'wb') as f:
        pickle.dump((files, predictions[:,1]), f)

