import cv2
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time
from sklearn.ensemble import RandomForestClassifier
import pickle
from Classifier import Classifier

def getFilePath (folder):
    video_files = os.listdir(folder)
    return [a + '/' + b for a, b in zip([folder] * len(video_files), video_files)]


def main(new_model_training = False):
    file_paths = getFilePath('input_data')
    file_labels = [i.split('_')[2] for i in file_paths]
    x_train, x_unused, y_train, y_unused = \
        train_test_split(file_paths, file_labels,
                         test_size=0.5, random_state=0, stratify = file_labels)

    cls = Classifier()

    if new_model_training:
        cls.add_training_data(x_train, y_train)
        rf = RandomForestClassifier(n_estimators=300)  # can be replaced with any other ML models
        cls.train(learner=rf)
        """
        with open ('model.pkl', 'wb') as file:
            pickle.dump(rf, file)
        with open ('scaler.pkl', 'wb') as file:
            pickle.dump(cls.scaler, file)
        """

    else:
        with open ('model.pkl', 'rb') as file:
            rf = pickle.load(file)
        with open ('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        cls.scaler = scaler

    predictions = cls.predict(learner=rf, x_test=x_unused)

    accuracy = ([i[2] for i in predictions.index.str.split('_')] == predictions).sum() / len(predictions)
    print ('accuracy:', accuracy)

    # extract test results
    """
    test_predictions = pd.DataFrame([i[2] for i in pd.Series(predictions.index).str.split('_')], columns=['actual'])
    test_predictions['prediction'] = predictions.reset_index()['predictions']
    test_predictions.groupby(['actual', 'prediction']).size()
    """


if __name__ == "__main__":
    if sys.argv[1:][0] == '--train=True':
        print ('Training = True activated')
        main(new_model_training=True)
    else:
        print('Training = False activated')
        main()
