# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
import pickle
from train import preprocessData, CNNModel,LSTMModel, Classification,Model

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='CNN_v1',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
    predictions=[(ecg_names[i], 'N') for i in range(0, len(ecg_names))] 
    if model_name=="CNN_v1":
        if is_binary_classifier:
            pd= preprocessData(Classification.BINARY,train=False)
            pd.readTestData(ecg_leads)
            spectograms,dimension = pd.generateSpectograms(pd.data)
            model = CNNModel(dimension,Classification.BINARY)           
            path = 'models\CNN_Binary_V1'
        else:
            pd= preprocessData(Classification.MULTICLASS,train=False)
            pd.readTestData(ecg_leads)
            spectograms,dimension = pd.generateSpectograms(pd.data)
            model = CNNModel(dimension,Classification.MULTICLASS)           
            path = 'models\CNN_Multiclass_V1'
        model_loaded = model.loadModel(path)
        y_pred = model_loaded.predict(spectograms.reshape((-1,  *dimension)), batch_size=128)
        class_pred = model.predictitionToClass(y_pred)  
        labels_pred = model.classIdxToLabels(class_pred,pd.classes)
        predictions= model.formatPrediction(labels_pred,ecg_names)
    elif model_name == "LSTM_v1":
        scaler= pickle.load(open('scaler/scaler.pkl', 'rb'))
        if is_binary_classifier:
            pd= preprocessData(Classification.BINARY,train=False)
            pd = preprocessData(train=False)
            pd.readTestData(ecg_leads)
            signals = pd.resizeData()
            model = LSTMModel(Classification.BINARY)           
            path = 'models\LSTM_Binary_V1'
        else:
            pd= preprocessData(Classification.MULTICLASS,train=False)
            pd = preprocessData(train=False)
            pd.readTestData(ecg_leads)
            signals = pd.resizeData()
            model = LSTMModel(Classification.MULTICLASS)           
            path = 'models\LSTM_Multiclass_V1'
        model_loaded = model.loadModel(path)
        y_pred = model_loaded.predict(scaler.transform(signals), batch_size=128)
        class_pred = model.predictitionToClass(y_pred)  
        labels_pred = model.classIdxToLabels(class_pred,pd.classes)
        predictions= model.formatPrediction(labels_pred,ecg_names)
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!