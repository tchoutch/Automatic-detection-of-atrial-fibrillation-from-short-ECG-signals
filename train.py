# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import csv
import argparse
from xmlrpc.client import boolean
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from ecgdetectors import Detectors

import tensorflow as tf
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM,Bidirectional, GRU, Flatten #could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers,regularizers,layers
from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
import imblearn

from pandas import read_csv
from random import seed
from random import random
from random import randint
from enum import Enum
from datetime import datetime
from scipy import signal
import scipy

from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import  RandomUnderSampler
from imblearn.pipeline import Pipeline

#from google.colab import drive
#drive.mount('/content/drive')
#!cp "/content/drive/My Drive/Wettbewerb_KI/training.zip" .
#!unzip training.zip

#!nvidia-smi

class Strategy(Enum):
  OVERSAMPLE = 0
  OVERUNDERSAMPLE = 1
  OVERUNDERSAMPLEWITHNEIGHBORS = 2

class Classification(Enum):
  BINARY = 0
  MULTICLASS = 1

class ModelType(Enum):
  LSTM = 0
  CNN = 1

class preprocessData:
  def __init__(self,classification_type= Classification.MULTICLASS,train=True,seed = 1):
    self.classes = {'N': 0 , 'A' : 1 , 'O' : 2 , '~' : 3}
    if classification_type== Classification.BINARY:
      self.classes = {'N': 0 , 'A' : 1}
    self.train=train
    if self.train:
      self.data = self.__readData()
      self.size = self.data.shape[0]
      self.signals_lengths = self.__getSignalsLengths()
      self.seed=seed

  def __readData(self):
    all_data  = []
    with open('training/REFERENCE.csv') as csv_file:     # Einlesen der Liste mit Dateinamen und Zuordnung
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
          data = sio.loadmat('training/'+row[0]+'.mat')   # Import der EKG-Dateien
          ecg_lead = data['val'][0]
          if row[1] in self.classes :
            all_data.append((ecg_lead,self.classes[row[1]]))
          line_count = line_count + 1
          if (line_count % 1000)==0:
            print(str(line_count) + "\t Dateien wurden verarbeitet.")
          self.data = np.array(all_data)
    return np.array(all_data)

  def readTestData(self,data):
    self.data = np.array(data)

  def __getSignalsLengths(self):
    return [len(a[0]) for a in self.data]
    
  def plotLengthDistribution(self):
    plt.hist(self.signals_lengths, bins=np.arange(2000, max(self.signals_lengths)+1,1000))
    plt.show()

  def getCounterLengths(self):
    x=Counter(self.signals_lengths)
    return sorted(x.items(), key=lambda i: i[1], reverse=True)
  
  def plotSample(self,class_name):
    n = randint(0, self.size-1)
    sample = self.data[n]
    assert class_name in self.classes
    while (sample[1] != self.classes[class_name]):
      n = randint(0, self.size-1)
      sample = self.data[n]
    plt.plot(sample[0])

  def getNumberSamplesEachClass(self,data= None):
    if data is None:
      data=self.data
    d={}
    if data.ndim == 1:
      singals_types = data
    elif data.shape[1] == 1:
      singals_types = data
    else:
      singals_types = data[:,1]
    for k,v in self.classes.items():
      d[k]=np.count_nonzero(singals_types == v)
    return d
  
  def filterData(self,data=None,signal_length = 9000):
    data = self.data if data is None else data
    return np.array([a for a in[(a[0][:signal_length],a[1]) for a in data] if len(a[0])==signal_length])

  def __getXY(self,data=None):
    if data is None:
      data=self.data
    return np.stack(data[:,0].tolist(), axis=0).astype('float') ,data[:,1].reshape(-1,1).astype('float')
  
  def balanceData(self,data = None, strategy = Strategy.OVERSAMPLE,classification_type=Classification.MULTICLASS, over_sampling_factor = 0.1, under_sampling_factor = 0.5,k_neighbors = 5):
    if not isinstance(data,tuple):
       X,y = self.__getXY(data)
    else:
      X,y = data
    if strategy == Strategy.OVERSAMPLE:
      oversample = SMOTE()
      X, y = oversample.fit_resample(X, y)
    elif strategy == Strategy.OVERUNDERSAMPLE:
      if classification_type == Classification.MULTICLASS:
        over_sampling_factor,under_sampling_factor=self.__getWeightsMultiClass(y,over_sampling_factor,under_sampling_factor)
      over = SMOTE(sampling_strategy=over_sampling_factor)
      under = RandomUnderSampler(sampling_strategy=under_sampling_factor)
      steps = [('o', over), ('u', under)]
      pipeline = Pipeline(steps=steps)
      X, y = pipeline.fit_resample(X, y) #pipeline
    elif strategy == Strategy.OVERUNDERSAMPLEWITHNEIGHBORS:
      if classification_type == Classification.MULTICLASS:
        over_sampling_factor,under_sampling_factor=self.__getWeightsMultiClass(y,over_sampling_factor,under_sampling_factor)
      over = SMOTE(sampling_strategy=over_sampling_factor,k_neighbors=k_neighbors)
      under = RandomUnderSampler(sampling_strategy=under_sampling_factor)
      steps = [('o', over), ('u', under)]
      pipeline = Pipeline(steps=steps)
      X, y = pipeline.fit_resample(X, y)
    return (X,y) , X.shape[1]

  def splitData(self,data,test_size=0.2,random_state = 111):
    if not isinstance(data,tuple):
       X,y = self.__getXY(data)
    else:
      X,y = data
    return train_test_split(X,y,test_size=test_size,random_state = random_state)

  def scaleData(self,X_train,X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test =  scaler.transform(X_test)
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    return X_train,X_test
  
  def splitAndScaleData(self,data,test_size=0.2):
    X_train, X_test, y_train, y_test = self.splitData(data,test_size)
    X_train_scaled, X_test_scaled = self.scaleData( X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

  def __resampleSignal(self,signals=None,length=9000):
    data= self.data if signals is None else signals
    return np.array([signal.resample(y,length) for y in data])

  def __resizeSignal(self,signal,length=9000):
    if len(signal)>=length:
      return signal[:length]
    if not isinstance(signal,list):
      signal= signal.tolist()
    return  signal + [0.] * (length-len(signal))

  def resizeData(self,data=None,length=18000):
    data = self.data if data is None else data
    return np.array([(self.__resizeSignal(a[0],length),a[1]) for a in data])

  def resizeAndScale(self,scaler,data=None):
    resized_data=self.resizeData(data)
    scaled_data = scaler.transform(resized_data)
    # Maybe add crop 
    return scaled_data

  def __getWeightsMultiClass(self,y,over_sampling_factor,under_sampling_factor):
    classes = self.getNumberSamplesEachClass(y)
    weights_over = {} 
    weights_under = {} 
    for k,v in classes.items():
      if v/len(y) < over_sampling_factor:
        weights_over[self.classes[k]] = (int) (len(y) * over_sampling_factor)
      else:
        weights_over[self.classes[k]]=v
    for k,v in weights_over.items():
      if v/len(y) > under_sampling_factor:
        weights_under[k] = (int) (len(y) * under_sampling_factor)
      else:
        weights_under[k]=v
    return weights_over, weights_under

  def __extend_ts(self,ts, length):
    extended = np.zeros(length)
    if isinstance(ts,list):
      l=len(ts)
    else:
      l=ts.shape[0]
    siglength = np.min([length, l]) # ,ts.shape[0]
    extended[:siglength] = ts[:siglength]
    return extended 

  def extendData(self,data = None, length = 18000):#
    if data is None:
      data=self.data
    extend_all = np.array([self.__extend_ts(signal, length) for signal in data[:,0].tolist()])
    #print(extend_all.shape)
    #print(self.data[:,0].shape)
    return extend_all, data[:,1].reshape(-1,1).astype('float')

  def extendTestData(self,data=None,length=18000):
    if data is None:
      data=self.data
    extend_all = np.array([self.__extend_ts(signal, length) for signal in data.tolist()])
    #print(extend_all.shape)
    #print(self.data[:,0].shape)
    return extend_all
  
  def __spectrogram(self,data, nperseg=64, noverlap=32, log_spectrogram = False):
    fs = 300
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx,[0,2,1])
    if log_spectrogram:
        Sxx = abs(Sxx) # Make sure, all values are positive before taking log
        mask = Sxx > 0 # We dont want to take the log of zero
        Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx

  # Data augmentation scheme: Dropout bursts
  def __zero_filter(self,input, threshold=2, depth=8):
    shape = input.shape
    # compensate for lost length due to mask processing
    noise_shape = [shape[0], shape[1] + depth]
    
    # Generate random noise
    noise = np.random.normal(0,1,noise_shape)
    
    # Pick positions where the noise is above a certain threshold
    mask = np.greater(noise, threshold)
    
    # grow a neighbourhood of True values with at least length depth+1
    for d in range(depth):
        mask = np.logical_or(mask[:, :-1], mask[:, 1:])
    output = np.where(mask, np.zeros(shape), input)
    return output

  # Helper functions needed for data augmentation
  def __stretch_squeeze(self,source, length):
    target = np.zeros([1, length])
    interpol_obj = scipy.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result

  def __fit_tolength(self,source, length):
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target

  # Data augmentation scheme: Random resampling
  def __random_resample(self,signals, upscale_factor = 1):
    [n_signals,length] = signals.shape
    # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
    new_length = np.random.randint(
        low=int(length*80/120),
        high=int(length*80/60),
        size=[n_signals, upscale_factor])
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    sigs = [self.__stretch_squeeze(s,l) for s,nl in zip(signals,new_length) for l in nl]
    sigs = [self.__fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs

    

  def generateSpectograms(self,ecg_lead_list_padded,max_length =18000,nperseg=64,noverlap=32 ):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    #X = np.empty((kwargs['batch_size'], *kwargs['dim'], kwargs['n_channels']), dtype = float)
    #y = np.empty((kwargs['batch_size']), dtype = int)

    X = []

    # Generate data
    #for i, ID in enumerate(list_IDs_temp):
    for data in ecg_lead_list_padded:
        #data = extend_ts(self.h5file[ID]['ecgdata'][:, 0], sequence_length)
        data = np.reshape(data, (1, len(data)))
        #print(data.shape)
        # data augmentation
        if False:
            # dropout bursts
            data = self.__zero_filter(data)
            # random resampling
            data = self.__random_resample(data)
        
        # Generate spectrogram
        data_spectrogram = self.__spectrogram(data, nperseg, noverlap)[2]
        dimension = data_spectrogram[0].shape
        # Normalize spectrogram
        data_norm = (data_spectrogram - np.mean(data_spectrogram))/np.std(data_spectrogram)
        X.append(data_norm) #np.expand_dims(data_norm, axis = 3)
    return np.array(X) , dimension

  def plotSpectogram(self,spectogram):
    plt.imshow(spectograms[0,:,:].transpose(), cmap = 'jet', aspect = 'auto')


class Model:
    def __init__(self,input_shape,classification_type = Classification.MULTICLASS, test = True):
      self.model = Sequential()
      self.classification_type = classification_type
      self.input_shape = input_shape
      self.test = test

    def train(self,X_train : np.ndarray ,X_test : np.ndarray,y_train : np.ndarray,y_test : np.ndarray, epochs : int, batch_size : int, verbose : int, shuffle : bool) -> tf.keras.callbacks.History:
        """Train the model"""
        pass
    
    def predict(self,data,batch_size):
      return self.model.predict(data,batch_size)

    def predictitionToClass(self,y_pred,threshhold=0.5):
      if self.classification_type == Classification.MULTICLASS:
        return np.argmax(y_pred, axis=1)
      return self.__roundit(y_pred,threshhold)

    def classIdxToLabels(self,class_pred,classes):
      return np.vectorize(dict((v,k) for k,v in classes.items()).get)(class_pred)  
  
    def formatPrediction(self,labels_pred,ecg_names):
      return [(ecg_names[i], labels_pred[i]) for i in range(0, len(ecg_names))] 

    def loadModel(self,checkpoint_path):
      self.model.reset_states()
      self.model.load_weights(checkpoint_path)
      loss = tf.keras.losses.BinaryCrossentropy() if self.classification_type == Classification.BINARY else tf.keras.losses.SparseCategoricalCrossentropy()
      metrics=['accuracy']
      self.model.compile(loss=loss, optimizer='adam', metrics=metrics)
      return self.model
    
    def __roundit (self,list,threshhold=0.5):
      #list[np.isnan(list)] = 0
      return np.array([1. if a>threshhold else 0. for a in list])

class LSTMModel(Model):
  def __init__(self,dimension,classification_type = Classification.MULTICLASS,test=True):
    super().__init__(dimension,classification_type,test)
    self.__buildModel(dimension,test)

  def __buildModel(self,dimension,test=False):
      self.model.add(LSTM(32, return_sequences=True, input_shape=((dimension,1))))
      self.model.add(LSTM(16, return_sequences=True))
      self.model.add(LSTM(4, return_sequences=True))
      self.model.add(Flatten())
      self.model.add(Dense(64))
      self.model.add(Dense(16))
      if self.classification_type == Classification.MULTICLASS:
        self.model.add(Dense(4, activation='softmax'))
      else:
        self.model.add(Dense(1, activation='sigmoid'))
      if not test: 
        self.model.summary()

  def train(self,X_train,X_test,y_train,y_test,epochs=50,batch_size = 128,verbose = 1,shuffle=True):
    model_name= self.__generateName(epochs,batch_size)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/"+model_name,save_weights_only=True,verbose=1, monitor='val_accuracy',mode='max',save_best_only=True)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    loss = tf.keras.losses.BinaryCrossentropy() if self.classification_type == Classification.BINARY else tf.keras.losses.SparseCategoricalCrossentropy()
    metrics=['accuracy']
    self.model.compile(loss=loss, optimizer='adam', metrics=metrics)
    history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=shuffle, verbose=verbose, validation_data=(X_test,y_test),callbacks=[callback,es,tensorboard_callback])
    self.model.save_weights("models/"+model_name)
    return history

  def evaluate(self,X,y,verbose=1):
    loss, acc = self.model.evaluate(X, y, verbose=2)
    return loss,acc

  def loadModel(self,checkpoint_path):
    self.model.reset_states()
    self.model.load_weights(checkpoint_path)
    loss = tf.keras.losses.BinaryCrossentropy() if self.classification_type == Classification.BINARY else tf.keras.losses.SparseCategoricalCrossentropy()
    metrics=['accuracy']
    self.model.compile(loss=loss, optimizer='adam', metrics=metrics)
    return self.model

  def __generateName(self, epochs=50, batch_size = 32):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print("date and time =", dt_string)
    model_name= ""
    if self.classification_type == Classification.MULTICLASS:
      model_name = f"LSTM_Multiclass_{epochs}epochs_{batch_size}batchsize_"+dt_string
    else:
      model_name = f"LSTM_Binary_{epochs}epochs_{batch_size}batchsize_"+dt_string
    return model_name


class CNNModel(Model):
  def __init__(self,dimension,classification_type = Classification.MULTICLASS,test=True):
    super().__init__(dimension,classification_type,test)
    self.__buildModel(dimension,test)

  def __buildModel(self,dimension,n_blocks = 6, test = False):
    filters_start = 32 # Number of convolutional filters
    layer_filters = filters_start # Start with these filters
    filters_growth = 32 # Filter increase after each convBlock
    strides_start = (1, 1) # Strides at the beginning of each convBlock
    strides_end = (2, 2) # Strides at the end of each convBlock
    depth = 4 # Number of convolutional layers in each convBlock
    n_blocks = 6 # Number of ConBlocks
    n_channels = 1 # Number of color channels
    input_shape = (*dimension, n_channels) # input shape for first layer
    classes = 4 if self.classification_type == Classification.MULTICLASS else 2
    for block in range(n_blocks):

      if block == 0:
          provide_input = True
      else:
          provide_input = False
      
      self.model = self.__conv2d_block(self.model, depth,
                          layer_filters,
                          filters_growth,
                          strides_start, strides_end,
                          input_shape,
                          first_layer = provide_input)
      
      layer_filters += filters_growth

    self.model.add(layers.Reshape((-1, 224))) #(batch, time steps, filters)
    self.model.add(layers.core.Masking(mask_value = 0.0))
    self.model.add(self.__MeanOverTime())

    # Ouutput layer
    if self.classification_type == Classification.MULTICLASS:
      self.model.add(layers.Dense(4, activation='softmax', kernel_regularizer = regularizers.l2(0.1)))
    else:
      self.model.add(layers.Dense(1, activation='sigmoid'))
    if not test: 
      self.model.summary()
  
  def train(self,X_train,X_test,y_train,y_test,epochs=50,batch_size = 128,verbose = 1,shuffle=True):
    model_name= self.__generateName(epochs,batch_size)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/"+model_name,save_weights_only=True,verbose=1, monitor='val_accuracy',mode='max',save_best_only=True)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    loss = tf.keras.losses.BinaryCrossentropy() if self.classification_type == Classification.BINARY else tf.keras.losses.SparseCategoricalCrossentropy()
    metrics=['accuracy']
    self.model.compile(loss=loss, optimizer='adam', metrics=metrics)
    history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=shuffle, verbose=verbose, validation_data=(X_test,y_test),callbacks=[callback,es,tensorboard_callback])
    #history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=shuffle, verbose=verbose, validation_data=(X_test,y_test),callbacks=[callback])
    self.model.save_weights("models/"+model_name)
    return history
    
  def __conv2d_block(self,model, depth, layer_filters, filters_growth, 
                 strides_start, strides_end, input_shape, first_layer = False):
    
    ''' Convolutional block. 
    depth: number of convolutional layers in the block (4)
    filters: 2D kernel size (32)
    filters_growth: kernel size increase at the end of block (32)
    first_layer: provide input_shape for first layer'''
    
    # Fixed parameters for convolution
    conv_parms = {'kernel_size': (3, 3),
                  'padding': 'same',
                  'dilation_rate': (1, 1),
                  'activation': None,
                  'data_format': 'channels_last',
                  'kernel_initializer': 'glorot_normal'}

    for l in range(depth):
      if first_layer:
        model.add(layers.Conv2D(filters = layer_filters,
                                  strides = strides_start,
                                  input_shape = input_shape, **conv_parms))
        first_layer = False
      
      else:
        if l == depth - 1:
          layer_filters += filters_growth
          model.add(layers.Conv2D(filters = layer_filters,
                                      strides = strides_end, **conv_parms))
        else:
          model.add(layers.Conv2D(filters = layer_filters,
                                      strides = strides_start, **conv_parms))
      
      model.add(layers.BatchNormalization(center = True, scale = True))
      model.add(layers.Activation('relu'))
  
    return model

  def __MeanOverTime(self):
    lam_layer = layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (1, s[2]))
    return lam_layer

  def __generateName(self,epochs=50,batch_size = 32):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print("date and time =", dt_string)
    model_name= ""
    if self.classification_type == Classification.MULTICLASS:
      model_name = f"CNN_Multiclass_{epochs}epochs_{batch_size}batchsize_"+dt_string
    else:
      model_name = f"CNN_Binary_{epochs}epochs_{batch_size}batchsize_"+dt_string
    return model_name

class Evaluation():
  def __init__(self,classification_type= Classification.MULTICLASS):
    self.classification_type=classification_type
  
  def computeF1(self,y_pred,y_true,average=None):
    if self.classification_type==Classification.MULTICLASS:
      return f1_score(y_true, y_pred, average=average)
    return f1_score(y_true, y_pred)

if __name__ == '__main__':
  # Make False if Jupyter used
  if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, required=True, help='BINARY: 0, MULTICLASS:1')
    parser.add_argument('--model', type=int, required=True, help='LSTM: 0, CNN:1')
    parser.add_argument('--sampling', type=int,required = True, help='Oversampling: 0, Overundersampling:1')
    parser.add_argument('--over',type=float,help='Oversampling rate')
    parser.add_argument('--under',type=float,help='Undersampling rate')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--extension', type=int, help='WITHOUT: 0, WITH:1')
    args = parser.parse_args()
    model= ModelType.CNN if args.model else ModelType.LSTM
    mode= Classification.MULTICLASS if args.mode else Classification.BINARY
    batch_size = 128 if args.batch is None else args.batch
    epochs = 50 if args.epochs is None else args.epochs
    strategy = Strategy.OVERUNDERSAMPLE if args.sampling else Strategy.OVERSAMPLE
    over_sampling_factor= 0.1 if args.over is None else args.over
    under_sampling_factor=0.5 if args.under is None else args.under
    extension = False if args.extension is None else args.extension
  if False:
    model = ModelType.CNN
    mode= Classification.MULTICLASS
    batch_size = 128
    epochs = 50
    Strategy = Strategy.OVERUNDERSAMPLE
    over_sampling_factor= 0.1
    under_sampling_factor=0.5
    
    from google.colab import drive
    drive.mount('/content/drive')
    #!cp "/content/drive/My Drive/Wettbewerb_KI/training.zip" .
    #!unzip training.zip
    #!pip install py-ecg-detectors
    #!sudo pip install imbalanced-learn

  if model == ModelType.LSTM:
    pd = preprocessData(mode)
    if extension:
      filtered_data = pd.resizeData()
    else:
      filtered_data = pd.filterData()
    balanced_data,dimension = pd.balanceData(filtered_data,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor)
    X_train, X_test, y_train, y_test = pd.splitAndScaleData(balanced_data)
    model = LSTMModel(dimension,mode)
    history= model.train(X_train, X_test, y_train, y_test,epochs,batch_size)
    y_pred = model.predict(X_test, batch_size)
    class_pred = model.predictitionToClass(y_pred) 
    ev = Evaluation(mode)
    print(ev.computeF1(class_pred,y_test))
    print(ev.computeF1(class_pred,y_test,"macro"))
  else:
    pd_CNN = preprocessData(mode)
    extended_data_CNN = pd_CNN.extendData()
    balanced_data_CNN = pd_CNN.balanceData(extended_data_CNN,strategy,over_sampling_factor=over_sampling_factor,under_sampling_factor=under_sampling_factor) #Strategy.OVERUNDERSAMPLE
    spectograms,dimension = pd_CNN.generateSpectograms(balanced_data_CNN[0])
    X_train, X_test, y_train, y_test = pd_CNN.splitData((spectograms,balanced_data_CNN[1]))
    model = CNNModel(dimension,mode)
    history = model.train(X_train.reshape((-1, *dimension)), X_test.reshape((len(y_test),*dimension)), y_train, y_test,epochs,batch_size)
    y_pred = model.predict(X_test.reshape((-1, *dimension)), batch_size)
    class_pred = model.predictitionToClass(y_pred) 
    ev = Evaluation(mode)
    print(ev.computeF1(class_pred,y_test))
    print(ev.computeF1(class_pred,y_test,"micro"))
    print(ev.computeF1(class_pred,y_test,"macro"))