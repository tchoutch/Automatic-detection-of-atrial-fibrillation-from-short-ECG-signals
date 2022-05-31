# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import math

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references('training') # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
sdnn_afib = np.array([])
for idx, ecg_lead in enumerate(ecg_leads):
    r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
    sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
    if ecg_labels[idx]=='N':
      sdnn_normal = np.append(sdnn_normal,sdnn)         # Zuordnung zu "Normal"
    if ecg_labels[idx]=='A':
      sdnn_afib = np.append(sdnn_afib,sdnn)             # Zuordnung zu "Vorhofflimmern"
    if (idx % 100)==0:
      print(str(idx) + "\t EKG Signale wurden verarbeitet.")

fig, axs = plt.subplots(2,1, constrained_layout=True)
axs[0].hist(sdnn_normal,2000)
axs[0].set_xlim([0, 300])
axs[0].set_title("Normal")
axs[0].set_xlabel("SDNN (ms)")
axs[0].set_ylabel("Anzahl")
axs[1].hist(sdnn_afib,300)
axs[1].set_xlim([0, 300])
axs[1].set_title("Vorhofflimmern")
axs[1].set_xlabel("SDNN (ms)")
axs[1].set_ylabel("Anzahl")
#plt.show()

sdnn_total = np.append(sdnn_normal,sdnn_afib) # Kombination der beiden SDNN-Listen
p05 = np.nanpercentile(sdnn_total,5)          # untere Schwelle
p95 = np.nanpercentile(sdnn_total,95)         # obere Schwelle
thresholds = np.linspace(p05, p95, num=20)    # Liste aller möglichen Schwellwerte
F1 = np.array([])
for th in thresholds:
  TP = np.sum(sdnn_afib>=th)                  # Richtig Positiv
  TN = np.sum(sdnn_normal<th)                 # Richtig Negativ
  FP = np.sum(sdnn_normal>=th)                # Falsch Positiv
  FN = np.sum(sdnn_afib<th)                   # Falsch Negativ
  F1 = np.append(F1, TP / (TP + 1/2*(FP+FN))) # Berechnung des F1-Scores

th_opt=thresholds[np.argmax(F1)]
print(th_opt)              # Bestimmung des Schwellwertes mit dem höchsten F1-Score

if os.path.exists("model.npy"):
    os.remove("model.npy")
with open('model.npy', 'wb') as f:
    np.save(f, th_opt)


fig, ax = plt.subplots()
ax.plot(thresholds,F1)
ax.plot(th_opt,F1[np.argmax(F1)],'xr')
ax.set_title("Schwellwert")
ax.set_xlabel("SDNN (ms)")
ax.set_ylabel("F1")
#plt.show()

fig, axs = plt.subplots(2,1, constrained_layout=True)
axs[0].hist(sdnn_normal,2000)
axs[0].set_xlim([0, 300])
tmp = axs[0].get_ylim()
axs[0].plot([th_opt,th_opt],[0,10000])
axs[0].set_ylim(tmp)
axs[0].set_title("Normal")
axs[0].set_xlabel("SDNN (ms)")
axs[0].set_ylabel("Anzahl")
axs[1].hist(sdnn_afib,300)
axs[1].set_xlim([0, 300])
tmp = axs[1].get_ylim()
axs[1].plot([th_opt,th_opt],[0,10000])
axs[1].set_ylim(tmp)
axs[1].set_title("Vorhofflimmern")
axs[1].set_xlabel("SDNN (ms)")
axs[1].set_ylabel("Anzahl")
#plt.show()


rus=RandomUnderSampler(sampling_strategy=1)

#create label, 1 for normal 0 for afib
label=np.zeros(3581)+1
label2=np.zeros(521)
y=np.concatenate((label,label2))

#undersampling
sdnn_total=sdnn_total.reshape((4102,1))
X_res, y_res=rus.fit_resample(sdnn_total,y)
#print(X_res)

#Split training validation
X_train, X_valid, y_train, y_valid = train_test_split(X_res, y_res, test_size=0.1, random_state=27)

# Filtering nan
X_train[np.isnan(X_train)]=0
X_valid[np.isnan(X_valid)]=0

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
filename = 'RandomForestModel.sav'
pickle.dump(model, open(filename, 'wb'))

params = {'criterion':('gini', 'entropy'), 'max_depth':[3, 4, 5], 'n_estimators':[10, 20, 30], 'random_state':[0]}
gscv = GridSearchCV(model, params, cv=5)
gscv.fit(X_valid, y_valid)
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))


model = RandomForestClassifier(max_depth=3, n_estimators=20, criterion='gini', random_state=0)
model.fit(X_train, y_train)
predictions=model.predict(X_train)
#print(predictions)
errors=abs(predictions-y_train)
print('Mean Absolute Error:', round(np.mean(errors), 2))

from sklearn.metrics import mean_squared_error
import math
mse =mean_squared_error(y_train, predictions)
rmse = math.sqrt(mse)
print('Accuracy for Random Forest',100*max(0,rmse)) 