import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook,Workbook
from scipy.signal import butter, lfilter, freqz, find_peaks
import scipy.fftpack
from keras.models import Sequential, Model, load_model, model_from_json

from Strum_detection import *

file_test = 'Dataset/testinon3.wav'
offset = 0
duration = 10

time, peak_value = peak_strum(file_test, offset, duration)

json_file = open('note_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("note_weight.h5")
print("Loaded model from disk")

loaded_model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

class_note = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']

for i in range(len(time)):
    y, sr = librosa.load(file_test, offset=time[i],duration=0.25)  
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    ps_test = np.array([ps.reshape( (128, 11, 1) )])

    predictions = loaded_model.predict_classes(ps_test)
    if i != len(time)-1:
        print(str(round(time[i],2)) + ' s - '  + str(round(time[i+1],2)) + ' s note: ' + class_note[predictions[0]])
    else:
        print(str(round(time[i],2)) + ' s note: ' + class_note[predictions[0]])