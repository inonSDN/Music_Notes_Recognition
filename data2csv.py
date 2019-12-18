import numpy as np
import os
from progress.bar import Bar
import csv
from pandas import DataFrame
import pandas as pd
import librosa


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.1 * noise

    return data_noise

def stretch(data, rate=1):
    data = librosa.effects.time_stretch(data, rate)
        
    return data

    
classes = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
base_path = 'ALL/'
for r, d, files in os.walk(base_path):
    for f in files:
        for cl in range(len(classes)):
            if r == 'ALL/'+ classes[cl] :
                datao, sr = librosa.load('ALL/'+ classes[cl] + '/' + f)
                data_noise = add_noise(datao)

                librosa.output.write_wav('ALL/'+ classes[cl] + '/' + 'noise_01' + f, data_noise, sr)

# data to csv 
data = {}
path = []
c = []
label = []

for r, d, files in os.walk(base_path):
    for f in files:
        for cl in range(len(classes)):
            if r == 'ALL/'+ classes[cl] :
                path.append(f)
                c.append(cl)
                label.append(classes[cl])

data = {'Path':path,
        'Classes':c,
        'Labels':label}
df = DataFrame(data, columns= ['Path','Classes','Labels'])

export_csv = df.to_csv('data.csv', index = None, header=True) 

print(df)