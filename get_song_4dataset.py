import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook,Workbook
from scipy.signal import butter, lfilter, freqz, find_peaks
import scipy.fftpack

def detect_strum(audiofile, offset, duration):

    
    data, sr = librosa.load(audiofile,offset=offset,duration=duration)

    x = np.linspace(0, len(data), len(data))

    """
    Find peak of signal that threshold = 0.45

    peak_s is time that have peak value
    """
    max_d = []
    Thes = 0.2
    peak, _ = find_peaks(data,height=0.2)
    plt.plot(data)
    plt.plot(peak, data[peak], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")


    peak_s = list(peak*duration/len(data))
    

    """
    set to check duration 0.25 s
    (120 bpm 1/2 beat)
    """
    idx_time = []
    peak_time = []
    peak_time.append(peak_s[0])
    for i in range(1, len(peak_s)):
        if peak_s[i] - peak_s[i-1] > 0.25:
            peak_time.append(peak_s[i])

    plt.show()


    print(len(peak_time))
    return peak_time

def duration_strum(audiofile, offset, duration):
    time = detect_strum(audiofile, offset, duration)
    drt = []
    duration_st = []

    for i in range(0,len(time)+1):
        if i == 0:
            drt.append(0)
            drt.append(time[i])
        elif i == len(time):
            drt.append(time[i-1])
            drt.append(duration)
        else:
            drt.append(time[i-1])
            drt.append(time[i])
        
        duration_st.append(drt)
        drt = []

    return duration_st, time

duration_st, time = duration_strum('guitar1.wav', 0, 184)

for i in range(len(duration_st)):
    y, sr = librosa.load('guitar1.wav', offset=duration_st[i][0], duration=1)
    librosa.output.write_wav('Dataset_from_code/Guitar1/' + str(i) + '.wav', y, sr)