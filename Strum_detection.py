import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook,Workbook
from scipy.signal import butter, lfilter, freqz, find_peaks
import scipy.fftpack

def sortSecond(val): 
    return val[1]  

def sortFirst(val): 
    return val[0]  

def detect_window(audiofile, offset, duration):

    # window size 10 ms
    window_time = 0.01
    # threshold of peak is 25%
    threshold = 0.25

    data, sr = librosa.load(audiofile,offset=offset,duration=duration)

    peak, _ = find_peaks(data,height=threshold)

    window_peak = int(window_time * len(data) / duration)           # term of data x-axis

    return window_peak, data

def cal_energy(audiofile, offset, duration):

    cal_temp = []
    energy = []

    window_peak, data = detect_window(audiofile, offset, duration)
    
    # calculate amplitude^2 in window
    for d in range(len(data)):
        if d % int(window_peak/2) == 0 :
            if d + window_peak >= len(data):
                break
            
            else:
                for w in range(window_peak):
                    cal_temp.append(data[d+w]*data[d+w])
                
                energy.append([((d + window_peak))*duration/len(data), sum(cal_temp)])
                cal_temp = []

    # test plot
    t = []
    p = []
    for en in range(0,len(energy)):
        t.append(energy[en][0])
        p.append(energy[en][1])

    plt.plot(t,p)
    plt.ylabel("Amplitude")
    plt.xlabel("Second")
    plt.show()

    x = np.linspace(0, len(data), len(data))
    xt = x*duration/len(data)
    plt.plot(xt,data)
    plt.ylabel("Amplitude")
    plt.xlabel("Second")
    plt.show()

    return energy, data

def peak(audiofile, offset, duration, threshold_amp_rate):
    
    # find peak that have amplitude more than threshold_amp_rate
    peak_amp = []
    energy, data = cal_energy(audiofile, offset, duration)
    
    energy.sort(key=sortSecond, reverse=True)

    print(energy)
    print(len(energy))

    threshold_amp = energy[0][1] * threshold_amp_rate
    for e in energy:
        
        if e[1] >= threshold_amp:
            peak_amp.append(e)

    print(len(peak_amp))

    return peak_amp, data

def peak_strum(audiofile, offset, duration, threshold_amp_rate=0.1, strum_time=0.25):

    time = []
    peak_value = []
    keep = []
    check = []

    peak_amp, data = peak(audiofile, offset, duration, threshold_amp_rate)

    keep.append(peak_amp[0])
    for pa in range(1,len(peak_amp)):
        for cc in range(len(keep)):
            if np.abs(peak_amp[pa][0] - keep[cc-1][0]) >= strum_time:
                check.append(1)
            else:
                check.append(0)

            if cc == len(keep) - 1:
                if all(i >= 0.5 for i in check) == True:
                    keep.append(peak_amp[pa])
                check = []

    keep.sort(key=sortFirst, reverse=False)
    for kp in range(0,len(keep)):
        time.append(keep[kp][0])
    
    return time, peak_value
