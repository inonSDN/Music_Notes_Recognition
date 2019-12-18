import numpy as np
import pandas as pd
import librosa
import random
from progress.bar import Bar


def load_csv_data(file_csv, duration=0.25):

    data = [] 
    data_csv = pd.read_csv(file_csv)
    
    data_csv['path'] =  data_csv['Labels'].astype('str') + '/' + data_csv['Path'].astype('str')
    
    
    with Bar('Loading data:', max=len(data_csv)) as bar:   
        for row in data_csv.itertuples():
            y, sr = librosa.load('Dataset/ALL/'+ row.path,duration=duration)  
            ps = librosa.feature.melspectrogram(y=y, sr=sr)
                    
            data.append( (ps, row.Classes) )
            
            bar.next()

    return data

def load_train_test(file_csv):
    dataset = load_csv_data(file_csv, duration=0.25)

    random.shuffle(dataset)
    # seperate data for train and test
    train = dataset[:2100]
    test = dataset[2100:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    X_train = np.array([x.reshape( (128, 11, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 11, 1) ) for x in X_test])


    # One-Hot encoding for classes
    y_train = np.array(keras.utils.to_categorical(y_train, 12))
    y_test_values=y_test
    y_test = np.array(keras.utils.to_categorical(y_test, 12))

    print('Finish preparing data')

    return X_train, X_test, y_train, y_test

def load_model():

    model = Sequential()

    input_shape=(128, 11, 1)

    model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape))
    model.add(MaxPooling2D((4, 2), strides=(1, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(24, (5, 5), padding="valid"))
    model.add(MaxPooling2D((4, 2), strides=(2, 2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(rate=0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(12))
    model.add(Activation('softmax'))

    print('Load model')
    print(model.summary())

    return model
