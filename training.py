import keras
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from load_data import *

# load data
X_train, X_test, y_train, y_test = load_train_test('data.csv')

# load model 
model = load_model()

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

history = model.fit(
	x=X_train, 
	y=y_train,
    epochs=70,
    batch_size=16,
    validation_data= (X_test, y_test))

model_json = model.to_json()
with open("note_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("note_weight.h5")

score = model.evaluate(x=X_test, y=y_test)
print(score)

# plot graph train and test 
plt.plot(np.squeeze(history.history['acc']))
plt.plot(np.squeeze(history.history['loss']))
plt.plot(np.squeeze(history.history['val_acc']))
plt.plot(np.squeeze(history.history['val_loss']))

plt.legend(['acc', 'loss', 'val_acc', 'val_loss'], loc='upper left')

plt.ylabel('acc & loss')
plt.xlabel('epoch')
plt.title('acc & loss')
plt.show()