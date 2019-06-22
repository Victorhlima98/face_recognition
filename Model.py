import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from keras import layers
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras import optimizers


def model(input_shape,num_classes):
    max_count=100
    reg_val=[]
    lr_val=[]
    test_loss=[]
    test_acc=[]

    for i in range(max_count):

        print ("*"*30)
        print (str(i+1)+"/"+str(max_count))
        print ("*"*30)
    # Sampling learning rate and regularization from a uniform distribution

        reg=10**(np.random.uniform(-4,0))
        lr=10**(np.random.uniform(-3,-4))

    model=Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.6))
    model.add(layers.Conv2D(256,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.Dense(2,activation='sigmoid',kernel_regularizer=regularizers.l2(reg)))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=lr), metrics=['acc'])

    model.summary()
    return model

