import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from Utils.utils import *
from settings import *


def build_fcn(input_shape, nb_classes, file_path):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
 
    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
                  metrics=['categorical_accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
                                                  min_lr=0.0001)
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
                                                       save_best_only=True)

    callbacks = [reduce_lr,model_checkpoint]

    return callbacks, model


def main():
    print("FILE GENUINE: "+ FILENAME_GENUINE)
    df = pd.read_csv(FILENAME_GENUINE)
    df = utils.standardize_rows(df)

    y = df[df.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    X = df.drop(df.columns[-1], axis=1).values
    
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, 128, 8)

    savedModelName = ''
    if (SKILLED_FORGERY == True):
        savedModelName = 'FCN_MOBISIG_Skilled.hdf5'
    else:
        savedModelName = 'FCN_MMOBISIG_Random.hdf5'
    #MCYT
    # cb, model = build_fcn((128, 8), 100, savedModelName)
    #MOBISIG
    cb, model = build_fcn((128, 8), 83, savedModelName)

    if (SKILLED_FORGERY == True):
        X_train = X
        y_train = y

        print("FILE FORGERY: "+ FILENAME_FORGERY)
        df_f = pd.read_csv(FILENAME_FORGERY)
        df_f = utils.standardize_rows(df_f)

        y = df_f[df_f.columns[-1]].values
        y = LabelEncoder().fit_transform(y) + 1
        X = df_f.drop(df_f.columns[-1], axis=1).values
        
        enc = OneHotEncoder()
        enc.fit(y.reshape(-1,1))
        y = enc.transform(y.reshape(-1, 1)).toarray()
        X = X.reshape(-1, 128, 8)
        
        X_test = X
        y_test = y
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

    mini_batch_size = int(min(X_train.shape[0]/10, BATCH_SIZE))
    start_time = time.time()
    hist = model.fit(X_train, y_train, 
                        batch_size=mini_batch_size, 
                        epochs=EPOCHS,
                        verbose=True, 
                        validation_data=(X_val, y_val), 
                        callbacks=cb)
    duration = time.time() - start_time
    y_pred = model.predict(X_test)
    
    metrics.accuracy_score(np.argmax(y_test, axis=-1),
                       np.argmax(model.predict(X_test), axis=-1))
    keras.backend.clear_session() 



if __name__ == "__main__":
    main()