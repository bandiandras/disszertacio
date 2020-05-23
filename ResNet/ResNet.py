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


def build_resnet(input_shape, nb_classes, file_path):
    n_feature_maps = 64
    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1 
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum 
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2 
    conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum 
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3 
    conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal 
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL 
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
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

    if DATASET == "MCYT":
        if (SKILLED_FORGERY == True):
            savedModelName = 'RESNET_MCYT_Skilled.hdf5'
        else:
            savedModelName = 'RESNET_MCYT_Random.hdf5'
        cb, model = build_resnet((128, 8), 100, savedModelName)
    if DATASET == "MOBISIG":     
        if (SKILLED_FORGERY == True):
            savedModelName = 'RESNET_MOBISIG_Skilled.hdf5'
        else:
            savedModelName = 'RESNET_MOBISIG_Random.hdf5'
        cb, model = build_resnet((128, 8), 83, savedModelName)

    
    if (SKILLED_FORGERY == True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)

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
        
        X_test = np.concatenate((X_test, X))
        y_test = np.concatenate((y_test, y))        

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
    
    print (metrics.accuracy_score(np.argmax(y_test, axis=-1),
                       np.argmax(model.predict(X_test), axis=-1)))
    keras.backend.clear_session() 


if __name__ == "__main__":
    main()