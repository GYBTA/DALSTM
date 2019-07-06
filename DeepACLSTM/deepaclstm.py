
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, GRU, TimeDistributedDense, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
np.random.seed(2018)
rn.seed(2018)

def build_model():
    # design the deepaclstm model
    main_input = Input(shape=(700,), dtype='float32', name='main_input')
    #main_input = Masking(mask_value=23)(main_input)
    x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
    auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)
    print main_input.get_shape()
    print auxiliary_input.get_shape()
    concat = merge([x, auxiliary_input], mode='concat', concat_axis=-1)    

    conv1_features = Convolution1D(42,1,activation='relu', border_mode='same', W_regularizer=l2(0.001))(concat)
    # print 'conv1_features shape', conv1_features.get_shape()
    conv1_features = Reshape((700, 42, 1))(conv1_features)    

    conv2_features = Convolution2D(42,3,1,activation='relu', border_mode='same', W_regularizer=l2(0.001))(conv1_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()    

    conv2_features = Reshape((700,42*42))(conv2_features)
    conv2_features = Dropout(0.5)(conv2_features)
    conv2_features = Dense(400, activation='relu')(conv2_features)

    #, activation='tanh', inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5
    lstm_f1 = LSTM(output_dim=300,return_sequences=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(conv2_features)
    lstm_b1 = LSTM(output_dim=300, return_sequences=True, go_backwards=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(conv2_features)    

    lstm_f2 = LSTM(output_dim=300, return_sequences=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(lstm_f1)
    lstm_b2 = LSTM(output_dim=300, return_sequences=True, go_backwards=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(lstm_b1)    

    concat_features = merge([lstm_f2, lstm_b2, conv2_features], mode='concat', concat_axis=-1)    

    concat_features = Dropout(0.4)(concat_features)
    protein_features = Dense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)    

    main_output = TimeDistributedDense(8, activation='softmax', name='main_output')(protein_features)    
    

    model = Model(input=[main_input, auxiliary_input], output=[main_output])
    adam = Adam(lr=0.003)
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['weighted_accuracy'])
    model.summary()
    return model

# load train/validation dataset
traindatahot, trainpssm, trainlabel, valdatahot, valpssm, vallabel = load_cul6133_filted()
# load test dataset
testdatahot,testpssm,test_label = load_cb513()

model = build_model()
# print "####### look at data's shape#########"
# print traindatahot.shape, trainpssm.shape, trainlabel.shape, testdatahot.shape, testpssm.shape,testlabel.shape, valdatahot.shape,valpssm.shape,vallabel.shape
earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=5, verbose=1, mode='auto')
######################
# load_file = "./model/ACNN/acnn1-3-42-400-300-blstm-FC600-42-cb6133F-0.5-0.4.h5"
#################################
# load_file = "./model/ac_LSTM_best_time_17.h5" # M: weighted_accuracy E: val_weighted_accuracy
load_file = "./model/ac_LSTM_best_time_26.h5" # M: val_loss E: val_weighted_accuracy
checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)

history=model.fit({'main_input': traindatahot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': valdatahot, 'aux_input': valpssm},{'main_output': vallabel}),
        nb_epoch=200, batch_size=42, callbacks=[checkpointer, earlyStopping], verbose=2, shuffle=True)
# plotLoss(history)
model.load_weights(load_file)
print "#########evaluate:##############"
score = model.evaluate({'main_input': testdatahot, 'aux_input': testpssm},{'main_output': test_label}, verbose=2, batch_size=2)
print score 
print 'test loss:', score[0]
print 'test accuracy:', score[1]
