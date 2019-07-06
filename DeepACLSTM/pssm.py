'''
The training dataset CB5534
the test datsaset CB513
The experiment evaluate the performance of our method based on profile features
'''


# coding: utf-8
from __future__ import division
import numpy as np
import gzip
import h5py
import time
import sys
#########################################################################
# ##############Cullpdb+profile_6133_filtered #########################
########################################################################

print("Loading train data (Cullpdb_filted)...")
t1 = time.time()
train_data = np.load(gzip.open('data/cullpdb+profile_6133_filtered.npy.gz', 'rb'))
t2 = time.time()

print  type(train_data), len(train_data), train_data.shape

print 'Memory:', sys.getsizeof(train_data) / (1024 * 1024), 'M'
print 'load 6133_filtered Runtime:', '%.2f' % (t2 - t1), 'seconds'

train_data= np.reshape(train_data, (-1, 700, 57))
print  'train_data',type(train_data), len(train_data), train_data.shape
dataonehot=train_data[:, :, 0:21]#sequence feature
# print 'sequence feature',dataonehot[1,:3,:]
datapssm=train_data[:, :, 35:56]#profile feature
# print 'profile feature',datapssm[1,:3,:]

train_label = train_data[:, :, 22:30]    # secondary struture label , 8-d
dataonehot=train_data[:, :, 0:21]#sequence feature
datapssm=train_data[:, :, 35:56]#profile feature

# shuffle data
np.random.seed(2018)
num_seqs, seqlen, feature_dim = np.shape(train_data)
num_classes = 8
seq_names = np.arange(0, num_seqs)
np.random.shuffle(seq_names)


#train data
trainhot = dataonehot[seq_names[:5278]]
trainlabel = train_label[seq_names[:5278]]
trainpssm = datapssm[seq_names[:5278]] 

#val data
vallabel = train_label[seq_names[5278:5534]]
valpssm = datapssm[seq_names[5278:5534]] 
valhot = dataonehot[seq_names[5278:5534]]


####################################################
####################test data#######################
####################################################
print("Loading Test data (CB513)...")
t0 = time.time()
test_data = np.load(gzip.open('data/cb513+profile_split1.npy.gz', 'rb'))
t00 = time.time()
# print  'test_data',type(test_data), len(test_data), test_data.shape
print 'load cb513 Runtime:', '%.2f' % (t00 - t0), 'seconds'
test_data = np.reshape(test_data,(-1,700,57))
print  'test_data',type(test_data), len(test_data), test_data.shape
testhot = test_data[:, :, 0:21]#sequence feature
testpssm = test_data[:, :, 35:56]#profile feature

test_label = test_data[:, :, 22:30]    # secondary struture label , 8-d
print  'test_label',type(test_label), len(test_label), test_label.shape

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
np.random.seed(1)
rn.seed(1)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
import keras
from keras import backend as K
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###To do 
def convert_Q8_to_readable(predictedSS):
    """
    predictedSS is a 2D matrix, Proteins * residue labels
    """
    ssConvertMap = {0: 'H', 1: 'B', 2: 'E', 3: 'G', 4: 'I', 5: 'T', 6: 'S', 7: 'L'}
    result = []
    for i in range(len(predictedSS)):
        single=[]
        for j in range(0, 700):
            single.append(ssConvertMap[predictedSS[i][j]])
        result.append( ''.join(single) )
    return result
###
def plotLoss(history):

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+'lstmloss06' +".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait', 
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.title('model  accuracy')
    plt.ylabel('Q8 accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['weighted_accuracy'])
    plt.plot(history.history['val_weighted_accuracy'])
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')

    plt.savefig("figures/"+'lstmaccuracy06'+ ".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait', 
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
def build_model():
    auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)
    concat = auxiliary_input
    
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
    aux_output = TimeDistributedDense(8, activation='softmax', name='aux_output')(protein_features)

    deepaclstm = Model(input=[auxiliary_input], output=[aux_output])
    adam = Adam(lr=0.003)
    deepaclstm.compile(optimizer = adam, loss={'aux_output': 'categorical_crossentropy'}, metrics=['weighted_accuracy'])
    deepaclstm.summary()
    return deepaclstm

###building the model
model = build_model()
earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=5, verbose=1, mode='auto')

load_file = "./model/ac_LSTM_best_time_06.h5" 
checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)
#####Training the model
history=model.fit({'aux_input': trainpssm}, {'aux_output': trainlabel},validation_data=({'aux_input': valpssm},{'aux_output': vallabel}),
        nb_epoch=200, batch_size=42, callbacks=[checkpointer, earlyStopping], verbose=2, shuffle=True)
plotLoss(history)
model.load_weights(load_file)

#########evaluating the model##############
score = model.evaluate({'aux_input': testpssm},{'aux_output': test_label}, verbose=2, batch_size=2)
print score 
print 'test loss:', score[0]
print 'test accuracy:', score[1]

