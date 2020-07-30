#!/usr/bin/env python
# coding: utf-8

# In[5]:


# load packages
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf # tensorflow v2
# from tensorflow.compat.v1.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
import psycopg2
import sqlalchemy

#from sklearn.externals import joblib

# set random seeds
#np.random.seed(1)
#tf.random.set_random_seed(2)

# limit gpu usage for keras
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# In[6]:


## Model Architect
def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))
    
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    
    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

    # use the MC dropout here
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

deeplob = create_deeplob(100, 40, 64)


# In[3]:


# Load postgreSQL data

# Prepare data
def get_SQLtrain(index, conn):
    
    # Get the indexes of selected training batch starting row
    queryIndex = f'''SELECT index FROM "NoAuction_DecPre"
                     WHERE rand_id = {index};'''
                    #WHERE istrainingdata = 1
                    #AND batch = 7
    ResIndex = pd.read_sql(queryIndex, conn)
    print(ResIndex)
    idx = np.array(ResIndex)
    
    N = len(idx)
    T = 100
    D = 40
    dataX = np.zeros((N, T, D))
    
    dataY = np.zeros(N)
    
    for i in range(len(idx)):
        start = int(idx[i])
        queryLob = f'''SELECT * FROM "NoAuction_DecPre"
                       WHERE index BETWEEN {start} AND {start+T-1};'''      
        ResLob = pd.read_sql(queryLob, conn)
        sqlTrainLob = ResLob.loc[:, 'pa1':'vb10']
        dataX[i] = sqlTrainLob
    
        queryLabel = f'''SELECT * FROM "NoAuction_DecPre_Label1"
                         WHERE index = {start+T-1};'''
        ResLabel = pd.read_sql(queryLabel, conn)
        sqlTrainLabel = ResLabel.at[0, '5-step'] - 1
        dataY[i] = sqlTrainLabel
    
    return dataX.reshape(dataX.shape + (1,)), dataY

def get_SQLtest(conn):

    queryLob1 = f'''SELECT * FROM "NoAuction_DecPre" 
                    WHERE istrainingdata = 0
                    AND batch IN (7, 8, 9);'''
    ResLob1 = pd.read_sql(queryLob1, conn)
    sqlTestLob = ResLob1.loc[:, 'pa1':'vb10']
    
    queryLabel1 = f'''SELECT * FROM "NoAuction_DecPre_Label1"
                      WHERE istrainingdata = 0
                      AND batch IN (7, 8, 9);'''
    ResLabel1 = pd.read_sql(queryLabel1, conn)
    sqlTestLabel = ResLabel1.loc[:, '5-step'] - 1
    #sqlTestLabel = sqlTestLabel*(-1) + 2 - 1
    
    return sqlTestLob, sqlTestLabel


# In[4]:


def data_classification(X, Y, T):
    [N, D] = X.shape
    
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


# In[5]:


# Connect to postgresql on VM
engine = create_engine('postgresql://name:password@155.246.104.52/hftlob')
conn = engine.connect()


# In[6]:


# prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
with engine.connect() as conn:
    # Get test data
    #sqlTestLOB, sqlTestLabel = get_SQLtest(conn)
    #testX_CNN, testY_CNN = data_classification(sqlTestLOB, sqlTestLabel, T=100)
    #testY_CNN = np_utils.to_categorical(testY_CNN, 3)
    
    for i in np.random.choice(range(50), 100):
        # Get training data
        trainX_CNN, trainY_CNN = get_SQLtrain(i, conn)
        trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)
                
        deeplob.fit(trainX_CNN, trainY_CNN, epochs=5, batch_size=128, verbose=2, validation_split=0.2)


# In[ ]:




