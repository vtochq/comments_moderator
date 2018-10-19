#!/usr/bin/python3
import my_lib
import pymongo
import datetime
import codecs
import pickle
#import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gensim
import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors


from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import random
#import nltk
import sys

savefprefix="w2v_26_09_"
regenerate_data = True
EMBEDDING_DIM=500
NUM_WORDS=200000

if regenerate_data:
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    mydb = myclient["mydb"]

    col_comments = mydb["comments"]
    col_assets = mydb["assets"]

    comments = col_comments.find( {}, { 'body': 1, 'status': 1, 'asset_id': 1 }) #.limit(10)

    data = []
    labels = []
    pos = neg = 0
    '''
    print(datetime.datetime.now().replace(microsecond=0), " Loading stopwords")
    with codecs.open('russianSTOPlist.txt', 'r', encoding = 'utf8')\
        as wordsfile: wds = wordsfile.readlines()
    stoplist = []
    for i in wds:
        stoplist.append(i[:-2])


    print("Stop list size: ", len(stoplist))
    '''

    print(datetime.datetime.now().replace(microsecond=0), " Loading data")
    #[:100000]
    for x in comments:
        body = x['body']
        if re.match(r"^((?![әғқңөұүһі]).)*$", body):
            body = my_lib.spec2space(body)
            if len(body.split())<600:
                asset = col_assets.find_one( {'id': x['asset_id']}, {'title': 1} )
                data.append(asset['title']+" "+body)
                if x['status'] == "ACCEPTED":
                  labels.append(1)
                  pos=pos+1
                else:
                  labels.append(0)
                  neg=neg+1


    test_size = round(len(data)*0.01)  # 5% of train_data

    # conver list to numpy array
    labels=np.array(labels)
    #sys.exit()
    '''
    # randomize data order
    print(datetime.datetime.now().replace(microsecond=0), " Randomizing data order")
    indices = np.arange(len(data))
    np.random.shuffle(indices)
#    data = data[indices]
#    labels = labels[indices]
    data = [x for _,x in sorted(zip(indices,data))]
    labels = [x for _,x in sorted(zip(indices,labels))]
    '''

    print(datetime.datetime.now().replace(microsecond=0), " Splitting data to learn and test")
    # split data and test data
    data_test = data[:test_size]
    labels_test = labels[:test_size]

    data = data[test_size:]
    labels = labels[test_size:]

    print(datetime.datetime.now().replace(microsecond=0), " Start tokenazing")
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True, char_level=False)
    tokenizer.fit_on_texts(data)
    print(datetime.datetime.now().replace(microsecond=0) ," Tokenazing learn data")
    data = tokenizer.texts_to_sequences(data)
    print(datetime.datetime.now().replace(microsecond=0), " Tokenazing test data")
    data_test = tokenizer.texts_to_sequences(data_test)
    print('Found %s unique tokens.' % (len(tokenizer.word_index)+1))
    word_dict = tokenizer.word_index

    print(datetime.datetime.now().replace(microsecond=0), " Padding learn data")
    data = keras.preprocessing.sequence.pad_sequences(data, value=0, padding='post')  # , maxlen=256

    print(datetime.datetime.now().replace(microsecond=0), " Loading word2vec")
    word_vectors = KeyedVectors.load_word2vec_format('all.norm-sz500-w10-cb0-it3-min5.w2v', binary=True, unicode_errors='ignore')
    vocabulary_size = min(len(word_dict),NUM_WORDS)

    #word_vectors={}
    print(datetime.datetime.now().replace(microsecond=0), " Zeroing embedding matrix")
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    not_found_in_vec=0
    print(datetime.datetime.now().replace(microsecond=0), " Filling embedding matrix")
    for word, i in word_dict.items():
        if i>=NUM_WORDS:
            continue
        try:
            embedding_matrix[i] = word_vectors[word]
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,0.5,EMBEDDING_DIM)
            not_found_in_vec=not_found_in_vec+1

    print("Words not in w2v: ", not_found_in_vec)
#    del(word_vectors)
    max_word_count = len(max(data, key=len))
    # save train and test data
    print(datetime.datetime.now().replace(microsecond=0), " Saving vars")
    with open(savefprefix+'objs.pkl', 'wb') as f:
        pickle.dump([data, labels, data_test, labels_test, word_dict, embedding_matrix, vocabulary_size, tokenizer, max_word_count], f)

    # save tokenizer and max word count only
    with open(savefprefix+'tokenizerwc.pkl', 'wb') as f:
        pickle.dump([tokenizer, max_word_count], f)


else:

    # loading train_data
    with open(savefprefix+'objs.pkl', 'rb') as f:
        data, labels, data_test, labels_test, word_dict, embedding_matrix, vocabulary_size, tokenizer, max_word_count = pickle.load(f)


"""
print(datetime.datetime.now().replace(microsecond=0), " Counting unknown words in test data")
unk_words=0
for i in range(len(data_test)):
    for j in range(len(data_test[i])):
        if data_test[i][j] == 2:
            unk_words=unk_words+1
"""


#dict_size = len(dict)

print("Learn Data size: ", len(data))
#print("Dict size: ", dict_size)
#print("Unknown words in test data: ", unk_words)
print("Max word count in string: ", max_word_count)

#data = keras.preprocessing.sequence.pad_sequences(data, value=0, padding='post', maxlen=max_word_count)  # , maxlen=256
print(datetime.datetime.now().replace(microsecond=0), " Padding test data")
data_test = keras.preprocessing.sequence.pad_sequences(data_test, value=0, padding='post', maxlen=max_word_count)  # , maxlen=256

#print (data)
'''
# randomize learn data order
if not regenerate_data:
    print(datetime.datetime.now().replace(microsecond=0), " Randomizing learn data")
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = [x for _,x in sorted(zip(indices,data))]
    labels = [x for _,x in sorted(zip(indices,labels))]

    data = data[indices]
    labels = labels[indices]
'''
#print (data)

# ML Magic
exec(open("./ml.py").read())

"""
sequence_length = data.shape[1]
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5

print(datetime.datetime.now().replace(microsecond=0), " Creating embedding layer")
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)

inputs = Input(shape=(sequence_length,))
print(datetime.datetime.now().replace(microsecond=0), " Embedding")
embedding = embedding_layer(inputs)
print(datetime.datetime.now().replace(microsecond=0), " Adding other layers")
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((3*num_filters,))(flatten)
dropout = Dropout(drop)(reshape)
output = Dense(units=1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(dropout)

# this creates a model that includes
print(datetime.datetime.now().replace(microsecond=0), " Creating model")
model = Model(inputs, output)

adam = Adam(lr=1e-3)

print(datetime.datetime.now().replace(microsecond=0), " Start compiling model")
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              metrics=['acc'])

model.summary()

callbacks = [EarlyStopping(monitor='val_loss')]
print(datetime.datetime.now().replace(microsecond=0), " Start learning")
model.fit(data, labels, batch_size=1000, epochs=10, verbose=1, validation_split=0.1, callbacks=callbacks)  # starts training

print(datetime.datetime.now().replace(microsecond=0), " Start saving")
model.save(savefprefix+'model.h5')

print(datetime.datetime.now().replace(microsecond=0), " Start testing")
score = model.evaluate(data_test, labels_test)
print("Score [loose, accurance]: ", score)
"""
'''
model = keras.Sequential()
model.add(keras.layers.Embedding(dict_size, 100, input_length=max_word_count))

model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))
'''

'''
# Conv
model.add(keras.layers.Conv1D(128, 5, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(128, 5, activation='relu'))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Conv1D(128, 5, activation='relu'))
model.add(keras.layers.MaxPooling1D(round(max_word_count/125)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
'''

'''
# Dense
model.add(keras.layers.Flatten())
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.sigmoid))
model.add(keras.layers.MaxPooling1D(5))
model.add(keras.layers.Dense(1, activation=tf.nn.softmax))
'''

'''
model.summary()

#model.get_config()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])

print(len(data))

print("Start learning", datetime.datetime.now().replace(microsecond=0))

history = model.fit(data, labels, epochs=3, validation_split=0.2, verbose=1)

print("Start testing", datetime.datetime.now().replace(microsecond=0))
score = model.evaluate(data_test, labels_test)

print("Score [loose, accurance]: ", score)


model.save(savefprefix+'model.h5')
'''
