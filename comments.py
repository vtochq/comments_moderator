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

savefprefix="nurall_"
regenerate_data = True

if regenerate_data:
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    mydb = myclient["talk"]

    mycol = mydb["comments"]

    #x = mycol.find_one()
    #print(x)

    #myquery = { 'status': 'REJECTED' }
    myquery = {}

    mydoc = mycol.find(myquery, { 'body': 1, 'status': 1 })

    data = []
    labels = []
    pos = neg = 0

    with codecs.open('russianSTOPlist.txt', 'r', encoding = 'utf8')\
        as wordsfile: wds = wordsfile.readlines()
    stoplist = []
    for i in wds:
        stoplist.append(i[:-2])

    print("Stop list size: ", len(stoplist))

    #[:100000]
    for x in mydoc[:1000]:
        if len(x['body'].split())<100:
            data.append(my_lib.splitstringStemKazStop(x['body'], stoplist))
            if x['status'] == "ACCEPTED":
              labels.append(1)
              pos=pos+1;
            else:
              labels.append(0)
              neg=neg+1;


    test_size = round(len(data)*0.05)  # 5% of train_data

    # conver list to numpy array
    labels=np.array(labels)

    # split data and test data
    data_test = data[:test_size]
    labels_test = labels[:test_size]

    data = data[test_size:]
    labels = labels[test_size:]


    dict = my_lib.WordsDic(data)

    print("Negative: ",neg,". Positive: ", pos)

    data = my_lib.WordReplaceNP(data, dict)
    data_test = my_lib.WordReplaceNP(data_test, dict)

    # save dictionary
    with codecs.open(savefprefix+'dict.txt', 'w', encoding = 'utf8')\
        as wordfile: wordfile.writelines(i + '\n' for i in dict)
    # save train and test data
    with open(savefprefix+'objs.pkl', 'wb') as f:
        pickle.dump([data, labels, data_test, labels_test], f)

else:
    # loading dictionary
    with codecs.open(savefprefix+'dict.txt', 'r', encoding = 'utf8')\
        as wordsfile: wds = wordsfile.readlines()
    dict = {}
    index = 0
    for i in wds:
        dict.update({i[:-1]: index})
        index += 1

    # loading train_data
    with open(savefprefix+'objs.pkl', 'rb') as f:
        data, labels, data_test, labels_test = pickle.load(f)



unk_words=0
for i in range(len(data_test)):
    for j in range(len(data_test[i])):
        if data_test[i][j] == 2:
            unk_words=unk_words+1

max_word_count = len(max(data, key=len))
dict_size = len(dict)

print("Learn Data size: ", len(data))
print("Dict size: ", dict_size)
print("Unknown words in test data: ", unk_words)
print("Max word count in string: ", max_word_count)

#data = keras.preprocessing.sequence.pad_sequences(data, value=0, padding='post', maxlen=max_word_count)  # , maxlen=256
data_test = keras.preprocessing.sequence.pad_sequences(data_test, value=0, padding='post', maxlen=max_word_count)  # , maxlen=256

#print (data)

# randomize learn data order
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

#print (data)

# ML Magic

model = keras.Sequential()
model.add(keras.layers.Embedding(dict_size, 100, input_length=max_word_count))

model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))

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

model.summary()

#model.get_config()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy', 'binary_crossentropy'])

print(len(data))

print("Start learning", datetime.datetime.now())

history = model.fit(data, labels, epochs=3, validation_split=0.2, verbose=1)

print("Start testing", datetime.datetime.now())
score = model.evaluate(data_test, labels_test)

print("Score [loose, accurance]: ", score)


model.save(savefprefix+'model.h5')
