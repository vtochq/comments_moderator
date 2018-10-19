#!/usr/bin/python3

import tensorflow
import numpy as np
import pickle
import my_lib
from flask import Flask, render_template, flash, request
from flask_restful import Resource, Api
from flask_wtf import Form
#import FlaskForm as Form
from wtforms import TextField, SubmitField
from wtforms import validators, ValidationError

#savefprefix="w2v_wTitle_tacc74_"
savefprefix="w2v_tacc76_"


#with open(savefprefix+'tokenizerwc.pkl', 'rb') as f:
#    tokenizer, max_word_count = pickle.load(f)


with open(savefprefix+'objs.pkl', 'rb') as f:
    data, labels, data_test, labels_test, word_dict, embedding_matrix, vocabulary_size, tokenizer, max_word_count = pickle.load(f)


model = tensorflow.keras.models.load_model(savefprefix+'model.h5')

# fix for keras work inside class (i don't know...)
model.predict(tensorflow.keras.preprocessing.sequence.pad_sequences(np.zeros((1,1)), value=0, padding='post', maxlen=max_word_count), verbose=0)



class _Request(Resource):
    def get(self, X):
        return pred(X)

def pred(X):
    X = my_lib.spec2space(X)
    print ("Исходный запрос: ",X)
    X = tokenizer.texts_to_sequences([X])
    X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, value=0, padding='post', maxlen=max_word_count)
    print ("Tokenized запрос: ",X)
    predict = float(model.predict(X, batch_size=512, verbose=0)[0][0])
    print("Output: ", predict)

    if predict > 0.6:
        result = "ACCEPTED"
    elif predict < 0.4:
        result = "REJECTED"
    else: result = "Не однозначненько как-то"
    return result


class ReqForm(Form):
    name = TextField("Comment",[validators.Required("Please enter comment.")])
    submit = SubmitField("Send")


app = Flask(__name__)
app.secret_key = 'development key'
#api = Api(app)
#api.add_resource(_Request, '/request/<X>')

@app.route('/reqform', methods = ['GET', 'POST'])
def reqform():
   form = ReqForm()

   if request.method == 'POST':
         result = pred(request.form['title']+" "+request.form['comment'])
         return render_template('comments.html', form = form, result = result)
   elif request.method == 'GET':
     return render_template('comments.html', form = form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5002')
