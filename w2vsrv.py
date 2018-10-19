#!/usr/bin/python3

import my_lib
from flask import Flask, render_template, flash, request
from flask_restful import Resource, Api
from flask_wtf import Form
#import FlaskForm as Form
from wtforms import TextField, SubmitField
from wtforms import validators, ValidationError

from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('all.norm-sz500-w10-cb0-it3-min5.w2v', binary=True, unicode_errors='ignore', limit=5000) #, limit=500


class _Request(Resource):
    def get(self, X):
        return pred(X)

def pred(pos, neg):
    #X = my_lib.spec2space(X).split()
    #X = X.split()
    print ("Исходный запрос: ",pos, " ", neg)
    try:
        predict = word_vectors.most_similar(positive=pos, negative=neg, topn=1)[0]
    except KeyError:
        predict = "word not found"
    except ValueError:
        predict = "empty request"
    print("Output: ", predict)

    return predict


class ReqForm(Form):
    #name = TextField("Comment",[validators.Required("Please enter fields.")])
    submit = SubmitField("Send")


app = Flask(__name__)
app.secret_key = 'development key'
api = Api(app)
api.add_resource(_Request, '/request/<X>')

@app.route('/reqform', methods = ['GET', 'POST'])
def reqform():
   form = ReqForm()

   if request.method == 'POST':
         result = pred(request.form['pos'].split(), request.form['neg'].split())
         return render_template('forms-simil.html', form = form, result = result)
   elif request.method == 'GET':
     return render_template('forms-simil.html', form = form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5003')
