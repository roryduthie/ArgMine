from flask import render_template, request, redirect, session, Markup
from . import app
import pandas as pd
from urllib.request import urlopen
import requests
import json
import urllib
import tempfile
import os
import uuid
from joblib import load


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index_post():
    aif_mode = 'false'
    han_mode = 'false'
    external_text = request.form['edata']
    source_text = request.form['sdata']
    aif_mode = request.form['aif_mode']
    session['s_text'] = source_text
    session['e_text'] = external_text
    session['aif'] = aif_mode
    session['han'] = han_mode

    return redirect('/results')


@app.route('/results')
def render_text():
    source_text = session.get('s_text', None)
    print(source_text)
    txt_df = sent_to_df(source_text)
    result = predict_topic(txt_df)
    print(result)
    return render_template('results.html', source_text=source_text)

def sent_to_df(txt):
    txt_pred = {'text': [txt]}
    df = pd.DataFrame(data=txt_pred)
    return df

def predict_topic(df):
    model_path = 'static/model/20192020topicmodel.joblib'
    with app.open_resource(model_path) as load_m:
        loaded_m = load(load_m)
    pred = loaded_m.predict(df['text'])
    result = pred[0]
    return result

