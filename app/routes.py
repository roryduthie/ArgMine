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
    hansard_fp = get_hansard_file_path('2020-05-24', result, 'HansardDataAMF')
    print(hansard_fp)
    get_hansard_text(hansard_fp)
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

def get_hansard_file_path(input_date, topic, han_directory):
    files_list = []
    for subdir, dirs, files in os.walk(os.path.join(app.static_folder, han_directory)):
        for file_name in files:
            if 'txt' in file_name:
                full_path = subdir + '/' + file_name
                date = subdir.split(os.path.sep)[1]
                date = date.replace("-","")
                file = str(file_name).lower()
                file_tup = (full_path, date, file)
                files_list.append(file_tup)

    sorted_files = sorted(files_list, key=lambda tup: tup[1], reverse=True)
    print(sorted_files)
    input_date = input_date.replace('-', '')
    selected_file = ''
    for tup in sorted_files:
        date = tup[1]
        file_name = tup[2]
        file_path = tup[0]
        if input_date < date:
            continue
        else:
            if topic in file_name:
                selected_file = file_path

    if selected_file == '':
        for tup in sorted_files:
            date = tup[1]
            file_name = tup[2]
            file_path = tup[0]
            if topic in file_name:
                selected_file = file_path

    if not selected_file == '':
        selected_file = selected_file.split('/app/')[1]
    return selected_file

def get_hansard_text(file_path):

    with app.open_resource(file_path) as text_file:
        text = text_file.read()

    print(text)
    return ''
