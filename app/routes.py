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
from app.centrality import Centrality


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
    external_text = session.get('e_text', None)
    aif_mode = session.get('aif', None)
    han_mode = session.get('han', None)


    centra = Centrality()
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
    model_path = 'static/model/final_hansard_topic_model.joblib'
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
    text = text.encode('utf-8')
    print(text)
    return ''

def text_to_lines(textData):
    fin_list = []
    lines_speakers = textData.splitlines(keepends=True)
    for line in lines_speakers:
        sentence_list = sent_tokenize(line)
        if len(sentence_list) > 0 and len(sentence_list) < 2:
            sent = sentence_list[0]
            if len(sent) > 0:
                fin_list.append(line)
        elif len(sentence_list) > 0:
            fin_list.append(line)

    return fin_list

def chunk_words(text_list):
    word_counter = 0
    chunks = []
    temp_list = []
    word_count_flag = False
    for line in text_list:
        words = line.split()
        word_counter += len(words)
        if word_counter > 700:
            word_counter = len(words)
            chunks.append(deepcopy(temp_list))
            temp_list = []
            word_count_flag = True
        temp_list.append(line)
    if word_counter < 700:
        chunks.append(deepcopy(temp_list))
    return chunks

def aif_upload(url, aif_data):
    aif_data = str(aif_data)
    filename = uuid.uuid4().hex
    filename = filename + '.json'
    with open(filename,"w") as fo:
        fo.write(aif_data)
    files = {
        'file': (filename, open(filename, 'rb')),
    }
    #get corpus ID

    aif_response = requests.post(url, files=files, auth=('test', 'pass'))
    #change this to pass the response back as text rather than as the full JSON output, this way we either pass back that a corpus was added to or a map uplaoded with map ID. Might be worth passing MAPID and Corpus name back in that situation.

    os.remove(filename)
    return aif_response.text

def post_turns(url,text_str):
    text_str = str(text_str)
    filename = uuid.uuid4().hex
    filename = filename + '.txt'
    with open(filename,"w") as fo:
        fo.write(text_str)
    files = {
        'file': (filename, open(filename, 'rb')),
    }
    #get corpus ID
    response = requests.post(url, files=files)
    os.remove(filename)
    return response

def post(url,text_str):
    #print(text_str)
    #text_str = str(text_str)
    filename = uuid.uuid4().hex
    filename = filename + '.txt'
    with open(filename,"w") as fo:
        fo.write(text_str)
    files = {
        'file': (filename, open(filename, 'rb')),
    }
    #get corpus ID

    response = requests.post(url, files=files)
    os.remove(filename)
    return response

def call_amf(chunks):
    map_nums = []
    url_turn = 'http://turninator.arg.tech/turninator'
    url_props = 'http://propositionalizer.arg.tech/propositionalizer'
    url_aif = 'http://www.aifdb.org/json/'
    #URL for hosting outwith ARG-Tech Infrastrucutre
    #url_inf = 'http://dam-02.arg.tech/dam-02'
    url_inf = 'http://cicero.arg.tech:8092/dam-02'
    for i, chunk in enumerate(chunks):
        #print('######################################################')
        #print('Processing chunk ' + str(i) + ' of ' + str(len(chunks)))
        out_str = " ".join(chunk)
        out_str = out_str.replace('’', '')
        out_str = out_str.replace('‘', '')
        out_str = out_str.replace(',', '')
        out_str = out_str.replace('–', '')
        out_str = out_str.replace(')', '')
        out_str = out_str.replace('(', '')
        out_str = out_str.replace("/", '')
    #out_str = repr(out_str)
        word_count = len(out_str.split())
        #print(word_count)
    #print(out_str)
        #print('Getting Turns from AMF')
        prop_text_resp = post_turns(url_turn, out_str)
        prop_text = prop_text_resp.text
    #print(prop_text)
    #print(prop_text_resp)
        if prop_text == '':
        #print(prop_text)
            print('EMPTY return TURNS')
    #print(prop_text)
        #print('Getting Propositions from AMF')
        inf_text_resp = post(url_props, prop_text)
        inf_text = inf_text_resp.text
    #print(inf_text)
        if inf_text == '':
            print('EMPTY return PROPS')
        #break
        #print('Getting Inference relations from AMF')
        aif_json_resp = post(url_inf, inf_text)
        aif_json = aif_json_resp.text
    #print(aif_json)
        if aif_json == '':
            print('EMPTY return INF')
        #break
        #print('Uploading AIF to AIFdb')
    #print(aif_json)


    #Commented out so as to not ruin AIFdb

        #map_response = aif_upload(url_aif, aif_json)
        #map_data = json.loads(map_response)
        #map_id = map_data['nodeSetID']
        #map_nums.append(map_id)
        #print('Got nodeset ' + str(map_id) )
    map_nums = [10672, 10670]
    return map_nums



