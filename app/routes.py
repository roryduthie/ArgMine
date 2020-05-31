from flask import render_template, request, redirect, session, Markup
from . import application
import pandas as pd
from urllib.request import urlopen
import requests
import json
import urllib
import tempfile
import os
import uuid
import nltk
from nltk.tokenize import sent_tokenize
from joblib import load
from app.centrality import Centrality
from app.SentenceSimilarity import SentenceSimilarity
from fuzzywuzzy import fuzz
import spacy
from copy import deepcopy
import glob
import ast
from collections import defaultdict




@application.route('/')
@application.route('/index')
def index():
    return redirect('/home')
@application.route('/home')
def home_render():
    return render_template('home.html')

@application.route('/home', methods=['POST'])
def index_post():
    aif_mode = 'false'
    han_mode = 'false'
    ex_aif_mode = 'false'
    s_date = ''
    external_text = request.form['edata']
    source_text = request.form['sdata']
    aif_mode = request.form['aif_mode']
    ex_aif_mode = request.form['ex_aif_mode']
    han_mode = request.form['han_mode']
    s_date = request.form['date']
    session['s_date'] = s_date
    session['s_text'] = source_text
    session['e_text'] = external_text
    session['aif'] = aif_mode
    session['han'] = han_mode
    session['e_aif'] = ex_aif_mode

    return redirect('/results')


@application.route('/results')
def render_text():
    source_text = session.get('s_text', None)
    external_text = session.get('e_text', None)
    aif_mode = session.get('aif', None)
    han_mode = session.get('han', None)
    ex_aif_mode = session.get('e_aif', None)
    source_date = session.get('s_date', None)
    centra = Centrality()
    s_map_numbers = []
    ex_map_numbers = []
    ma_thresh = 0.85
    ra_thresh = 0.55
    s_l_i_nodes, ex_l_i_nodes, s_l_nodes, ex_l_nodes = []
    h_l_i_nodes, h_l_nodes = []

    if aif_mode == "true" and han_mode == "true" and ex_aif_mode == "false":
        # Source Map and Hansard

        sources = source_text.split(',')
        s_map_numbers = [int(i) for i in sources]
        central_nodes, s_l_i_nodes, s_l_nodes = centra.get_top_nodes_combined(s_map_numbers)

        source_topic_text = get_topic_text(central_nodes)
        txt_df = sent_to_df(source_topic_text)
        result = predict_topic(txt_df)
        hansard_fp = get_hansard_file_path(source_date, result, 'HansardDataAMF')
        hansard_map_num = check_hansard_path(hansard_fp)
        if hansard_map_num[0] == '':
            hansard_text = get_hansard_text(hansard_fp)
            hansard_text = hansard_text.decode("utf-8")
            ex_map_numbers = do_amf_calls(hansard_text, True)
            write_to_csv(ex_map_numbers, hansard_fp)
        else:
            ex_map_numbers = hansard_map_num

            ex_map_numbers = ast.literal_eval(ex_map_numbers)

        h_i_nodes, h_l_i_nodes, h_l_nodes = centra.get_all_nodes_combined(ex_map_numbers)

        relations = itc_matrix(central_nodes, h_i_nodes, ma_thresh, ra_thresh)
        if len(relations) > 0:
            #Build itc map
            map_id = build_itc_map(relations, s_l_i_nodes, h_l_i_nodes, s_l_nodes, h_l_nodes)



    elif aif_mode == "true" and han_mode == "false" and ex_aif_mode == "true":
        # Source Map and External Maps

        sources = source_text.split(',')
        s_map_numbers = [int(i) for i in sources]

        external = external_text.split(',')
        ex_map_numbers = [int(i) for i in external]

        central_nodes, s_l_i_nodes, s_l_nodes = centra.get_top_nodes_combined(s_map_numbers)
        ex_i_nodes, ex_l_i_nodes, ex_l_nodes = centra.get_all_nodes_combined(ex_map_numbers)

        relations = itc_matrix(central_nodes, ex_i_nodes, ma_thresh, ra_thresh)
        if len(relations) > 0:
            #Build itc map
            map_id = build_itc_map(relations, s_l_i_nodes, ex_l_i_nodes, s_l_nodes, ex_l_nodes)


    elif aif_mode == "false" and han_mode == "true" and ex_aif_mode == "false":
        # Source Text and Hansard
        s_map_numbers = do_amf_calls(source_text, False)
        central_nodes, s_l_i_nodes, s_l_nodes = centra.get_top_nodes_combined(s_map_numbers)

        source_topic_text = get_topic_text(central_nodes)
        txt_df = sent_to_df(source_topic_text)
        result = predict_topic(txt_df)
        hansard_fp = get_hansard_file_path(source_date, result, 'HansardDataAMF')
        hansard_map_num = check_hansard_path(hansard_fp)

        if hansard_map_num[0] == '':
            hansard_text = get_hansard_text(hansard_fp)
            hansard_text = hansard_text.decode("utf-8")

            ex_map_numbers = do_amf_calls(hansard_text, True)
            write_to_csv(h_map_numbers, hansard_fp)
        else:

            ex_map_numbers = hansard_map_num

            ex_map_numbers = ast.literal_eval(ex_map_numbers)

        h_i_nodes, h_l_i_nodes, h_l_nodes = centra.get_all_nodes_combined(ex_map_numbers)

        #print(central_nodes, h_i_nodes)

        relations = itc_matrix(central_nodes, h_i_nodes, ma_thresh, ra_thresh)
        if len(relations) > 0:
            #Build itc map
            map_id = build_itc_map(relations, s_l_i_nodes, h_l_i_nodes, s_l_nodes, h_l_nodes)





    elif aif_mode == "false" and han_mode == "false" and ex_aif_mode == "false":
        # Source Text and External Text

        s_map_numbers = do_amf_calls(source_text, False)
        central_nodes, s_l_i_nodes, s_l_nodes = centra.get_top_nodes_combined(s_map_numbers)

        ex_map_numbers = do_amf_calls(external_text, False)
        ex_i_nodes, ex_l_i_nodes, ex_l_nodes = centra.get_all_nodes_combined(ex_map_numbers)

        relations = itc_matrix(central_nodes, ex_i_nodes, ma_thresh, ra_thresh)
        if len(relations) > 0:
            #Build itc map
            map_id = build_itc_map(relations, s_l_i_nodes, ex_l_i_nodes, s_l_nodes, ex_l_nodes)

    elif aif_mode == "true" and han_mode == "false" and ex_aif_mode == "false":
        # Source Text and External Text

        sources = source_text.split(',')
        s_map_numbers = [int(i) for i in sources]

        central_nodes, s_l_i_nodes, s_l_nodes = centra.get_top_nodes_combined(s_map_numbers)

        ex_map_numbers = do_amf_calls(external_text, False)
        ex_i_nodes, ex_l_i_nodes, ex_l_nodes = centra.get_all_nodes_combined(ex_map_numbers)

        relations = itc_matrix(central_nodes, ex_i_nodes, ma_thresh, ra_thresh)
        if len(relations) > 0:
            #Build itc map
            map_id = build_itc_map(relations, s_l_i_nodes, ex_l_i_nodes, s_l_nodes, ex_l_nodes)


    elif aif_mode == "false" and han_mode == "false" and ex_aif_mode == "true":
        # Source Text and External Map

        s_map_numbers = do_amf_calls(source_text, False)
        central_nodes, s_l_i_nodes, s_l_nodes = centra.get_top_nodes_combined(s_map_numbers)

        external = external_text.split(',')
        ex_map_numbers = [int(i) for i in external]
        ex_i_nodes, ex_l_i_nodes, ex_l_nodes = centra.get_all_nodes_combined(ex_map_numbers)


        relations = itc_matrix(central_nodes, ex_i_nodes, ma_thresh, ra_thresh)
        if len(relations) > 0:
            #Build itc map
            map_id = build_itc_map(relations, s_l_i_nodes, ex_l_i_nodes, s_l_nodes, ex_l_nodes)



    new_map_numbers = get_new_map_nums(s_map_numbers)
    if len(relations) > 0:
        itc_map_number = get_new_itc_map(map_id)
        itc_map_list = [itc_map_number]
        itc_map_view_list = create_argview_links(itc_map_list)
        itc_number = str(itc_map_view_list[0])
        itc_relations = [(rels[1], rels[3], rels[4] ) for rels in relations]
    else:
        itc_number = 'No ITC relations found'
        itc_relations = ['No ITC relations found']

    source_map_numbers_links = create_argview_links(new_map_numbers)

    ex_map_number_links = create_argview_links(ex_map_numbers)

    return render_template('results.html', source_text=source_text, source_maps_links = source_map_numbers_links, ex_map_links = ex_map_number_links, itc_number=itc_number, itc_relations=itc_relations)


def create_argview_links(map_numbers):
    link_list = []
    for nodeset in map_numbers:
        link = 'http://www.aifdb.org/argview/' + str(nodeset)

        link_list.append(link)

    return link_list
def get_new_itc_map(nodeset_id):
    new_map_id = get_arg_schemes(nodeset_id)

    if new_map_id == '':
        return nodeset_id
    else:
        return new_map_id


def get_new_map_nums(s_map_numbers):
    new_maps = []
    for nodeset in s_map_numbers:
        new_map_id = get_arg_schemes(nodeset)
        if new_map_id == '':
            new_maps.append(nodeset)
        else:
            new_maps.append(new_map_id)
    return new_maps

def sent_to_df(txt):
    txt_pred = {'text': [txt]}
    df = pd.DataFrame(data=txt_pred)
    return df

def predict_topic(df):
    model_path = 'static/model/final_hansard_topic_model_seed.joblib'
    with application.open_resource(model_path) as load_m:
        loaded_m = load(load_m)
    pred = loaded_m.predict(df['text'])
    result = pred[0]
    return result

def get_hansard_file_path(input_date, topic, han_directory):
    files_list = []
    for subdir, dirs, files in os.walk(os.path.join(application.static_folder, han_directory)):
        for file_name in files:
            if 'txt' in file_name:
                full_path = subdir + '/' + file_name
                date = subdir.split(os.path.sep)[1]
                date = date.replace("-","")
                file = str(file_name).lower()
                file_tup = (full_path, date, file)
                files_list.append(file_tup)

    sorted_files = sorted(files_list, key=lambda tup: tup[1], reverse=True)
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

    with application.open_resource(file_path) as text_file:
        text = text_file.read()
    #text = text.encode('utf-8')
    return text

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

def call_amf(chunks, test_flag):
    map_nums = []
    url_turn = 'http://turninator.arg.tech/turninator'
    url_props = 'http://propositionalizer.arg.tech/propositionalizer'
    url_aif = 'http://www.aifdb.org/json/'
    #URL for hosting outwith ARG-Tech Infrastrucutre
    url_inf = 'http://dam-02.arg.tech/dam-02'
    #url_inf = 'http://cicero.arg.tech:8092/dam-02'
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

        map_response = aif_upload(url_aif, aif_json)
        map_data = json.loads(map_response)
        map_id = map_data['nodeSetID']
        map_nums.append(map_id)
        #print('Got nodeset ' + str(map_id) )
    #if test_flag:
    #    map_nums = [10670, 10671]
    #else:
    #    map_nums = [10672]
    return map_nums


def get_similarity(sent1, sent2):
    sent_sim = SentenceSimilarity()
    similarity = sent_sim.main(sent1, sent2)
    return similarity

def get_fuzzy_similarity(sent1, sent2):
    sim = fuzz.token_set_ratio(sent1,sent2)
    if sim == 0:
        return 0
    else:
        return sim/100
def get_alternate_wn_similarity(sent1, sent2):
    sent_sim = SentenceSimilarity()
    similarity = sent_sim.symmetric_sentence_similarity(sent1, sent2)
    return similarity

def check_sim_thresholds(similarity, premise, conclusion, ma_thresh, ra_thresh):
    negation_list = ['no', 'not', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'scarcely', 'barely', 'doesnt', 'isnt', 'wasnt', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 'cant', 'dont']
    if similarity > ma_thresh:
        return 'MA'
    if similarity > ra_thresh:
        negation_flag = False
        for neg in negation_list:
            premise = premise.lower()
            premise = premise.replace("'","")
            if neg in premise:
                negation_flag = True

        node_t = conclusion.lower()
        if 'xxx' in node_t or '/' in node_t:
            return ''
        node_p = premise.lower()
        if 'xxx' in node_p or '/' in node_p:
            return ''

        if negation_flag:
            return 'CA'
        else:
            return 'RA'
    else:
        return ''
def get_parsed_text(txt):
    pos_tok_list = ['SYM', 'DET', 'ADP', 'PUNCT', 'AUX', 'PART', 'SCONJ', 'X', 'CONJ']
    txt = process_text(txt)
    nlp = spacy.load("en")
    orig_doc = nlp(txt)
    sent = []
    sent_remove = []
    for token in orig_doc:
        pos_tok = token.pos_
        if 'PROPN' in pos_tok:
            sent.append('it')
        else:
            sent.append(token.text)
        if pos_tok in pos_tok_list:
            sent_remove.append(token.text)

    new_txt = ' '.join(sent)
    words = []
    doc = nlp(new_txt)
    for chunk in doc.noun_chunks:
        if 'nsubj' in chunk.root.dep_ or 'dobj' in chunk.root.dep_ or 'pobj' in chunk.root.dep_ or 'nmod' in chunk.root.dep_ or 'obl' in chunk.root.dep_:
            words.append(chunk.text)
            words.append(chunk.root.head.text)
    words = [i.strip() for i in words]
    res = list(set(words)^set(sent_remove))
    a = set(res)
    new_res = list(a)
    parsed_text = ' '.join(new_res)
    parsed_text = parsed_text.replace(".", "")
    parsed_text = parsed_text.replace(",", "")
    return parsed_text

def process_text(txt):
    txt = txt.lower()

    if 'but' in txt:
        txt = txt.split('but')[0]
    if 'because' in txt:
        txt = txt.split('because')[0]
    # and? .? ,? because?
    return txt

def get_topic_text(central_nodes_tup_list):
    overall_text = ''
    for tup in central_nodes_tup_list:
        txt = tup[1]
        parsed_text = get_parsed_text(txt)
        overall_text = overall_text + parsed_text + ' '
    return overall_text

def do_amf_calls(s_txt, test_flag):
    s_txt_lst = text_to_lines(s_txt)
    removetable = str.maketrans('', '', '@#%-;')
    out_list = [s.translate(removetable) for s in s_txt_lst]
    chunks = chunk_words(out_list)
    s_map_numbers = call_amf(chunks, test_flag)
    return s_map_numbers

def itc_matrix(source_nodes, other_nodes, ma_thresh, ra_thresh):
    relations = []
    for node in source_nodes:
        node_id = node[0]
        node_text = node[1]


        for ex_nodes in other_nodes:
            ex_id = ex_nodes[0]
            ex_text = ex_nodes[1]

            #node_parsed_text = get_parsed_text(node_text)
            #ex_parsed_text = get_parsed_text(ex_text)
            if ex_text == '' or node_text == '':
                continue
            else:
                similarity = get_similarity(node_text, ex_text)
                if similarity > 1 or similarity == 0:
                    #similarity = get_fuzzy_similarity(node_parsed_text, ex_parsed_text)
                    similarity = get_alternate_wn_similarity(node_text, ex_text)


                relation = check_sim_thresholds(similarity, ex_text, node_text, ma_thresh, ra_thresh)
                if relation == '':
                    continue
                else:
                    relation_tup = (node_id, node_text, ex_id, ex_text, relation)
                    relations.append(relation_tup)

    return relations


def check_hansard_path(hansard_fp):
    file_name = 'hansard_maps.csv'

    files_present = glob.glob(file_name)

    if not files_present:
        return ['']
    else:
        hansard_df = pd.read_csv(file_name)

        sel_df = hansard_df[hansard_df['filename'] == hansard_fp]
        if len(sel_df) < 1:
            return ['']
        else:
            sel_df.reset_index(inplace=True)
            return sel_df['map_id'][0]
def write_to_csv(map_numbers, hansard_fp):
    file_name = 'hansard_maps.csv'

    files_present = glob.glob(file_name)

    if not files_present:
        #create df and write
        df = pd.DataFrame({'filename': hansard_fp, 'map_id': [map_numbers]})
        df.to_csv(file_name)
    else:
        df = pd.DataFrame({'filename': hansard_fp, 'map_id': [map_numbers]})
        df.to_csv(file_name, mode='a', header=False)

def get_l_node_text(i_node_id, lnode_inode_list, l_node_list):
    for rel_tup in lnode_inode_list:
        lnode_id = rel_tup[0]
        inode_id = rel_tup[1]

        if i_node_id == inode_id:
            for tups in l_node_list:
                l_id = tups[0]
                if l_id == lnode_id:
                    ltext = tups[1]
                    return l_id, ltext
def build_itc_json(relations, aif_flags):
    node_list = []
    edge_list = []
    loc_list = []

    json_aif_dict = defaultdict(list)

    for i,rel in enumerate(relations):
        node_id = i + 1
        if not aif_flags:
            source_i_n = {"nodeID": "si" + str(node_id), "text": rel[0], "type": "I"}
            source_l_n = {"nodeID": "sl" + str(node_id), "text": rel[1], "type": "L"}
            ex_i_n = {"nodeID": "ei" + str(node_id), "text": rel[2], "type": "I"}
            ex_l_n = {"nodeID": "el" + str(node_id), "text": rel[3], "type": "L"}
            s_n = {"nodeID": "s" + str(node_id), "text": rel[5], "type": rel[4]}
            ya_n = {"nodeID": "ya" + str(node_id), "text": rel[6], "type": "YA"}
            ta_n = {"nodeID": "ta" + str(node_id), "text": "Default Transition", "type": "TA"}

            node_list.append(source_i_n)
            node_list.append(source_l_n)
            node_list.append(ex_i_n)
            node_list.append(ex_l_n)
            node_list.append(s_n)
            node_list.append(ya_n)
            node_list.append(ta_n)


            edge_1 = {"edgeID":"e" + str(node_id), "fromID":"el" + str(node_id), "toID":"ta" + str(node_id)}
            edge_2 = {"edgeID":"ee" + str(node_id), "fromID":"ta" + str(node_id), "toID":"sl" + str(node_id)}
            edge_3 = {"edgeID":"eee" + str(node_id), "fromID":"ta" + str(node_id), "toID":"ya" + str(node_id)}
            edge_4 = {"edgeID":"eeee" + str(node_id), "fromID":"ya" + str(node_id), "toID":"s" + str(node_id)}
            edge_5 = {"edgeID":"eeeee" + str(node_id), "fromID":"ei" + str(node_id), "toID":"s" + str(node_id)}
            edge_6 = {"edgeID":"eeeeee" + str(node_id), "fromID":"s" + str(node_id), "toID":"si" + str(node_id)}

            edge_list.append(edge_1)
            edge_list.append(edge_2)
            edge_list.append(edge_3)
            edge_list.append(edge_4)
            edge_list.append(edge_5)
            edge_list.append(edge_6)
        else:
            source_i_n = {"nodeID": "si" + str(node_id), "text": rel[0], "type": "I"}
            ex_i_n = {"nodeID": "ei" + str(node_id), "text": rel[2], "type": "I"}
            s_n = {"nodeID": "s" + str(node_id), "text": rel[5], "type": rel[4]}

            node_list.append(source_i_n)
            node_list.append(ex_i_n)
            node_list.append(s_n)


            edge_5 = {"edgeID":"eeeee" + str(node_id), "fromID":"ei" + str(node_id), "toID":"s" + str(node_id)}
            edge_6 = {"edgeID":"eeeeee" + str(node_id), "fromID":"s" + str(node_id), "toID":"si" + str(node_id)}

            edge_list.append(edge_5)
            edge_list.append(edge_6)




    json_aif_dict["nodes"].extend(node_list)
    json_aif_dict["edges"].extend(edge_list)
    json_aif_dict["locutions"].extend(loc_list)


    aif_json = json.dumps(json_aif_dict)
    return aif_json

def build_itc_map(relations, source_l_i_list, ex_l_i_list, source_l_list, ex_l_list):
    map_rels = []
    aif_flags = False
    if not source_l_i_list or not ex_l_i_list or not source_l_list or not ex_l_list:
        aif_flags = True
    for rel_tups in relations:
        s_i_id = rel_tups[0]
        s_i_text = rel_tups[1]
        ex_i_id = rel_tups[2]
        ex_i_text = rel_tups[3]
        rel = rel_tups[4]
        ya = ''
        scheme_text = ''
        # call get_l_node_text for each i_id to get L

        if not aif_flags:
            source_l = get_l_node_text(s_i_id, source_l_i_list, source_l_list)
            ex_l = get_l_node_text(ex_i_id, ex_l_i_list, ex_l_list)


            s_l_id = source_l[0]
            s_l_text = source_l[1]

            ex_l_id = ex_l[0]
            ex_l_text = ex_l[1]


        if rel == 'MA':
            ya = 'Restating'
            scheme_text = 'Default Rephrase'
        elif rel == 'RA':
            ya = 'Arguing'
            scheme_text = 'Default Inference'
        elif rel == 'CA':
            ya = 'Disagreeing'
            scheme_text = 'Default Conflict'

        if not aif_flags:

            rel_tuple = (s_i_text, s_l_text, ex_i_text, ex_l_text, rel, scheme_text, ya)
            map_rels.append(rel_tuple)
        else:
            rel_tuple = (s_i_text, ex_i_text, rel, scheme_text)
            map_rels.append(rel_tuple)

    aif_json = build_itc_json(map_rels, aif_flags)
    url_aif = 'http://www.aifdb.org/json/'
    map_response = aif_upload(url_aif, aif_json)
    map_data = json.loads(map_response)
    map_id = map_data['nodeSetID']

    return map_id


def get_arg_schemes(nodeset):
    cent = Centrality()

    j_url = cent.create_json_url(str(nodeset),True)
    graph = cent.get_graph_url(j_url)
    json_data = get_json_string(j_url)

    ras = cent.get_ras(graph)
    ras_i_list = cent.get_ra_i_nodes(graph, ras)

    ra_changes = []
    for ns in ras_i_list:
        ra_id = ns[0]
        s_id = ns[1]
        e_id = ns[2]

        schemes = identifyScheme(e_id, s_id)

        if len(schemes) < 1:
            continue
        else:
            ra_tup = (ra_id, schemes[0])
            ra_changes.append(ra_tup)
            #get json string and replace text at ID then upload

    print(ra_changes)
    if len(ra_changes) < 1:
        return ''
    else:
        n_json_data = replace_node(json_data, ra_changes)
        url_aif = 'http://www.aifdb.org/json/'
        jsn_data = json.dumps(n_json_data)
        map_response = aif_upload(url_aif, jsn_data)
        map_data = json.loads(map_response)
        fin_map_id = map_data['nodeSetID']


    return fin_map_id


def get_json_string(node_path):
    try:
        jsn_string = requests.get(node_path).text
        strng_ind = jsn_string.index('{')
        n_string = jsn_string[strng_ind:]
        dta = json.loads(n_string)
    except(IOError):
        print('File was not found:')
        print(node_path)

    return dta
def replace_node(json_data, node_list):
    #json_data_dict = json.loads(json_data)

    for ns in node_list:
        n_id = ns[0]
        new_text = ns[1]
        for nodes in json_data['nodes']:
            json_n_id = nodes['nodeID']
            if str(n_id) == str(json_n_id):
                nodes['text'] = new_text

    return json_data

def identifyScheme(premise, conclusion):
    identifiedSchemes = []

    if (("similar" in premise or "generally" in premise) and ("be" in conclusion or "to be" in conclusion)):
        identifiedSchemes.append("Analogy")

    elif ("generally" in premise or "occur" in premise) or ("occur" in conclusion) :
        identifiedSchemes.append("CauseToEffect")

    elif("goal" in premise or "action" in premise) or ("ought" in conclusion or "perform" in conclusion) :
        identifiedSchemes.append("PracticalReasoning")

    elif(("all" in premise or "if" in premise) and ("be" in conclusion or "to be" in conclusion)) :
        identifiedSchemes.append("VerbalClassification")

    elif((("expert" in premise or "experience" in premise or "skill" in premise) and "said" in premise) and ("be" in conclusion or "to be" in conclusion)) :
        identifiedSchemes.append("ExpertOpinion")

    elif(("occur" in premise or "happen" in premise) and ("should" in conclusion or "must" in conclusion)) :
        identifiedSchemes.append("PositiveConsequences")

    return identifiedSchemes

