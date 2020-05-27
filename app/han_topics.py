import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def get_par_stopwords(file_path):
    with app.open_resource(file_path) as stop_file:
        st = stop_file.read()
    st = st.strip()
    st = st.replace('\n', '')
    st = st.replace(' ', '')
    par_stops = st.split(',')
    return par_stops



def combine_sw_lists(par_stops):
    SW = ['gentleman', 'lady', 'he', 'she', 'member', 'government', 'minister', 'chancellor', 'friend', 'secretary', 'opposition', 'labour', 'conservative', 'tory', 'mr', 'speaker', 'cabinet']
    SW1 = ["ten","nine","eight","seven","six","five","four","three","two","one","tabling","table","shuffle","session","order","point","chair","panel","procedure","motion","manifesto","legislation","lobbying","lobby","introduce","introduction","vote","deputy","devolution","clause","committee","code","chamber","house","motion","budget","bill","bar","debate","allowance","so","fisherman","workers","farmers","farm","factory","plant","mill","seem","seeming", "seemed","please","pleased", "pleasent","see","sees","third","second","first","also","accepts","a","about","above","alone", "grateful","refers","refer","referred","referring","it","is","poll","too","show","report","reports","reported","ExPect", "ministerial", "represent", "represents", "therefore", "thus", "met", "constituencies", "constituent", "constituency", "like", "told", "now", "learned", "hon." , "ask", "us", "act", "can", "thank", "but", "hope", "wish", "because", "which", "where", "why", "when", "member", "lady", "friend", "proposal", "think", "confirm", "answer", "you", "does", "also", "give", "he", "she", "himself", "herself", "was", "sure", "said", "have", "do", "and", "recall", "that", "tell", "view", "his", "her", "accept", "aware", "agree", "agrees", "disagree", "disagrees", "question", "has", "may", "know", "knows", "not", "my", "no", "honourable", "hon" ]

    stop_words = text.ENGLISH_STOP_WORDS.union(SW)
    stop_words = stop_words.union(SW1)
    stop_words = stop_words.union(par_stops)

    return stop_words

