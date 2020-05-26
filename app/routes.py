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


@application.route('/')
@application.route('/index')
def index():
    return redirect('/results')


@application.route('/results')
def render_text():

    return render_template('results.html')

