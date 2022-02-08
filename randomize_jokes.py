import base64
import io
import json
import os
import gdown
#import fastbook
#fastbook.setup_book()
import fastai
import pandas as pd
import requests
import torchtext
import nltk
import snscrape.modules.twitter as sntwitter
from copy import deepcopy

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
#from fastbook import *
from torchtext.data import get_tokenizer
from fastai.text.all import *

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# from nltk.corpus import wordnet
# from nltk import FreqDist
from string import punctuation
import mmap

import pathlib
posixpath_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from sklearn.decomposition import PCA



path = os.getcwd()
path_resources = os.path.join(path, 'resources')
available_nouns = []
available_verbs = []
available_adjectives = []

class RandomizeJokes:
    df = None
    data_x = None
    data_y = None
    dls = None
    txts = None
    learn = None

    embeddings = None

    available_nouns = []
    available_verbs = []
    available_adjectives = []



    def main(self):
        # getting all nouns
        available_nouns = self.read_file('most-common-nouns-english.csv')
        available_verbs = self.read_file('most-common-verbs-english.csv')
        available_adjs = self.read_file('english-adjectives.txt')

        '''
        print(available_nouns[0:20])
        print(len(available_nouns))
        print(available_verbs[0:20])
        print(len(available_verbs))
        print(available_adjs[0:20])
        print(len(available_adjs))
        '''
    
    def read_file(self, file_path):
        try:
            df_thisfile = pd.read_csv(os.path.join(path_resources, file_path))
            return df_thisfile.iloc[0:len(df_thisfile),0]
        except:
            print('Could not read file: ' + os.path.join(path, file_path))
            return []



r = RandomizeJokes()
r.main()