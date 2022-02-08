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

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk import FreqDist
from string import punctuation
import mmap

import pathlib
posixpath_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from sklearn.decomposition import PCA



path = os.getcwd()

class RandomizeJokes:
    df = None
    data_x = None
    data_y = None
    dls = None
    txts = None
    learn = None

    embeddings = None

r = RandomizeJokes()
r.main()