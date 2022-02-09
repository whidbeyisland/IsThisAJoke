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

class RandomizeJokes:
    df = None
    data_x = None
    data_y = None
    dls = None
    txts = None
    learn = None

    embeddings = None

    unconverted_jokes = []
    available_nouns = []
    available_verbs = []
    available_adjectives = []
    randomized_jokes = []

    def main(self):
        # getting all parts of speech that oculd be used to randomize jokes
        self.available_nouns = self.read_file('most-common-nouns-english.csv', 0)
        self.available_verbs = self.read_file('most-common-verbs-english.csv', 0)
        self.available_adjs = self.read_file('english-adjectives.txt', 0)

        # getting all jokes to be randomized
        self.unconverted_jokes = self.read_file('unconverted-jokes.txt', list(range(0, 2)))

        # randomizing jokes
        for i in range(0, len(self.unconverted_jokes)):
            self.randomized_jokes.append(self.randomize_joke(self.unconverted_jokes[i, 0]))
        print(self.randomized_jokes[0:3])

        # writing both unconverted and randomized jokes to new file
        jokes_df = pd.df(self.unconverted_jokes, columns = ['Joke', 'Category'])

        # coati: leaving off here, create new DF with 2 columns: "Joke" and then "Category"
        # which is always "Not Joke"
        jokes_df = jokes_df.concat(pd.df(self.randomized_jokes))

    
    def read_file(self, file_path, column_range):
        try:
            df_thisfile = pd.read_csv(os.path.join(path_resources, file_path))
            return df_thisfile.iloc[0:len(df_thisfile),column_range]
        except:
            print('Could not read file, or file was empty: ' + os.path.join(path, file_path))
            return []
    
    def randomize_joke(self, joke):
        randomized_joke = joke
        num_words_to_replace = 2
        joke_nopunct = re.sub(r'[^\w\s]', '', joke)
        joke_words = joke_nopunct.split()

        # get predicted parts of speech of all words in the joke
        pos_joke_predicted = nltk.tag.pos_tag(joke_words)

        # find the indices of the content words (i.e. nouns, verbs, adjectives) in the joke
        joke_content_word_indices = []
        for i in range(0, len(pos_joke_predicted)):
            try:
                if pos_joke_predicted[i][1][:2] in ('NN', 'VB', 'JJ'):
                    joke_content_word_indices.append(i)
            except:
                pass

        # find the content words in the joke
        joke_content_words = [[idx, word] for (idx, word) in enumerate(joke_words) if idx in joke_content_word_indices]

        # pick random words in the joke, and replace them with random content words of
        # the same part of speech
        random_words_to_replace = random.choices(joke_content_words, k=num_words_to_replace)
        for i in range(0, num_words_to_replace):
            word_to_replace = random_words_to_replace[i][1]
            word_to_replace_pos = pos_joke_predicted[random_words_to_replace[i][0]][1]
            if word_to_replace_pos[:2] == 'NN':
                randomized_joke = randomized_joke.replace(word_to_replace, 
                random.choice(self.available_nouns))
            elif word_to_replace_pos[:2] == 'VB':
                randomized_joke = randomized_joke.replace(word_to_replace,
                random.choice(self.available_verbs))
            else:
                randomized_joke = randomized_joke.replace(word_to_replace,
                random.choice(self.available_adjectives))

        return randomized_joke



r = RandomizeJokes()
r.main()