from flair.data import Sentence
from flair.models import SequenceTagger
import csv
import os
import pickle
import json


tagger = SequenceTagger.load("flair/ner-english-large")

pickle_fp = "YOUR_RESULT_PATH"
f_full = open('full_keywords.txt', "w")
keywords={}

for root, dirs, files in os.walk(pickle_fp):
    files.sort()
    for i, file in enumerate(files):
        with open(os.path.join(pickle_fp, file), 'rb') as f:
            or_text, rc_text, tensor = pickle.load(f)
        keyword = []
        sentence = Sentence(or_text)
        tagger.predict(sentence)
        f_full.write("full sentence:{}".format(or_text))
        for entity in sentence.get_spans('ner'):
            print(entity)
            f_full.write("\nsentence {}, keyword ".format(i))
            print(entity.tokens)
            f_full.write(str(entity))
            print(entity.text)
            keyword.append(entity.text)
        keywords[or_text] = keyword

with open("keyword.json", 'w') as file:
    file.write(json.dumps(keywords))