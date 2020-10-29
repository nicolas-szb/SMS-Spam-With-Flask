import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

import en_core_web_sm
from spacy.lang.fr.stop_words import STOP_WORDS

import re

from flask import Flask, request, render_template
import requests
import json

app = Flask(__name__)

nlp = en_core_web_sm.load()


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=['POST'])
def result():
    # Création d'un tf dataset à partir de pandas
    def create_tf_dataset(texte):
        l = [{"texte": texte,
         "vide": 0}]
        df = pd.DataFrame(l)
        tf_ds = tf.data.Dataset.from_tensor_slices((df.iloc[:,0].values, df.iloc[:,1].values))

        return tf_ds

    # https://docs.python.org/fr/3/howto/regex.html#regex-howto

    def create_token_doc(data):

        #corpus = ' '.join(data)
        corpus = data.strip().encode("utf-8").decode("utf-8")

        corpus = re.sub(r"\s+", " ", corpus) # eliminer les espaces blancs en double

        #filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        filters = '\t\n'

        corpus = tf.keras.preprocessing.text.text_to_word_sequence(corpus,
                                                          filters=filters,
                                                          lower=True,
                                                          split=' ')

        corpus = ' '.join(corpus)

        # Besoin de verifier la taille sinon nlp.max_length est bloque a 1000000
        if len(corpus) >= 1000000:
            nlp.max_length = len(corpus)+1 # +1 dans le doute
            doc = nlp(corpus)
        else:
            doc = nlp(corpus)


        return doc

    def tokens_cleaning(doc):

        # Create list of word tokens
        token_list = []
        for token in doc:
            token_list.append(token.text)

        # Create list of word tokens after removing stopwords
        stop_word_removed = []

        for word in token_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                stop_word_removed.append(word)

        # Create list of word token after removing punctuation
        punctuation_removed = []

        for word in stop_word_removed:
            lexeme = nlp.vocab[word]
            if lexeme.is_punct == False:
                punctuation_removed.append(word)

        #
        doc = nlp(' '.join(punctuation_removed))


        # Lemmatization
        doc_cleaned = [token.lemma_ for token in doc]

        return doc_cleaned

    # Encodons maintenant tous les éléments d'un dataset
    def encode(text_tensor, label):
      encoded_text = encoder.encode(text_tensor.numpy())
      return encoded_text, label

    # Utilisation du fonction py_function pour encoder tout le dataset
    def encode_map_fn(text, label):
      return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


    encoder = tfds.features.text.TokenTextEncoder.load_from_file('filename_prefix')

    model_loaded = tf.keras.models.load_model('sms_spam_detector_model.h5')

    if request.method=='POST':

        form_items = dict(request.form.items())

        texte = form_items['name']

        doc = create_token_doc(texte)
        doc_cleaned = tokens_cleaning(doc)

        # encoder = tfds.features.text.TokenTextEncoder.load_from_file('filename_prefix')

        tf_ds = create_tf_dataset(texte)

        all_encoded_data = tf_ds.map(encode_map_fn)

        texte_data = all_encoded_data.take(1)
        texte_data = texte_data.padded_batch(16,  padded_shapes=([-1], []))

        # model_loaded = tf.keras.models.load_model('sms_spam_detector_model.h5')

        # Predictions
        pred = model_loaded.predict(texte_data)
        pred = pred[0][0]

        if pred > 0.5:
            phrase = "a spam"
            pred = round(pred * 100, 2)

        else:
            phrase = "not a spam"
            pred = round(100 - pred * 100, 2)


    return render_template("result.html", result=phrase, prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
