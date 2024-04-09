#!pip install -r requirements.txt

import streamlit as st
import requests
import os
import tensorflow as tf
import re
import numpy as np
from gensim.models import KeyedVectors
from opentelemetry import trace
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def preprocess_text(text):
    """
    Fonction pour le nettoyage de base du texte des tweets.
    """
    # Suppression des URL
    text = re.sub(r"http/S+|www/S+|https/S+", '', text, flags=re.MULTILINE)
    # Suppression des mentions et hashtags
    text = re.sub(r'/@/w+|/#', '', text)
    # Suppression des caractères spéciaux et numériques
    text = re.sub(r'/d+', '', text)
    text = re.sub(r'/W+', ' ', text, flags=re.MULTILINE)
    # Minuscules
    text = text.lower()
    return text
# Fonction pour générer des vecteurs moyens à partir des plongements pour chaque tweet


def vectorize_text(model, text):
    # Assuming `text` is a list of words
    vectors = [model[word]
               for word in text if word in model.key_to_index]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


model_path = "Models/model"

keras_model = tf.keras.models.load_model(model_path, compile=True)
keras_model.summary()

w2v_model = KeyedVectors.load_word2vec_format(
    "Models/model.bin.gz", binary=True)

tokenizer = tf.keras.preprocessing.text.Tokenizer()


# Création de la variable d'environnement
APPLICATIONINSIGHTS_CONNECTION_STRING = 'InstrumentationKey=4b8b585c-3050-4865-8542-6dabfe289835;IngestionEndpoint=https://francecentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://francecentral.livediagnostics.monitor.azure.com/'

# Configure OpenTelemetry to use Azure Monitor
connection_string = "InstrumentationKey=4b8b585c-3050-4865-8542-6dabfe289835;IngestionEndpoint=https://francecentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://francecentral.livediagnostics.monitor.azure.com/"
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(
        AzureMonitorTraceExporter.from_connection_string(connection_string))
)

# Get a tracer for the current module
tracer = trace.get_tracer(__name__)


st.title('Analyse de Sentiment des Tweets')

tweet_text = st.text_area("Entrez le tweet à analyser:", "Super service!")

if st.button('Predict'):
    preprocessed_text = preprocess_text(tweet_text)
    vectorized_text = vectorize_text(w2v_model, preprocessed_text.split())
    vectorized_text = np.expand_dims(vectorized_text, axis=0)
    prediction = keras_model.predict(vectorized_text)
    sentiment = 'positif' if prediction[0] > 0.5 else 'négatif'
    st.write(f'Ce tweet est : {sentiment}')

if st.checkbox('La prédiction semble erronée'):
    st.write("Alerte envoyée. Merci pour votre feedback!")
