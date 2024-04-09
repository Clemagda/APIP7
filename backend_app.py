#!pip install -r requirements.txt

# from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout
from gensim.models import Word2Vec, FastText, KeyedVectors
from opentelemetry import trace
# from azure.monitor.opentelemetry import configure_azure_monitor
import tensorflow as tf
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import re
import numpy as np


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize Precision and Recall metrics
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update precision and recall with the current batch's labels and predictions
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        # Calculate F1 score from the precision and recall values
        p = self.precision.result()
        r = self.recall.result()
        # Use harmonic mean formula for F1 calculation
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        # Reset states of precision and recall metrics at the start of each epoch
        self.precision.reset_states()
        self.recall.reset_states()




# L'URI du modèle
#directory_path = "APIP7" 
# model_path = 'G:/Mon Drive/Documents/Apprentissage/OpenClassroom/Projet_7_Analyse_de_sentiments/Models/keras_simple.keras'
model_path = "Models/baseline.keras"
# Charger le modèle

keras_model = tf.keras.models.load_model(model_path, compile=True)
keras_model.summary()
w2v_model = None
# w2v_model = Word2Vec.load(directory_path+'w2v_model.model')
#w2v_model = KeyedVectors.load_word2vec_format(
 #   "Models/model.bin.gz", binary=True)

tokenizer = tf.keras.preprocessing.text.Tokenizer()


# Fonctions de prétraitement :
def preprocess_text(text):
    """
    Fonction pour le nettoyage de base du texte des tweets.
    """
    # Suppression des URL
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Suppression des mentions et hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Suppression des caractères spéciaux et numériques
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text, flags=re.MULTILINE)
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


app = FastAPI()


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


class Tweet(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    global w2v_model
    # Modèle initialisé à None, chargement paresseux
    # model = your_model_loader_function.load()

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API de prédiction de sentiment. Utilisez le point de terminaison /predict pour analyser le sentiment."}
    
@app.post("/predict/")
async def predict_sentiment(tweet: Tweet):
    global w2v_model
    if w2v_model is None:
        w2v_model = KeyedVectors.load_word2vec_format("Models/model.bin.gz", binary=True)
    
    try:
        preprocessed_text = preprocess_text(tweet.text)
        vectorized_text = vectorize_text(w2v_model, preprocessed_text.split())
        vectorized_text = np.expand_dims(vectorized_text, axis=0)
        prediction = keras_model.predict(vectorized_text)
        sentiment = 'positif' if prediction[0] > 0.5 else 'négatif'

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

    return {"sentiment": sentiment}
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Utiliser le port défini par Azure ou, par défaut, 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

# @app.post("/send_trace/")
# def send_trace():
 #   tracer = trace.get_tracer(__name__)
  #  with tracer.start_as_current_span("feedback_incorrect"):
   #     print("Alerte envoyée. Merci pour votre feedback!")
