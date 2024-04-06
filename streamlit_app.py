import streamlit as st
import requests
import os

# Récupérez l'URL de base de l'API à partir des variables d'environnement ou utiliser localhost par défaut
BASE_URL = os.getenv("API_URL", "https://p7app.azurewebsites.net")

st.title('Analyse de Sentiment des Tweets')

tweet_text = st.text_area("Entrez le tweet à analyser:", "Super service!")

if st.button('Predict'):
    predict_url = f"{BASE_URL}/predict/"
    response = requests.post(predict_url, json={"text": tweet_text})
    print("Status Code:", response.status_code)
    print("Response Content:", response.content)
    if response.status_code == 200:
        sentiment = response.json()['sentiment']
        st.write(f'Ce tweet est : {sentiment}')
    else:
        st.error("Erreur lors de la prédiction. Veuillez réessayer.")

if st.checkbox('La prédiction semble erronée'):
    trace_url = f"{BASE_URL}/send_trace/"
    requests.post(trace_url)
    st.write("Alerte envoyée. Merci pour votre feedback!")
