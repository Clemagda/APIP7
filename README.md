# api_p7
L'objectif de ce projet était de créer une application d'analyse de sentiment de texte déployée sur le Cloud tout en démontrant l'exploration de plusieus solutions tant en terme d'architecture(embedding, LSTM, BERT) que de préparation des données (embedding, tokenisation, lemmatisation etc.).
Cette application s'appuie sur un modèle keras déployé sur le service Azure Web App pour la partie Cloud et accessible via une interface Streamlit. 
Le déploiement a été effectué via un workflow GitHub Action afin dans une démarche CI/CD propre au MLOps. 
Toutes les dépendances nécessaires sont inscrites dans le fichier requirements.txt présent dans ce dossier. 

Structure du repertoire :
* **.github** : contient le script .yml de déploiement de l'API sur Azure Web App
* **/Models** :  contient les modèles nécessaires au fonctionnement de l'API
* **/Notebooks** : contient les notebooks de modélisation
