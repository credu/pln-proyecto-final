# PLN - Proyecto final - 20/7/25

# Importamos librerias
import random
import json
import pickle
import numpy as np
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Descargamos los paquetes de nltk que usaremos
# punk_tab & punk: tokenizador de oraciones
# wordnet: Paquete relacionado a los bancos de palabras
# omw-1.4: Banco de palabras multilenguajes
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializamos un lematizador para convertir palabras a su forma base
# Ejemplo: Bailando -> Bailar
lemmatizer = WordNetLemmatizer()

# Cargamos el archivo json que contiene las intenciones para responder prompts
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Cargamos los arreglos anteriormente
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Cargamos el modelo entrenado anteriormente
model = load_model('chatbot_UG.keras')


def normalize_text(text):
    """
    Quitar las tildes que estan causando muchos problemas y el tiempo se esta
    acabando
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def clean_up_sentences(sentence):
    # Normalizamos la oracion removiendo tildes y convirtiendolo a minusculas
    sentence = normalize_text(sentence.lower())
    # Tokenizamos la oraciones
    sentence_words = nltk.word_tokenize(sentence)
    # Normalizamos cada palabra de la oracion
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str):
    # Se procesa la oracion
    sentence_words = clean_up_sentences(sentence)
    # Llenamos un arreglo con ceros del mismo tamaño de las palabras
    # para despues llenarlo con 1 si la palabra aparece en la oracion
    bag = [0] * len(words)
    # Por cada palabra en la oracion revisamos si ya existe con el vocabulario
    # deserializado y si existe le asignamos 1 al espacio correspondiente
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence: str):
    # Obtenemos el BoW de la oracion
    bow = bag_of_words(sentence)
    # Pasamos el Bow al modelo y obtenemos una respuesta
    res = model.predict(np.array([bow]))[0]
    # Obtenemos el indice del numero mas alto en el arreglo
    index = np.argmax(res)
    # Obtenemos la confidencia del modelo
    confidence = res[index]

    # Si el nivel de confidencia no es suficiente devolvemos None
    if confidence < 0.7:
        return None

    tag = classes[index]
    return tag


def get_response(tag, intents_json):
    # Obtenemos las intenciones del json
    list_of_intents = intents_json['intents']
    result = ""
    # Iteramos cada una de las intenciones buscando la que concuerde con la tag
    for i in list_of_intents:
        if i["tag"] == tag:
            # Devolvemos cualquiera de las respuestas posibles
            result = random.choice(i['responses'])
            break
    return result

# Ejemplo de uso
# while True:
#     message = input("")
#     tag = predict_class(message)
#     if tag:
#         res = get_response(tag, intents)
#     else:
#         res = "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"
#     print(res)
