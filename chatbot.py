#PLN - Proyecto final - 20/7/25 - estamos casi listos
import random
import json
import pickle
import numpy as np
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_UG.keras')

#Quitar las tildes que estan causando muchos problemas y el tiempo se esta acabando
def normalize_text(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def clean_up_sentences(sentence):
    sentence = normalize_text(sentence.lower())
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    index = np.argmax(res)
    confidence = res[index]

    if confidence < 0.7: #Esto es para que no responda cualquier cosa, sino que responda que no entendio
        return None  
    tag = classes[index]
    return tag

def get_response(tag, intents_json):
    list_of_intents= intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
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